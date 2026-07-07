"""Slack Bolt application — the live surface for Lore.

Two layers:

* ``handle_query`` — the pure, text-only research path (used by tests and as a fallback).
* ``research_and_respond`` — the full live orchestrator: streams a research trace into the
  assistant split-view, writes a cited **Canvas** report, shares it with the channel, and
  posts the final answer with a "View Canvas" button.

The module is import-safe with no environment: the Bolt ``App`` / ``Assistant`` are built
inside a guarded block, so ``import conduit.slack_app`` works in CI with no tokens. The live
process is started with ``python -m conduit.slack_app`` (Socket Mode — no public URL needed).
"""
from __future__ import annotations

import logging
import os
import re
import sys
from typing import Any, Optional

from conduit.dedup import EventDedup

logger = logging.getLogger(__name__)
_DEDUP = EventDedup()

# Cached workspace identity from auth.test (team url + id), for building Canvas deep-links.
_TEAM: dict[str, str] = {}

_MENTION_RE = re.compile(r"<@[A-Z0-9]+>")


# --------------------------------------------------------------------------- #
# Backend construction (live Slack + local model, with honest offline fallbacks)
# --------------------------------------------------------------------------- #
def _team_info(client: Any) -> dict[str, str]:
    """Cache ``auth.test`` → {team_url, team_id, bot_user_id}. Best-effort, called once."""
    if _TEAM or client is None:
        return _TEAM
    try:
        resp = client.auth_test()
        _TEAM["team_url"] = (resp.get("url") or "").rstrip("/")
        _TEAM["team_id"] = resp.get("team_id") or ""
        _TEAM["bot_user_id"] = resp.get("user_id") or ""
    except Exception:
        logger.exception("auth.test failed — Canvas deep-links may be degraded")
    return _TEAM


def _discover_channels(client: Any) -> dict[str, str]:
    """Channels to index for research: ``LORE_CHANNELS`` env override (``C123:name,...``),
    else every public/private channel the bot is a member of (via conversations.list)."""
    override = os.environ.get("LORE_CHANNELS", "").strip()
    if override:
        out: dict[str, str] = {}
        for part in override.split(","):
            part = part.strip()
            if not part:
                continue
            cid, _, name = part.partition(":")
            out[cid.strip()] = (name.strip() or cid.strip())
        return out
    channels: dict[str, str] = {}
    if client is None:
        return channels
    try:
        cursor = None
        while True:
            resp = client.conversations_list(
                types="public_channel,private_channel",
                exclude_archived=True,
                limit=200,
                **({"cursor": cursor} if cursor else {}),
            )
            for c in resp.get("channels", []) or []:
                if c.get("is_member"):
                    channels[c["id"]] = c.get("name", c["id"])
            cursor = (resp.get("response_metadata") or {}).get("next_cursor")
            if not cursor:
                break
    except Exception:
        logger.exception("conversations.list failed — set LORE_CHANNELS to name channels explicitly")
    return channels


_RTS_CACHE: dict = {}  # channels-key -> (rts, built_at)


def _build_rts(client=None):
    """Pick the retrieval backend behind the shared ``search()`` seam, honestly, by what's
    configured:

    1. **Official Slack Search API** (``RTSClient`` → ``search.messages``) when a USER token
       (``SLACK_USER_TOKEN``, ``xoxp-…`` with ``search:read``) is present and
       ``LORE_USE_RTS_API`` is on — Slack's first-party search.
    2. **SlackHistoryRTS** (``conversations.history`` + local lexical/recency ranker) when a
       real bot token + client are present — the default live backend (bot tokens can't call
       ``search.messages``, so this is what runs in the sandbox).
    3. **FakeRTS** seeded corpus otherwise, so the surface still answers offline / in the demo.

    The SlackHistoryRTS index is cached briefly (``LORE_INDEX_TTL``, default 120s) so
    back-to-back queries don't re-read every channel's history — the main per-query latency."""
    # (1) Official Slack Search API — opt-in, requires a user token with search:read.
    user_token = os.environ.get("SLACK_USER_TOKEN", "")
    use_rts_api = os.environ.get("LORE_USE_RTS_API", "").strip().lower() in {"1", "true", "yes", "on"}
    if use_rts_api and user_token.startswith("xoxp-"):
        try:
            from conduit.rts_client import RTSClient
            logger.info("retrieval backend: official Slack Search API (search.messages, user token)")
            return RTSClient(token=user_token)
        except Exception:
            logger.exception("RTS API backend unavailable — falling back to SlackHistoryRTS/FakeRTS")

    token = os.environ.get("SLACK_BOT_TOKEN", "")
    if client is not None and token.startswith("xoxb-") and token != "xoxb-placeholder":
        try:
            import time
            from conduit.live_rts import SlackHistoryRTS
            channels = _discover_channels(client)
            if channels:
                key = tuple(sorted(channels))
                # Short TTL: back-to-back queries in a burst still reuse the index (fast), but a
                # message a judge posts mid-demo becomes visible within seconds instead of up to
                # two minutes. Set LORE_INDEX_TTL=0 to rebuild on every query (always-fresh).
                ttl = float(os.environ.get("LORE_INDEX_TTL", "20"))
                cached = _RTS_CACHE.get(key)
                if cached and (time.time() - cached[1]) < ttl:
                    return cached[0]
                team = _team_info(client)
                rts = SlackHistoryRTS(
                    client, channels=channels, team_url=team.get("team_url", "")
                ).refresh()
                _RTS_CACHE[key] = (rts, time.time())
                return rts
            logger.warning("no member channels found — invite Lore to channels; using FakeRTS")
        except Exception:
            logger.exception("live RTS unavailable — falling back to FakeRTS")
    from conduit.fake_rts import FakeRTS
    return FakeRTS()


def _index_channel_names(rts: Any) -> list[str]:
    """Best-effort list of channel names the RTS layer indexed (for the empty-state hint)."""
    try:
        names = getattr(rts, "_channel_names", None)
        if isinstance(names, dict) and names:
            return list(names.values())
    except Exception:
        pass
    return []


def _live_mode() -> bool:
    token = os.environ.get("SLACK_BOT_TOKEN", "")
    return token.startswith("xoxb-") and token != "xoxb-placeholder"


def _build_llm():
    """A local Ollama model when configured, else a deterministic fake so the pipeline
    still runs offline. In live mode a missing model is loud (the demo must not silently
    answer from canned responses)."""
    if os.environ.get("OLLAMA_API_BASE") or os.environ.get("LORE_USE_OLLAMA"):
        try:
            from conduit.agent import OllamaLLMClient
            # A 35B model doing several research calls (and a cold first load) needs a
            # generous timeout — 30s (the default) reliably times out.
            timeout = float(os.environ.get("LORE_LLM_TIMEOUT", "180"))
            return OllamaLLMClient(model=os.environ.get("LORE_MODEL", "llama3.2"), timeout=timeout)
        except Exception:
            logger.exception("Ollama LLM unavailable — falling back to deterministic LLM")
    if _live_mode():
        logger.warning(
            "LIVE mode but no OLLAMA_API_BASE/LORE_USE_OLLAMA set — using the deterministic "
            "FakeLLMClient. Set OLLAMA_API_BASE to research with a real model."
        )
    from conduit.agent import FakeLLMClient
    return FakeLLMClient()


def _clean_question(text: str) -> str:
    """Strip ``<@U…>`` mention markup and surrounding whitespace so it doesn't pollute
    keyword tokenization."""
    return _MENTION_RE.sub("", text or "").strip()


# --------------------------------------------------------------------------- #
# Text-only path (tests + fallback)
# --------------------------------------------------------------------------- #
def _format_answer(answer) -> str:
    """Render an Answer (text + citations) as Slack-friendly text with deep-links."""
    parts = [answer.text]
    if getattr(answer, "citations", None):
        parts.append("")
        for c in answer.citations:
            link = getattr(c, "permalink", "") or ""
            ch = getattr(c, "channel", "") or ""
            parts.append(f"[{c.index}] <{link}|#{ch}>" if link else f"[{c.index}] #{ch}")
    return "\n".join(parts)


def handle_query(text: str, client=None, rts=None, llm=None) -> str:
    """Run the REAL research pipeline (RTS multi-hop → citation synthesis → deterministic
    contradiction/timeline resolution) and return a Slack-formatted answer. Uses live Slack
    history + a local model by default; ``rts``/``llm`` are injectable for tests + the
    assistant surface."""
    import time
    from conduit.research import run, synthesize
    question = _clean_question(text)
    if not question:
        return "Ask me a question about your team's Slack history and I'll research it."
    t0 = time.time()
    logger.info("QUERY start: %r", question[:120])
    try:
        rts = rts if rts is not None else _build_rts(client)
        llm = llm if llm is not None else _build_llm()
        result = run(question, rts, llm)
        answer = synthesize(result, llm)
        cites = len(getattr(answer, "citations", []) or [])
        drift = getattr(answer, "drift", None)
        logger.info("QUERY done in %.1fs: %d evidence, %d citations%s",
                    time.time() - t0, len(result.evidence), cites,
                    f", drift {drift.old_value}->{drift.current_value}" if drift else "")
        return _format_answer(answer)
    except Exception:  # a Slack handler must never crash the app
        # Log the full exception server-side, but never echo the raw message (which can carry
        # endpoint URLs / hosts / IDs) into a channel-visible reply.
        logger.exception("research failed")
        return "Sorry — research hit an error. I've logged the details; please try again."


# --------------------------------------------------------------------------- #
# Full live orchestrator: streaming trace → Canvas → final answer
# --------------------------------------------------------------------------- #
def _create_canvas(client: Any, answer: Any, question: str, channel: str,
                   graph: Any = None, user_id: str = "") -> str:
    """Create a Canvas report, share it read-only, and return its URL — or ``""`` so callers omit
    the "View Canvas" button rather than linking a doc the viewer can't open.

    A bot-owned standalone canvas is invisible until shared, so the URL is returned ONLY when a
    read grant actually succeeds. On a public/private channel (id starts ``C``/``G``) we share to
    the channel; on a DM / Assistant container (not a shareable channel) we grant the invoking
    ``user_id`` directly — otherwise a judge on the Assistant surface would click through to an
    access-denied page (a visible failure of a headline deliverable)."""
    from conduit.canvas import build_report_markdown
    try:
        markdown = build_report_markdown(answer, question, graph=graph)
        resp = client.canvases_create(
            title=f"Lore — {question[:70]}",
            document_content={"type": "markdown", "markdown": markdown},
        )
        canvas_id = resp.get("canvas_id") or resp.get("canvas", {}).get("id", "")
        if not canvas_id:
            return ""
        team = _team_info(client)
        base, tid = team.get("team_url", ""), team.get("team_id", "")
        if not (base and tid):
            return ""  # can't build a real URL — omit the button rather than link a bare id

        granted = False
        if channel and channel[:1] in ("C", "G"):
            try:
                client.canvases_access_set(
                    canvas_id=canvas_id, access_level="read", channel_ids=[channel]
                )
                granted = True
            except Exception:
                logger.warning("canvases.access.set (channel) failed", exc_info=True)
        if user_id:
            try:
                client.canvases_access_set(
                    canvas_id=canvas_id, access_level="read", user_ids=[user_id]
                )
                granted = True
            except Exception:
                logger.warning("canvases.access.set (user) failed", exc_info=True)
        if not granted:
            # Nobody was granted read access → don't render a button to a canvas they can't open.
            logger.warning("no canvas access grant succeeded — omitting the View-Canvas button")
            return ""
        return f"{base}/docs/{tid}/{canvas_id}"
    except Exception:
        logger.exception("canvas creation failed")
        return ""


def _stream_enabled() -> bool:
    """Whether to stream the live research trace on non-assistant surfaces (/lore, @mention,
    DM). Default on — the streaming trace is a headline feature and should be visible however a
    judge invokes Lore. Set ``LORE_STREAM_TRACE=0`` for a quieter, answer-only reply."""
    return os.environ.get("LORE_STREAM_TRACE", "1").strip().lower() in {"1", "true", "yes", "on"}


def _post_kwargs(channel: str, thread_ts: Optional[str]) -> dict:
    """chat_postMessage kwargs, omitting an empty thread_ts (Slack rejects ``thread_ts=""``)."""
    kw: dict[str, Any] = {"channel": channel}
    if thread_ts:
        kw["thread_ts"] = thread_ts
    return kw


def research_and_respond(
    client: Any,
    channel: str,
    thread_ts: Optional[str],
    question: str,
    *,
    is_assistant: bool = False,
    user_id: str = "",
) -> Optional[str]:
    """The money-shot path, now shared by EVERY surface (assistant, /lore, @mention, DM):
    streams a live research trace, builds a cited **Canvas**, shares it, and posts a rich
    Block Kit answer (Decision-Graph badge → decision timeline → conflicting-signals →
    cited answer → View-Canvas button). Never raises.

    Return value signals **delivery**, so callers can add an ephemeral fallback: a ``str`` (the
    Canvas URL, or ``""``) when a reply reached the user — including the empty-state and error
    cards — and ``None`` **only** when nothing could be posted at all (every post attempt failed).

    ``is_assistant`` selects the real Assistant split-view (uses ``assistant.threads.setStatus``
    and posts into the assistant thread). On other surfaces the trace streams as an in-place
    edited channel/thread message instead — same visible research, no assistant container.
    ``user_id`` (the invoker) is used to grant Canvas read access on DM / Assistant surfaces."""
    from conduit.research import run, synthesize
    from conduit.assistant_surface import ResearchAssistant, AssistantContext

    q = _clean_question(question)
    if not q:
        # Inside try: this function documents "Never raises" and callers depend on it — a
        # transient post failure here must not escape into Bolt as an unhandled listener error.
        try:
            client.chat_postMessage(
                **_post_kwargs(channel, thread_ts),
                text="Ask me a question about your team's Slack history and I'll research it.",
            )
            return ""  # delivered the prompt
        except Exception:
            logger.exception("failed to post empty-question prompt")
            return None  # nothing reached the user

    # Stream on the assistant surface always; on other surfaces when LORE_STREAM_TRACE is on.
    stream = is_assistant or _stream_enabled()
    assistant = ResearchAssistant(
        client,
        AssistantContext(channel=channel, thread_ts=thread_ts or ""),
        stream=stream,
        assistant_container=is_assistant,
    )

    try:
        rts = _build_rts(client)
        llm = _build_llm()
        result = run(q, rts, llm, assistant=assistant)
        answer = synthesize(result, llm)

        # Empty-state: no evidence found → a helpful Block Kit reply, not a bare sentence.
        if not result.evidence:
            from conduit.blocks import build_empty_state_blocks
            channels = _index_channel_names(rts)
            assistant.set_status("")
            client.chat_postMessage(**_post_kwargs(channel, thread_ts),
                                    blocks=build_empty_state_blocks(q, channels),
                                    text="No relevant history found.")
            return ""  # delivered the empty-state card

        canvas_url = _create_canvas(client, answer, q, channel,
                                    graph=getattr(result, "graph", None), user_id=user_id)
        assistant.set_status("")  # clear the thinking indicator (no-op off the assistant surface)
        assistant.post_result(answer, canvas_url or "", graph=getattr(result, "graph", None), question=q)
        return canvas_url or ""  # delivered the answer (always a str, so callers know it landed)
    except Exception as e:
        # Log the full exception server-side, but surface only the exception CLASS name to the
        # channel (never the raw message, which can leak endpoint URLs / hosts / channel IDs).
        logger.exception("live research failed")
        try:
            from conduit.blocks import build_error_blocks
            assistant.set_status("")
            client.chat_postMessage(**_post_kwargs(channel, thread_ts),
                                    blocks=build_error_blocks(type(e).__name__),
                                    text="Research hit an error.")
            return ""  # delivered an error card
        except Exception:
            pass
        return None  # nothing reached the user


# --------------------------------------------------------------------------- #
# Bolt event handlers
# --------------------------------------------------------------------------- #
def handle_mention(body, event, say, client, logger=logger):
    event_id = body.get("event_id") or body.get("event", {}).get("client_msg_id", "")
    if _DEDUP.is_seen(event_id):
        logger.debug("duplicate event %s — skipping", event_id)
        return
    text = event.get("text", "")
    from conduit.notify import notify_usage
    notify_usage("@mention", user=event.get("user", ""), text=text,
                 channel=event.get("channel", ""), client=client)
    _publish_home(client, event.get("user", ""))  # populate their Home the first time we see them
    thread_ts = event.get("thread_ts") or event.get("ts")
    # Full money-shot in the thread: streaming trace → cited Canvas → decision timeline.
    # (The streamed trace posts within ~1s, so it doubles as the "researching…" feedback.)
    research_and_respond(client, event.get("channel", ""), thread_ts, text,
                         user_id=event.get("user", ""))


def handle_thread_message(body, event, say, client, logger=logger):
    # Ignore bot echoes, edits/joins/other subtypes, and channel chatter that isn't a
    # direct message to Lore — otherwise the bot answers every message in every channel.
    if event.get("bot_id") or event.get("subtype"):
        return
    if event.get("channel_type") not in ("im",):
        return
    event_id = body.get("event_id") or event.get("client_msg_id", "")
    if _DEDUP.is_seen(event_id):
        logger.debug("duplicate event %s — skipping", event_id)
        return
    text = event.get("text", "")
    _publish_home(client, event.get("user", ""))
    # DMs get the same money-shot (streaming trace + cited Canvas + timeline) as every surface.
    research_and_respond(client, event.get("channel", ""), event.get("thread_ts"), text,
                         user_id=event.get("user", ""))


def handle_lore(body, ack, say, client, logger=logger, respond=None):
    ack()
    # Slash commands are never redelivered by Slack, so keying dedup on the (always-empty)
    # event_id would drop every later /lore. Key on the unique trigger_id instead.
    event_id = body.get("trigger_id", "")
    if event_id and _DEDUP.is_seen(event_id):
        logger.debug("duplicate command %s — skipping", event_id)
        return
    text = body.get("text", "")
    channel = body.get("channel_id", "")
    from conduit.notify import notify_usage
    notify_usage("/lore", user=body.get("user_id", ""), text=text,
                 channel=body.get("channel_name", ""), client=client)
    _publish_home(client, body.get("user_id", ""))
    # Interim feedback (ephemeral, only the invoker sees it) — the public streaming trace +
    # cited Canvas answer are posted by research_and_respond below.
    try:
        (respond or say)("🔎 Researching your question across the workspace…")
    except Exception:
        pass
    # Full money-shot in-channel: streaming trace → cited Canvas → decision timeline.
    delivered = research_and_respond(client, channel, None, text, user_id=body.get("user_id", ""))
    # If NOTHING reached the channel (e.g. a private channel Lore isn't a member of, where even
    # chat:write.public can't post), fall back to an ephemeral answer only the invoker sees — so
    # the judge always gets the result instead of just the interim "Researching…". `delivered` is
    # a str ("" included) whenever a card posted, and None only when every post attempt failed, so
    # this never double-posts on the empty-state / error paths.
    if delivered is None:
        try:
            (respond or say)(handle_query(text, client=client))
        except Exception:
            pass


_HOME_PUBLISHED: set[str] = set()


def _publish_home(client: Any, user_id: str, *, force: bool = False) -> None:
    """Best-effort publish of the Lore App Home for a user. Because ``app_home_opened`` isn't
    guaranteed to reach the app (it must be subscribed at install time), we ALSO publish the home
    proactively the first time a user interacts — so the Home tab is populated (not Slack's default
    placeholder) regardless. Idempotent per process (unless ``force``) so it never spams
    views.publish; never raises."""
    if not user_id or client is None:
        return
    if not force and user_id in _HOME_PUBLISHED:
        return
    try:
        from conduit.blocks import build_lore_home_view
        _HOME_PUBLISHED.add(user_id)  # set first so a failure doesn't retry-storm on every event
        client.views_publish(user_id=user_id, view=build_lore_home_view())
    except Exception:
        logger.debug("home publish failed for %s", user_id, exc_info=True)


def handle_app_home_opened(event, client, logger=logger):
    """Publish the Lore home tab when a user opens the app's Home."""
    if event.get("tab") != "home":
        return
    _publish_home(client, event.get("user", ""), force=True)


def handle_home_ask(ack, body, client, logger=logger):
    """A Home example-question button was clicked → DM the asker a cited answer.

    Acks immediately, opens a DM, posts an interim line, then runs the full research on a
    background thread so the Bolt worker isn't blocked for the length of a research run."""
    ack()
    try:
        user = (body.get("user") or {}).get("id", "")
        actions = body.get("actions") or []
        question = (actions[0].get("value") if actions else "") or ""
        if not user or not question:
            return
        im = client.conversations_open(users=user)
        dm = (im.get("channel") or {}).get("id", "")
        if not dm:
            return
        try:
            client.chat_postMessage(channel=dm, text=f"🔎 Researching: *{question}*")
        except Exception:
            pass
        import threading
        threading.Thread(
            target=lambda: research_and_respond(client, dm, None, question, user_id=user),
            name="home-ask", daemon=True,
        ).start()
    except Exception:
        logger.exception("home_ask handler failed")


def handle_view_canvas_action(ack, logger=logger):
    """No-op ack for the 'View Canvas' link button so Slack doesn't warn."""
    ack()


# --------------------------------------------------------------------------- #
# Assistant (split-view) handlers — wired to Bolt's Assistant middleware
# --------------------------------------------------------------------------- #
def assistant_thread_started(payload, set_suggested_prompts, say, logger=logger):
    """Greet + populate suggested starter prompts when a user opens the Lore assistant."""
    from conduit.assistant_surface import suggested_prompts
    try:
        say("Hi! I'm *Lore* — I research your team's Slack history and answer with cited, "
            "deep-linked sources. Ask me anything, or try one of these:")
        prompts = suggested_prompts(payload.get("channel_id", ""))
        set_suggested_prompts(prompts=prompts, title="Research your team's memory")
    except Exception:
        logger.exception("assistant thread_started failed")


def assistant_user_message(payload, client, context, logger=logger):
    """Run the full streaming-trace + Canvas orchestrator for an assistant message."""
    channel = payload.get("channel") or context.get("channel_id", "")
    thread_ts = payload.get("thread_ts") or payload.get("ts", "")
    from conduit.notify import notify_usage
    notify_usage("assistant", user=payload.get("user", ""), text=payload.get("text", ""),
                 channel=channel, client=client)
    _publish_home(client, payload.get("user", ""))
    research_and_respond(client, channel, thread_ts, payload.get("text", ""),
                         is_assistant=True, user_id=payload.get("user", ""))


# --------------------------------------------------------------------------- #
# App wiring (guarded so import succeeds with no tokens / in CI)
# --------------------------------------------------------------------------- #
def build_app():
    """Construct and wire the Bolt App. Returns None if slack_bolt isn't importable."""
    try:
        from slack_bolt import App
    except ImportError:
        return None

    # token_verification_enabled=False so construction never blocks on a live auth.test —
    # the module must import in CI with a placeholder token; Socket Mode auth happens at start.
    app = App(
        token=os.environ.get("SLACK_BOT_TOKEN", "xoxb-placeholder"),
        signing_secret=os.environ.get("SLACK_SIGNING_SECRET", "placeholder-signing-secret"),
        token_verification_enabled=False,
    )

    # Log every incoming request so it's visible whether events reach the handlers.
    @app.middleware
    def _log_incoming(body, next, logger=logger):
        try:
            kind = (body.get("command") or (body.get("event") or {}).get("type")
                    or body.get("type") or "?")
            user = body.get("user_id") or (body.get("event") or {}).get("user") or ""
            channel = (body.get("channel_id") or (body.get("event") or {}).get("channel")
                       or (body.get("channel") or {}).get("id", "") if isinstance(body.get("channel"), dict)
                       else body.get("channel", ""))
            logger.info("INCOMING: %s (user=%s channel=%s)", kind, user, channel)
        except Exception:
            pass
        return next()

    app.event("app_mention")(handle_mention)
    app.event("message")(handle_thread_message)
    app.command("/lore")(handle_lore)
    app.event("app_home_opened")(handle_app_home_opened)
    app.action("view_canvas")(handle_view_canvas_action)
    app.action(re.compile(r"^home_ask"))(handle_home_ask)  # all Home example buttons route here

    # Assistant split-view (Agents & AI Apps). Attach via app.assistant() (NOT app.use()) so
    # Bolt routes assistant-thread events to these handlers.
    try:
        from slack_bolt import Assistant
        assistant = Assistant()
        assistant.thread_started(assistant_thread_started)
        assistant.user_message(assistant_user_message)
        app.assistant(assistant)
    except Exception:
        logger.info("Assistant middleware unavailable in this slack_bolt version — "
                    "assistant split-view disabled, mention/command paths still work",
                    exc_info=True)
    return app


def main() -> int:
    logging.basicConfig(level=os.environ.get("LORE_LOG_LEVEL", "INFO"))
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    app = build_app()
    if app is None:
        logger.error("slack_bolt not installed — `pip install -e .` first")
        return 1
    app_token = os.environ.get("SLACK_APP_TOKEN", "")
    if not app_token.startswith("xapp-"):
        logger.error("SLACK_APP_TOKEN (xapp-…) required for Socket Mode — see .env.example")
        return 1
    # Warm the local model in the background so the first real query isn't a cold-start.
    def _warm():
        try:
            _build_llm().chat([{"role": "user", "content": "ping"}])
            logger.info("model warm — ready for fast responses")
        except Exception:
            logger.info("model warmup skipped/failed (will load on first query)", exc_info=True)
    import threading
    threading.Thread(target=_warm, daemon=True).start()

    # Proactively publish the App Home for existing members of the channels Lore is in, so the Home
    # tab shows the rich onboarding surface (not Slack's default placeholder) even for a judge who
    # opens it without ever messaging Lore — app_home_opened isn't guaranteed to reach the app.
    def _populate_homes():
        try:
            client = app.client
            published = 0
            for cid in list(_discover_channels(client))[:20]:
                try:
                    members = (client.conversations_members(channel=cid, limit=100).get("members") or [])
                except Exception:
                    continue
                for uid in members:
                    if uid and uid not in _HOME_PUBLISHED and published < 50:
                        _publish_home(client, uid)
                        published += 1
            logger.info("App Home pre-published for %d workspace member(s)", published)
        except Exception:
            logger.debug("home pre-population skipped", exc_info=True)
    threading.Thread(target=_populate_homes, daemon=True).start()

    from slack_bolt.adapter.socket_mode import SocketModeHandler
    logger.info("Lore starting in Socket Mode…")
    SocketModeHandler(app, app_token).start()
    return 0


# Module-level app for import-time consumers/tests; None if slack_bolt missing.
try:
    app = build_app()
except Exception:
    app = None


if __name__ == "__main__":
    sys.exit(main())
