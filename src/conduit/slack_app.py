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


def _build_rts(client=None):
    """Live Slack history when a real bot token + client are present, else the seeded
    FakeRTS corpus so the surface still answers offline / in the demo. A fresh index is
    built per call so messages posted mid-demo are visible."""
    token = os.environ.get("SLACK_BOT_TOKEN", "")
    if client is not None and token.startswith("xoxb-") and token != "xoxb-placeholder":
        try:
            from conduit.live_rts import SlackHistoryRTS
            channels = _discover_channels(client)
            team = _team_info(client)
            if channels:
                return SlackHistoryRTS(
                    client, channels=channels, team_url=team.get("team_url", "")
                ).refresh()
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
            return OllamaLLMClient(model=os.environ.get("LORE_MODEL", "llama3.2"))
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
    from conduit.research import run, synthesize
    question = _clean_question(text)
    if not question:
        return "Ask me a question about your team's Slack history and I'll research it."
    try:
        rts = rts if rts is not None else _build_rts(client)
        llm = llm if llm is not None else _build_llm()
        result = run(question, rts, llm)
        answer = synthesize(result, llm)
        return _format_answer(answer)
    except Exception as e:  # a Slack handler must never crash the app
        logger.exception("research failed")
        return f"Sorry — research hit an error: {e}"


# --------------------------------------------------------------------------- #
# Full live orchestrator: streaming trace → Canvas → final answer
# --------------------------------------------------------------------------- #
def _create_canvas(client: Any, answer: Any, question: str, channel: str, graph: Any = None) -> str:
    """Create a Canvas report, share it read-only with the channel, and return its URL
    (empty string if the Canvas API is unavailable)."""
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
        try:
            client.canvases_access_set(
                canvas_id=canvas_id, access_level="read", channel_ids=[channel]
            )
        except Exception:
            logger.warning("canvases.access.set failed — judges may lack canvas access", exc_info=True)
        team = _team_info(client)
        base, tid = team.get("team_url", ""), team.get("team_id", "")
        if base and tid:
            return f"{base}/docs/{tid}/{canvas_id}"
        return ""  # can't build a real URL — callers omit the button rather than link a bare id
    except Exception:
        logger.exception("canvas creation failed")
        return ""


def research_and_respond(
    client: Any,
    channel: str,
    thread_ts: Optional[str],
    question: str,
    *,
    is_assistant: bool = False,
) -> Optional[str]:
    """The money-shot path. Streams a live research trace, builds a cited Canvas, shares it,
    and posts the final answer. Returns the Canvas URL (or None). Never raises."""
    from conduit.research import run, synthesize
    from conduit.assistant_surface import ResearchAssistant, AssistantContext

    q = _clean_question(question)
    if not q:
        client.chat_postMessage(
            channel=channel, thread_ts=thread_ts,
            text="Ask me a question about your team's Slack history and I'll research it.",
        )
        return None

    assistant: Optional[ResearchAssistant] = None
    if is_assistant and thread_ts:
        assistant = ResearchAssistant(
            client, AssistantContext(channel=channel, thread_ts=thread_ts), stream=True,
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
            if assistant is not None:
                assistant.set_status("")
            client.chat_postMessage(channel=channel, thread_ts=thread_ts,
                                    blocks=build_empty_state_blocks(q, channels),
                                    text="No relevant history found.")
            return None

        canvas_url = _create_canvas(client, answer, q, channel, graph=getattr(result, "graph", None))
        if assistant is not None:
            assistant.set_status("")  # clear the thinking indicator
            assistant.post_result(answer, canvas_url or "")
        else:
            from conduit.blocks import build_answer_blocks, final_block
            blocks = build_answer_blocks(_format_answer(answer))
            if canvas_url:
                blocks += final_block(answer.text[:280], canvas_url)
            client.chat_postMessage(channel=channel, thread_ts=thread_ts,
                                    blocks=blocks, text=answer.text[:2000])
        return canvas_url
    except Exception as e:
        logger.exception("live research failed")
        try:
            from conduit.blocks import build_error_blocks
            if assistant is not None:
                assistant.set_status("")
            client.chat_postMessage(channel=channel, thread_ts=thread_ts,
                                    blocks=build_error_blocks(str(e)),
                                    text="Research hit an error.")
        except Exception:
            pass
        return None


# --------------------------------------------------------------------------- #
# Bolt event handlers
# --------------------------------------------------------------------------- #
def handle_mention(body, event, say, client, logger=logger):
    event_id = body.get("event_id") or body.get("event", {}).get("client_msg_id", "")
    if _DEDUP.is_seen(event_id):
        logger.debug("duplicate event %s — skipping", event_id)
        return
    text = event.get("text", "")
    say(handle_query(text, client=client))


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
    say(handle_query(text, client=client))


def handle_lore(body, ack, say, client, logger=logger, respond=None):
    ack()
    # Slash commands are never redelivered by Slack, so keying dedup on the (always-empty)
    # event_id would drop every later /lore. Key on the unique trigger_id instead.
    event_id = body.get("trigger_id", "")
    if event_id and _DEDUP.is_seen(event_id):
        logger.debug("duplicate command %s — skipping", event_id)
        return
    text = body.get("text", "")
    answer = handle_query(text, client=client)
    (respond or say)(answer)


def handle_app_home_opened(event, client, logger=logger):
    """Publish the Lore home tab when a user opens the app's Home."""
    if event.get("tab") != "home":
        return
    try:
        from conduit.blocks import build_lore_home_view
        client.views_publish(user_id=event["user"], view=build_lore_home_view())
    except Exception:
        logger.exception("home publish failed")


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
    research_and_respond(client, channel, thread_ts, payload.get("text", ""), is_assistant=True)


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
    app.event("app_mention")(handle_mention)
    app.event("message")(handle_thread_message)
    app.command("/lore")(handle_lore)
    app.event("app_home_opened")(handle_app_home_opened)
    app.action("view_canvas")(handle_view_canvas_action)

    # Assistant split-view (Agents & AI Apps). Optional — only if this Bolt version has it.
    try:
        from slack_bolt import Assistant
        assistant = Assistant()
        assistant.thread_started(assistant_thread_started)
        assistant.user_message(assistant_user_message)
        app.use(assistant)
    except Exception:
        logger.info("Assistant middleware unavailable in this slack_bolt version — "
                    "assistant split-view disabled, mention/command paths still work")
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
