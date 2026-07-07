"""Assistant split-view streaming trace for research results."""
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from conduit.citations import Answer


def _extract_ts(resp: Any) -> Optional[str]:
    """Pull the message ``ts`` from a Slack response (dict or SlackResponse)."""
    try:
        if resp is None:
            return None
        if isinstance(resp, dict):
            return resp.get("ts")
        return resp["ts"]  # SlackResponse supports __getitem__
    except (KeyError, TypeError):
        return None


# Starter prompts shown in the assistant split-view before the user types. One is
# newcomer-framed on purpose — it doubles as on-screen "Agent for Good" evidence
# (knowledge equity: a day-one hire gets the same cited answer as a 5-year veteran).
DEFAULT_SUGGESTED_PROMPTS: list[dict[str, str]] = [
    {"title": "I'm new here — what's the story behind our pricing?",
     "message": "What did we decide about pricing, and did anything change since?"},
    {"title": "Summarise recent decisions",
     "message": "What decisions did we make in the last two weeks?"},
    {"title": "How does the deployment pipeline work?",
     "message": "How does the deployment pipeline work?"},
    {"title": "Have we changed our policy on API versioning?",
     "message": "Have we changed our policy on API versioning?"},
]


def suggested_prompts(channel_id: str = "") -> list[dict[str, str]]:
    """Return suggested starter prompts for the assistant thread (optionally per-channel)."""
    return list(DEFAULT_SUGGESTED_PROMPTS)


class SlackClient(Protocol):
    """Protocol for Slack API client methods used by ResearchAssistant."""
    
    def assistant_threads_setStatus(
        self,
        *,
        channel_id: str,
        thread_ts: str,
        status: str
    ) -> Any: ...
    
    def chat_postMessage(
        self,
        *,
        channel: str,
        thread_ts: str,
        blocks: list,
        text: str
    ) -> Any: ...

    def chat_update(
        self,
        *,
        channel: str,
        ts: str,
        blocks: list,
        text: str
    ) -> Any: ...


@dataclass
class AssistantContext:
    """Context for the assistant's current conversation."""
    channel: str
    thread_ts: str


@dataclass
class ResearchAssistant:
    """Assistant for streaming research trace to Slack split-view.

    When ``stream=True`` (live use), each trace step is rendered into ONE Slack message that
    is edited in place (post once, ``chat_update`` thereafter) — so the user watches the
    research unfold ("🔍 Decomposing… 🔎 Searching #pricing → 4 hits… ✅ cross-checking…")
    instead of a spinner followed by a dump. In tests it defaults to ``stream=False`` (buffer
    only), so the trace is inspectable without any Slack calls.
    """

    client: SlackClient
    context: AssistantContext
    stream: bool = False
    # True only in the real Assistant split-view, where ``assistant.threads.setStatus`` is a
    # valid call. On a plain channel/thread surface (``/lore``, ``@mention``, DM) that API
    # rejects the request, so we skip it and rely on the streamed trace message instead.
    assistant_container: bool = True
    _trace: list[str] = field(default_factory=list, init=False)
    _trace_ts: Optional[str] = field(default=None, init=False)
    _trace_blocks: list = field(default_factory=list, init=False)
    _posted: bool = field(default=False, init=False)          # have we attempted the first post?
    _stream_disabled: bool = field(default=False, init=False)  # give up streaming (buffer only)

    def _msg_kwargs(self) -> dict:
        """Base ``chat_postMessage`` kwargs — include ``thread_ts`` only when we actually have a
        thread (a ``/lore`` slash command has none, and Slack rejects an empty ``thread_ts``)."""
        kw: dict[str, Any] = {"channel": self.context.channel}
        if self.context.thread_ts:
            kw["thread_ts"] = self.context.thread_ts
        return kw

    def set_status(self, status: str) -> None:
        """Update the split-view thinking indicator (one API call). Defensive: a failed status
        update (e.g. outside a real assistant container) must never break the research run."""
        if not self.assistant_container:
            return  # not in an Assistant split-view — setStatus would 400; the trace covers it
        try:
            self.client.assistant_threads_setStatus(
                channel_id=self.context.channel,
                thread_ts=self.context.thread_ts,
                status=status
            )
        except Exception:
            import logging
            logging.getLogger(__name__).debug("set_status failed", exc_info=True)

    def emit_trace(self, phase: str, detail: str) -> None:
        """Record a trace step, and (when streaming) reflect it live in Slack."""
        self._trace.append(f"{phase}: {detail}")
        if self.stream:
            self._stream_step(phase, detail)

    def _stream_step(self, phase: str, detail: str) -> None:
        """Render one trace step into the single, edited-in-place research message.

        Posts exactly once, then edits that message in place. If the first post's ``ts`` can't be
        captured (unparseable response, or the post raised after Slack accepted it), streaming is
        disabled for the rest of the run — otherwise every subsequent step would post a NEW card
        and flood the channel/thread. ``_posted`` is set BEFORE the post attempt so a raised
        exception can never cause a re-post."""
        from conduit.blocks import trace_block, TraceStep
        if self._stream_disabled:
            return
        try:
            if not self._trace_blocks:
                self._trace_blocks.append({
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": "🔦 *Researching…*"},
                })
            self._trace_blocks.append(trace_block(TraceStep(phase, detail)))
            blocks = self._trace_blocks[-48:]  # stay under Slack's 50-block cap
            if not self._posted:
                self._posted = True  # set first: never post a second card even if this raises
                resp = self.client.chat_postMessage(
                    **self._msg_kwargs(), blocks=blocks, text="Researching…",
                )
                self._trace_ts = _extract_ts(resp)
                if self._trace_ts is None:
                    # no message id to edit → stop streaming rather than flooding with new cards
                    self._stream_disabled = True
            elif self._trace_ts is not None:
                self.client.chat_update(
                    channel=self.context.channel, ts=self._trace_ts,
                    blocks=blocks, text="Researching…",
                )
        except Exception:  # never let a UI update break the research
            import logging
            logging.getLogger(__name__).debug("trace stream update failed", exc_info=True)

    @property
    def trace_log(self) -> list[str]:
        """Copy of the accumulated trace lines, in order."""
        return self._trace.copy()
    
    def post_result(self, answer: Answer, canvas_url: str,
                    graph: Any = None, question: str = "") -> Any:
        """Post the final result card: the money-shot (Decision-Graph badge → decision timeline
        → conflicting-signals) then the cited answer and a "View Canvas" button.

        Args:
            answer: The synthesized answer with citations, drift, and graph_summary.
            canvas_url: URL to the Canvas report (button omitted when empty).
            graph: The KnowledgeGraph, so the decision timeline can be rendered inline.
            question: The original question (drives the graph's primary-topic timeline).

        Returns:
            The result of chat_postMessage.
        """
        from conduit.blocks import final_block, build_money_shot_blocks

        # Trace context — only when NOT streaming. When streaming, the live trace message
        # already persists in the thread, so re-appending it here would duplicate it.
        context_blocks = []
        if self._trace and not self.stream:
            context_text = "\n".join(self._trace)[:2800]
            context_blocks = [{
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Research Trace:*\n{context_text}"
                }
            }]

        # Money-shot: graph badge + decision timeline + conflicting-signals — the same
        # structured proof the Canvas shows, now visible on EVERY surface (not just Canvas).
        money_shot_blocks = build_money_shot_blocks(answer, graph=graph, question=question)

        # Final answer + Canvas button (button omitted when there's no Canvas URL, since
        # Block Kit rejects an empty/invalid url).
        if canvas_url:
            final_blocks = final_block(answer.text, canvas_url)
        else:
            text = f"📄 *Final Answer*\n{answer.text}"
            if len(text) > 2900:  # Slack section limit; full answer lives in the Canvas
                text = text[:2899].rstrip() + "…"
            final_blocks = [{
                "type": "section",
                "text": {"type": "mrkdwn", "text": text},
            }]

        # Citation blocks — cap at 5 so the total stays under Slack's 50-block limit; the
        # full list lives in the Canvas. The quote is UNTRUSTED indexed text → escape it so it
        # can't inject a link/markup; guard an empty permalink so we render #channel, not `<|#…>`.
        from conduit.textsafe import mrkdwn_safe
        citation_blocks = []
        for citation in answer.citations[:5]:
            where = (f"<{citation.permalink}|#{citation.channel}>"
                     if citation.permalink else f"#{citation.channel}")
            quote = mrkdwn_safe(citation.quote[:100])
            citation_blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"{where}: {quote}"
                }
            })
        if len(answer.citations) > 5:
            citation_blocks.append({
                "type": "context",
                "elements": [{"type": "mrkdwn",
                              "text": f"…and {len(answer.citations) - 5} more — see the Canvas."}],
            })

        # Combine all blocks
        all_blocks = context_blocks + money_shot_blocks + final_blocks + citation_blocks

        return self.client.chat_postMessage(
            **self._msg_kwargs(),
            blocks=all_blocks,
            text=answer.text
        )
