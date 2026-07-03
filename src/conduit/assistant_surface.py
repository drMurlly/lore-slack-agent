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
    _trace: list[str] = field(default_factory=list, init=False)
    _trace_ts: Optional[str] = field(default=None, init=False)
    _trace_blocks: list = field(default_factory=list, init=False)

    def set_status(self, status: str) -> None:
        """Update the split-view thinking indicator (one API call). Defensive: a failed status
        update (e.g. outside a real assistant container) must never break the research run."""
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
        """Render one trace step into the single, edited-in-place research message."""
        from conduit.blocks import trace_block, TraceStep
        try:
            if not self._trace_blocks:
                self._trace_blocks.append({
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": "🔦 *Researching…*"},
                })
            self._trace_blocks.append(trace_block(TraceStep(phase, detail)))
            blocks = self._trace_blocks[-48:]  # stay under Slack's 50-block cap
            if self._trace_ts is None:
                resp = self.client.chat_postMessage(
                    channel=self.context.channel, thread_ts=self.context.thread_ts,
                    blocks=blocks, text="Researching…",
                )
                self._trace_ts = _extract_ts(resp)
            else:
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
    
    def post_result(self, answer: Answer, canvas_url: str) -> Any:
        """Post final result with trace context and citations.
        
        Args:
            answer: The synthesized answer with citations.
            canvas_url: URL to the Canvas report.
            
        Returns:
            The result of chat_postMessage.
        """
        from conduit.blocks import final_block

        # Build context block with trace lines
        context_blocks = []
        if self._trace:
            context_text = "\n".join(self._trace)[:2800]
            context_blocks = [{
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Research Trace:*\n{context_text}"
                }
            }]

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
        # full list lives in the Canvas.
        citation_blocks = []
        for citation in answer.citations[:5]:
            citation_blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"<{citation.permalink}|#{citation.channel}>: {citation.quote[:100]}"
                }
            })
        if len(answer.citations) > 5:
            citation_blocks.append({
                "type": "context",
                "elements": [{"type": "mrkdwn",
                              "text": f"…and {len(answer.citations) - 5} more — see the Canvas."}],
            })

        # Combine all blocks
        all_blocks = context_blocks + final_blocks + citation_blocks
        
        return self.client.chat_postMessage(
            channel=self.context.channel,
            thread_ts=self.context.thread_ts,
            blocks=all_blocks,
            text=answer.text
        )
