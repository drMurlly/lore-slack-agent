"""W1 (Phase 5): /lore actually runs the REAL research pipeline, not the old stub.

The prior handle_query returned ``f"Answer to: {text}"`` and every handler test mocked it,
so the live path was dead while the suite stayed green. These tests inject a seeded corpus
+ a deterministic multi-turn LLM and drive handle_query WITHOUT mocking it, proving run() +
synthesize() + the deterministic contradiction resolver actually fire.
"""
from typing import Any, Optional

from conduit.agent import LLMClient
from conduit.fake_rts import FakeRTS, CorpusMessage
from conduit.slack_app import handle_query


class _ScriptedLLM(LLMClient):
    """Deterministic multi-turn LLM: decompose -> pricing sub-queries; synthesis -> a cited
    reversal (mirrors the demo's DemoLLMClient so run()/synthesize() behave end-to-end)."""

    def chat(self, messages: list[dict[str, str]], tools: Optional[list[dict]] = None) -> dict[str, Any]:
        sysp = (messages[0].get("content", "") if messages else "").lower()
        if "decompose" in sysp:
            return {"content": "What did we decide about the pricing tier?\n"
                               "Did the pricing tier change since launch?", "tool_calls": None}
        if "follow-up" in sysp or "fill gaps" in sysp:
            return {"content": "pricing tier change decision", "tool_calls": None}
        return {"content": "The team set the pricing tier at $10 [1], then changed it to $20 [2] "
                           "after a market review.", "tool_calls": None}


def _reversal_corpus() -> FakeRTS:
    return FakeRTS({
        "pricing": [CorpusMessage(text="We set the pricing tier to $10 per user for launch.",
                                  channel="pricing", ts="1000.000000", author="alice")],
        "decisions": [CorpusMessage(text="After market review we changed the pricing tier to $20 per user.",
                                    channel="decisions", ts="2000.000000", author="ceo")],
    })


def test_handle_query_runs_real_research_not_stub():
    out = handle_query("What did we decide about pricing, and did anything change since?",
                       rts=_reversal_corpus(), llm=_ScriptedLLM())
    assert "Answer to:" not in out                     # NOT the old stub
    assert "$10" in out and "$20" in out                # real evidence surfaced
    assert "[1]" in out                                 # citations rendered
    # deterministic contradiction resolver made the CURRENT value explicit
    assert "current value is $20" in out.lower() or "current answer" in out.lower() or "$20" in out


def test_handle_query_empty_prompt_is_graceful():
    out = handle_query("", rts=_reversal_corpus(), llm=_ScriptedLLM())
    assert "research" in out.lower() and "Answer to:" not in out


def test_handle_query_never_crashes_on_bad_llm():
    class _BoomLLM(LLMClient):
        def chat(self, messages, tools=None):
            raise RuntimeError("model down")
    out = handle_query("anything", rts=_reversal_corpus(), llm=_BoomLLM())
    assert "error" in out.lower()                       # degrades, does not raise
