"""End-to-end test for the Lore research pipeline (fully mocked, no live Slack).

The money-shot: a compound, contradiction-bearing question over a seeded FakeRTS corpus
flows decompose -> multi-hop -> synthesize -> cited Canvas payload -> trace/final blocks,
and the answer surfaces the REVERSED decision + states the current value with citations.
"""
from conduit.research import run, synthesize
from conduit.fake_rts import FakeRTS, CorpusMessage
from conduit.canvas import build_report
from conduit.blocks import trace_block, final_block, TraceStep
from conduit.agent import LLMClient


class ScriptedLLM(LLMClient):
    """Deterministic LLM: routes by the system prompt to decompose / follow-up / synth."""

    def chat(self, messages, tools=None):
        sys_prompt = messages[0].get("content", "").lower()
        if "decompose" in sys_prompt:
            return {"content": "What did we decide about pricing?\nDid the pricing change?",
                    "tool_calls": None}
        if "follow-up" in sys_prompt or "fill gaps" in sys_prompt:
            return {"content": "pricing final decision", "tool_calls": None}
        # synthesis
        return {"content": (
            "We initially decided on $10 [1], but later changed to $20 [2]. "
            "The current price is $20.\n"
            'CITATION: [1] | pricing | https://slack.com/p1 | "decided on $10"\n'
            'CITATION: [2] | pricing | https://slack.com/p2 | "changed to $20"'
        ), "tool_calls": None}


def _corpus():
    return {
        "pricing": [
            CorpusMessage(text="We decided on $10 for the pricing tier.",
                          channel="pricing", ts="1.1", author="alice"),
            CorpusMessage(text="Update: we changed pricing to $20 after review.",
                          channel="pricing", ts="2.2", author="bob"),
            CorpusMessage(text="Pricing decision discussion continues in the thread.",
                          channel="pricing", ts="3.3", author="carol"),
        ],
    }


def test_e2e_lore_pipeline_moneyshot():
    llm = ScriptedLLM()
    rts = FakeRTS(corpus=_corpus())
    question = "What did we decide about pricing, and did it change?"

    # 1. multi-hop research pulls evidence
    result = run(question, rts, llm)
    assert result.question == question
    assert len(result.evidence) >= 2

    # 2. cited synthesis surfaces BOTH values + the current one (the money-shot)
    answer = synthesize(result, llm)
    assert "$20" in answer.text        # current value surfaced
    assert "$10" in answer.text        # the reversed/earlier value mentioned
    assert len(answer.citations) >= 2
    assert all(c.permalink for c in answer.citations)

    # 3. Canvas payload built with citation deep-links
    canvas = build_report(answer, question)
    assert "document_content" in canvas
    assert canvas["document_content"]["blocks"]

    # 4. trace + final Block Kit render
    tb = trace_block(TraceStep(phase="search", detail="pricing"))
    assert tb["type"] == "section"
    fb = final_block("The current price is $20", "https://slack.com/canvas/x")
    assert any(b.get("type") == "actions" for b in fb)


def test_e2e_lore_empty_corpus_is_graceful():
    """No evidence → synthesize returns a safe, non-crashing answer."""
    llm = ScriptedLLM()
    rts = FakeRTS(corpus={"empty": []})
    result = run("anything at all?", rts, llm)
    answer = synthesize(result, llm)
    assert isinstance(answer.text, str)
