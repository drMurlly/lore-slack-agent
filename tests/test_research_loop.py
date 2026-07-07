"""Tests for the multi-hop research loop."""
import pytest
from typing import Any, Optional

from conduit.research import run, Evidence, ResearchResult
from conduit.fake_rts import FakeRTS, CorpusMessage
from conduit.agent import LLMClient


class CallCountingFakeLLM(LLMClient):
    """Fake LLM that counts chat calls and returns scripted responses."""
    
    def __init__(
        self,
        decomposition_response: str = "What is the team onboarding process?\nWhat are the engineering deployment procedures?",
        follow_up_response: str = "What is the expense report deadline?",
        call_count_attr: str = "_call_count"
    ):
        self.decomposition_response = decomposition_response
        self.follow_up_response = follow_up_response
        self._call_count = 0
        self._call_history: list[list[dict[str, str]]] = []
    
    def chat(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict]] = None
    ) -> dict[str, Any]:
        self._call_count += 1
        self._call_history.append(messages)
        
        # Determine response based on message content
        last_message = messages[-1].get("content", "")
        
        # Check if this is a follow-up query generation request
        if "follow-up" in last_message.lower() or "fill gaps" in last_message.lower():
            return {"content": self.follow_up_response, "tool_calls": None}
        
        # Default to decomposition response
        return {"content": self.decomposition_response, "tool_calls": None}
    
    @property
    def call_count(self) -> int:
        return self._call_count
    
    @property
    def call_history(self) -> list[list[dict[str, str]]]:
        return self._call_history


class TestResearchLoop:
    """Tests for the multi-hop research loop."""
    
    def test_question_spanning_two_channels_pulls_evidence_from_both(self):
        """Test that a question spanning two channels pulls evidence from BOTH."""
        # Create a corpus with messages in multiple channels
        corpus = {
            "general": [
                CorpusMessage(
                    text="The quarterly review meeting is scheduled for Friday at 3pm.",
                    channel="general",
                    ts="1234567890.123456",
                    author="alice",
                ),
                CorpusMessage(
                    text="Welcome to the team! Please read the onboarding docs.",
                    channel="general",
                    ts="1234567891.123456",
                    author="bob",
                ),
            ],
            "engineering": [
                CorpusMessage(
                    text="The API deployment completed successfully. All tests passing.",
                    channel="engineering",
                    ts="1234567892.123456",
                    author="dave",
                ),
                CorpusMessage(
                    text="We need to review the pull request for the new feature branch.",
                    channel="engineering",
                    ts="1234567893.123456",
                    author="eve",
                ),
            ],
        }
        
        fake_rts = FakeRTS(corpus=corpus)
        fake_llm = CallCountingFakeLLM(
            decomposition_response="quarterly review meeting\nAPI deployment"
        )
        
        result = run(
            question="What happened in the quarterly review and API deployment?",
            rts=fake_rts,
            llm=fake_llm,
            follow_up_threshold=2  # Set threshold ≤ unique hits to avoid follow-up
        )
        
        # Verify we got a ResearchResult
        assert isinstance(result, ResearchResult)
        assert result.question == "What happened in the quarterly review and API deployment?"
        
        # Verify evidence was pulled from both channels
        channels = {e.channel for e in result.evidence}
        assert "general" in channels
        assert "engineering" in channels
        
        # Verify we have evidence
        assert len(result.evidence) >= 2
        
        # Verify no follow-up was needed (threshold not reached)
        assert result.follow_up_hops == 0
    
    def test_thin_coverage_triggers_exactly_one_follow_up_hop(self):
        """Test that thin coverage triggers exactly one follow-up hop."""
        # Create a corpus with limited matching content
        corpus = {
            "general": [
                CorpusMessage(
                    text="The quarterly review meeting is scheduled for Friday at 3pm.",
                    channel="general",
                    ts="1234567890.123456",
                    author="alice",
                ),
            ],
            "engineering": [
                CorpusMessage(
                    text="Some unrelated engineering message.",
                    channel="engineering",
                    ts="1234567891.123456",
                    author="dave",
                ),
            ],
        }
        
        fake_rts = FakeRTS(corpus=corpus)
        fake_llm = CallCountingFakeLLM(
            decomposition_response="quarterly review\nengineering update",
            follow_up_response="expense report deadline"
        )
        
        result = run(
            question="What happened in the quarterly review and engineering update?",
            rts=fake_rts,
            llm=fake_llm,
            follow_up_threshold=5  # Low threshold to trigger follow-up
        )
        
        # Verify we got a ResearchResult
        assert isinstance(result, ResearchResult)
        
        # Verify exactly one follow-up hop was triggered
        assert result.follow_up_hops == 1
        
        # Verify the LLM was called for follow-up generation
        # Initial decomposition + follow-up generation = at least 2 calls
        assert fake_llm.call_count >= 2
    
    def test_evidence_has_stable_citation_indices(self):
        """Test that evidence items have stable 1-based citation indices."""
        corpus = {
            "general": [
                CorpusMessage(
                    text="Message one.",
                    channel="general",
                    ts="1234567890.123456",
                    author="alice",
                ),
                CorpusMessage(
                    text="Message two.",
                    channel="general",
                    ts="1234567891.123456",
                    author="bob",
                ),
            ],
        }
        
        fake_rts = FakeRTS(corpus=corpus)
        fake_llm = CallCountingFakeLLM(
            decomposition_response="Message one\nMessage two"
        )
        
        result = run(
            question="What are the messages?",
            rts=fake_rts,
            llm=fake_llm,
            follow_up_threshold=10
        )
        
        # Verify citation indices are 1-based and sequential
        indices = [e.citation_index for e in result.evidence]
        assert indices == list(range(1, len(indices) + 1))
    
    def test_evidence_includes_source_hit(self):
        """Test that each evidence item includes its source SearchHit."""
        corpus = {
            "general": [
                CorpusMessage(
                    text="Test message content.",
                    channel="general",
                    ts="1234567890.123456",
                    author="alice",
                ),
            ],
        }
        
        fake_rts = FakeRTS(corpus=corpus)
        fake_llm = CallCountingFakeLLM(
            decomposition_response="Test message"
        )
        
        result = run(
            question="What is the test message?",
            rts=fake_rts,
            llm=fake_llm,
            follow_up_threshold=10
        )
        
        # Verify each evidence has a source_hit
        for evidence in result.evidence:
            assert evidence.source_hit is not None
            assert evidence.source_hit.text == evidence.text
            assert evidence.source_hit.channel == evidence.channel
    
    def test_dedup_by_permalink(self):
        """Test that duplicate permalinks are deduplicated."""
        corpus = {
            "general": [
                CorpusMessage(
                    text="Same message.",
                    channel="general",
                    ts="1234567890.123456",
                    author="alice",
                ),
            ],
        }
        
        fake_rts = FakeRTS(corpus=corpus)
        # Decompose into queries that would match the same message
        fake_llm = CallCountingFakeLLM(
            decomposition_response="Same message\nSame message"
        )
        
        result = run(
            question="What is the message?",
            rts=fake_rts,
            llm=fake_llm,
            follow_up_threshold=10
        )
        
        # Verify no duplicate permalinks
        permalinks = [e.permalink for e in result.evidence]
        assert len(permalinks) == len(set(permalinks))
    
    def test_follow_up_hop_adds_new_evidence(self):
        """Test that follow-up hop adds new evidence to the result."""
        corpus = {
            "general": [
                CorpusMessage(
                    text="Initial message.",
                    channel="general",
                    ts="1234567890.123456",
                    author="alice",
                ),
            ],
            "random": [
                CorpusMessage(
                    text="Follow-up message about expenses.",
                    channel="random",
                    ts="1234567891.123456",
                    author="bob",
                ),
            ],
        }
        
        fake_rts = FakeRTS(corpus=corpus)
        fake_llm = CallCountingFakeLLM(
            decomposition_response="Initial message",
            follow_up_response="expenses"
        )
        
        result = run(
            question="What is the initial message and expenses?",
            rts=fake_rts,
            llm=fake_llm,
            follow_up_threshold=5
        )
        
        # Verify follow-up was triggered
        assert result.follow_up_hops == 1
        
        # Verify we have evidence from both initial and follow-up
        assert len(result.evidence) >= 1
        
        # Verify channels include both general and random (from follow-up)
        channels = {e.channel for e in result.evidence}
        assert "general" in channels
        assert "random" in channels


class TestGlossaryMattersForRetrieval:
    """The MCP glossary consult must EARN its place: resolved definitions are fed back into
    retrieval, so removing MCP measurably degrades recall on acronym/jargon questions."""

    def test_glossary_expansion_surfaces_evidence_the_acronym_alone_misses(self):
        # Evidence spells the term OUT ("Annual Recurring Revenue"); the question uses only the
        # acronym ("ARR"). Only the MCP-resolved expansion bridges the two.
        corpus = {
            "finance": [
                CorpusMessage(
                    text="Our Annual Recurring Revenue crossed $2M this quarter.",
                    channel="finance", ts="100.000100", author="cfo",
                )
            ]
        }
        rts = FakeRTS(corpus=corpus)

        class AcronymLLM(LLMClient):
            def chat(self, messages, tools=None):
                return {"content": "ARR", "tool_calls": None}

        class GlossaryStub:
            """Injected MCP manager (no subprocess) — resolves ARR to its long-form."""
            def call_tool(self, name, args=None, **kwargs):
                return [{"term": "ARR",
                         "definition": "Annual Recurring Revenue — the yearly run-rate."}]

        without_mcp = run("ARR?", rts, AcronymLLM(), glossary=False)
        with_mcp = run("ARR?", rts, AcronymLLM(), glossary=GlossaryStub())

        # Without the MCP consult, the acronym matches nothing in the spelled-out evidence.
        assert len(without_mcp.evidence) == 0
        # With it, the expansion "Annual Recurring Revenue" retrieves the message.
        assert len(with_mcp.evidence) >= 1
        assert any("Annual Recurring Revenue" in e.text for e in with_mcp.evidence)
        # And the resolved definition is surfaced on the result too.
        assert with_mcp.glossary and with_mcp.glossary[0]["term"] == "ARR"
