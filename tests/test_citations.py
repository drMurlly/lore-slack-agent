"""Tests for citation grounding and synthesis."""
import pytest
from conduit.citations import _extract_citations_from_response, Citation
from conduit.research import Evidence, ResearchResult
from conduit.agent import LLMClient


class TestCitationGrounding:
    """Tests ensuring citations are grounded from Evidence, not LLM hallucinations."""
    
    def test_grounding_beats_hallucination(self):
        """Test that permalink comes from Evidence, not from LLM's CITATION line.
        
        Even if the LLM provides a fake permalink in its CITATION: line,
        the actual Evidence permalink must be used.
        """
        # Create evidence with a known real permalink
        evidence = [
            Evidence(
                text="We decided on Python for the project",
                channel="#engineering",
                permalink="https://slack.com/REAL",
                author="alice",
                ts="1234567890.123456",
                citation_index=1,
                score=0.9,
                source_hit=None,
            )
        ]
        
        # LLM response with hallucinated fake permalink
        content = """The team chose Python [1].

CITATION: [1] | wrong | https://slack.com/FAKE | "x"
"""
        citations = _extract_citations_from_response(content, evidence)
        
        # Should have exactly one citation
        assert len(citations) == 1
        # The permalink MUST come from Evidence, not from LLM
        assert citations[0].permalink == "https://slack.com/REAL"
        assert citations[0].channel == "#engineering"
        assert citations[0].index == 1
    
    def test_dangling_marker_dropped(self):
        """Test that [n] markers without corresponding evidence are dropped.
        
        If LLM cites [3] but only 2 evidence items exist, no Citation with index 3.
        """
        evidence = [
            Evidence(
                text="First piece of evidence",
                channel="#general",
                permalink="https://slack.com/REAL1",
                author="alice",
                ts="1234567890.123456",
                citation_index=1,
                score=0.9,
                source_hit=None,
            ),
            Evidence(
                text="Second piece of evidence",
                channel="#general",
                permalink="https://slack.com/REAL2",
                author="bob",
                ts="1234567891.123456",
                citation_index=2,
                score=0.85,
                source_hit=None,
            ),
        ]
        
        # LLM cites [1], [2], and [3] but only 2 evidence items exist
        content = """Here is the answer [1] and [2] and [3].

CITATION: [1] | #general | https://slack.com/REAL1 | "First piece of evidence"
CITATION: [2] | #general | https://slack.com/REAL2 | "Second piece of evidence"
CITATION: [3] | #general | https://slack.com/FAKE | "Non-existent evidence"
"""
        citations = _extract_citations_from_response(content, evidence)
        
        # Should only have citations for indices 1 and 2
        assert len(citations) == 2
        indices = {c.index for c in citations}
        assert indices == {1, 2}
        assert 3 not in indices
    
    def test_quote_comes_from_evidence(self):
        """Test that Citation.quote comes from Evidence text, not LLM text.
        
        The quote should be a prefix of the actual evidence text.
        """
        evidence = [
            Evidence(
                text="This is the actual evidence text that should be quoted",
                channel="#engineering",
                permalink="https://slack.com/REAL",
                author="alice",
                ts="1234567890.123456",
                citation_index=1,
                score=0.9,
                source_hit=None,
            )
        ]
        
        # LLM provides a completely different quote in its CITATION line
        content = """The answer is here [1].

CITATION: [1] | #engineering | https://slack.com/REAL | "Completely different fake quote"
"""
        citations = _extract_citations_from_response(content, evidence)
        
        assert len(citations) == 1
        # Quote should come from evidence, not from LLM
        assert citations[0].quote == "This is the actual evidence text that should be quoted"
        assert citations[0].quote != "Completely different fake quote"
    
    def test_fallback_to_marker_extraction(self):
        """Test that [n] markers are extracted when no CITATION: lines exist.
        
        The fallback path should still ground data from Evidence.
        """
        evidence = [
            Evidence(
                text="Evidence one",
                channel="#general",
                permalink="https://slack.com/ONE",
                author="alice",
                ts="1234567890.123456",
                citation_index=1,
                score=0.9,
                source_hit=None,
            ),
            Evidence(
                text="Evidence two",
                channel="#general",
                permalink="https://slack.com/TWO",
                author="bob",
                ts="1234567891.123456",
                citation_index=2,
                score=0.85,
                source_hit=None,
            ),
        ]
        
        # No CITATION: lines, just [n] markers in text
        content = """The answer uses [1] and [2] for support.
"""
        citations = _extract_citations_from_response(content, evidence)
        
        assert len(citations) == 2
        # Both should be grounded from evidence
        assert citations[0].permalink == "https://slack.com/ONE"
        assert citations[1].permalink == "https://slack.com/TWO"
    
    def test_empty_evidence_returns_empty_citations(self):
        """Test that empty evidence list returns empty citations."""
        content = """The answer is [1].

CITATION: [1] | #general | https://slack.com/FAKE | "fake"
"""
        citations = _extract_citations_from_response(content, [])
        
        assert len(citations) == 0
    
    def test_invalid_index_in_citation_line(self):
        """Test that invalid indices in CITATION: lines are skipped."""
        evidence = [
            Evidence(
                text="Only one evidence",
                channel="#general",
                permalink="https://slack.com/REAL",
                author="alice",
                ts="1234567890.123456",
                citation_index=1,
                score=0.9,
                source_hit=None,
            )
        ]
        
        # LLM cites [1] and [99] but only [1] is valid
        content = """Answer [1] and [99].

CITATION: [1] | #general | https://slack.com/REAL | "Only one evidence"
CITATION: [99] | #general | https://slack.com/FAKE | "Non-existent"
"""
        citations = _extract_citations_from_response(content, evidence)
        
        assert len(citations) == 1
        assert citations[0].index == 1
