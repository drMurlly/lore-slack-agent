"""Tests for canvas report building."""
import pytest

from conduit.citations import Answer, Citation
from conduit.canvas import build_report, _format_answer_with_links, _build_sources_section


class TestBuildReport:
    """Tests for build_report function."""
    
    def test_build_report_with_single_citation(self):
        """Test building report with a single citation."""
        answer = Answer(
            text="The team uses Python [1] for backend services.",
            citations=[
                Citation(
                    index=1,
                    permalink="https://slack.com/archives/C123456",
                    channel="engineering",
                    quote="We'll use Python for the backend",
                )
            ]
        )
        
        result = build_report(answer, "What is the backend stack?")
        
        assert "document_content" in result
        assert "blocks" in result["document_content"]
        
        blocks = result["document_content"]["blocks"]
        
        # Check header
        assert blocks[0]["type"] == "header"
        assert "Research: What is the backend stack?" in blocks[0]["text"]["text"]
        
        # Check answer section
        assert blocks[1]["type"] == "section"
        assert blocks[1]["text"]["type"] == "mrkdwn"
        
        # Check sources section
        assert blocks[2]["type"] == "divider"
        assert blocks[3]["type"] == "section"
        assert "Sources:" in blocks[3]["text"]["text"]
    
    def test_build_report_with_multiple_citations(self):
        """Test building report with multiple citations."""
        answer = Answer(
            text="We use Python [1] but switched to Go [2] for performance.",
            citations=[
                Citation(
                    index=1,
                    permalink="https://slack.com/archives/C111",
                    channel="engineering",
                    quote="Initial decision: Python",
                ),
                Citation(
                    index=2,
                    permalink="https://slack.com/archives/C222",
                    channel="engineering",
                    quote="Switching to Go for performance",
                )
            ]
        )
        
        result = build_report(answer, "What is the backend stack?")
        
        blocks = result["document_content"]["blocks"]
        
        # Check that both citations appear in sources
        sources_text = blocks[3]["text"]["text"]
        assert "[1]" in sources_text
        assert "[2]" in sources_text
        assert "#engineering" in sources_text
    
    def test_build_report_without_citations(self):
        """Test building report without citations."""
        answer = Answer(
            text="This is a general answer without specific sources.",
            citations=[]
        )
        
        result = build_report(answer, "General question?")
        
        blocks = result["document_content"]["blocks"]
        
        # Should have header and answer, but no sources section
        assert len(blocks) == 2
        assert blocks[0]["type"] == "header"
        assert blocks[1]["type"] == "section"
    
    def test_build_report_links_citations(self):
        """Test that citations are converted to clickable links."""
        answer = Answer(
            text="See [1] and [2] for details.",
            citations=[
                Citation(
                    index=1,
                    permalink="https://slack.com/archives/C111",
                    channel="general",
                    quote="First source",
                ),
                Citation(
                    index=2,
                    permalink="https://slack.com/archives/C222",
                    channel="general",
                    quote="Second source",
                )
            ]
        )
        
        result = build_report(answer, "Question?")
        
        answer_text = result["document_content"]["blocks"][1]["text"]["text"]
        
        # Check that links are present
        assert "(https://slack.com/archives/C111)" in answer_text
        assert "(https://slack.com/archives/C222)" in answer_text
    
    def test_build_report_snapshot_stable(self):
        """Test that build_report produces stable output."""
        answer = Answer(
            text="The answer [1] is clear.",
            citations=[
                Citation(
                    index=1,
                    permalink="https://slack.com/archives/C123",
                    channel="test",
                    quote="Source text",
                )
            ]
        )
        
        result1 = build_report(answer, "Test question?")
        result2 = build_report(answer, "Test question?")
        
        assert result1 == result2


class TestFormatAnswerWithLinks:
    """Tests for _format_answer_with_links helper."""
    
    def test_single_citation_link(self):
        """Test converting single citation to link."""
        text = "Use Python [1] for this."
        citations = [
            Citation(
                index=1,
                permalink="https://slack.com/archives/C123",
                channel="engineering",
                quote="Python is good",
            )
        ]
        
        result = _format_answer_with_links(text, citations)
        
        assert "[1](https://slack.com/archives/C123)" in result
    
    def test_multiple_citation_links(self):
        """Test converting multiple citations to links."""
        text = "First [1] then [2] finally [3]."
        citations = [
            Citation(index=1, permalink="https://slack.com/1", channel="c1", quote="q1"),
            Citation(index=2, permalink="https://slack.com/2", channel="c2", quote="q2"),
            Citation(index=3, permalink="https://slack.com/3", channel="c3", quote="q3"),
        ]
        
        result = _format_answer_with_links(text, citations)
        
        assert "[1](https://slack.com/1)" in result
        assert "[2](https://slack.com/2)" in result
        assert "[3](https://slack.com/3)" in result
    
    def test_no_citations_returns_original(self):
        """Test that text without citations is returned unchanged."""
        text = "This has no citations."
        citations = []
        
        result = _format_answer_with_links(text, citations)
        
        assert result == text
    
    def test_missing_citation_index_keeps_marker(self):
        """Test that missing citation indices keep original marker."""
        text = "Reference [5] here."
        citations = [
            Citation(index=1, permalink="https://slack.com/1", channel="c1", quote="q1"),
        ]
        
        result = _format_answer_with_links(text, citations)
        
        # [5] should remain as-is since there's no citation 5
        assert "[5]" in result


class TestBuildSourcesSection:
    """Tests for _build_sources_section helper."""
    
    def test_sources_with_single_citation(self):
        """Test building sources section with one citation."""
        citations = [
            Citation(
                index=1,
                permalink="https://slack.com/archives/C123",
                channel="engineering",
                quote="We use Python",
            )
        ]
        
        result = _build_sources_section(citations)
        
        assert "*Sources:*" in result
        assert "[1]" in result
        assert "#engineering" in result
        assert "We use Python" in result
    
    def test_sources_sorted_by_index(self):
        """Test that sources are sorted by citation index."""
        citations = [
            Citation(index=3, permalink="https://slack.com/3", channel="c3", quote="q3"),
            Citation(index=1, permalink="https://slack.com/1", channel="c1", quote="q1"),
            Citation(index=2, permalink="https://slack.com/2", channel="c2", quote="q2"),
        ]
        
        result = _build_sources_section(citations)
        
        # Check order: [1] should come before [2], which comes before [3]
        pos_1 = result.find("[1]")
        pos_2 = result.find("[2]")
        pos_3 = result.find("[3]")
        
        assert pos_1 < pos_2 < pos_3
    
    def test_sources_escapes_quotes(self):
        """Test that quotes in source text are escaped."""
        citations = [
            Citation(
                index=1,
                permalink="https://slack.com/1",
                channel="general",
                quote='He said "hello"',
            )
        ]
        
        result = _build_sources_section(citations)
        
        # Double quotes should be single quotes
        assert '"hello"' not in result
        assert "'hello'" in result
