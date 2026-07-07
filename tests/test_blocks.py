"""Tests for Block Kit builders."""

import pytest
from conduit.blocks import build_answer_blocks


class TestBuildAnswerBlocks:
    """Tests for build_answer_blocks function."""

    def test_answer_without_tools(self):
        """Test building blocks for an answer with no tools used."""
        answer = "Hello! How can I help you today?"
        blocks = build_answer_blocks(answer)

        assert len(blocks) == 1
        assert blocks[0]["type"] == "section"
        assert blocks[0]["text"]["type"] == "mrkdwn"
        assert blocks[0]["text"]["text"] == answer

    def test_answer_with_tools(self):
        """Test building blocks for an answer with tools used."""
        answer = "I found the information you requested."
        tools_used = ["search_docs", "get_user_info"]
        blocks = build_answer_blocks(answer, tools_used)

        assert len(blocks) == 2
        assert blocks[0]["type"] == "section"
        assert blocks[0]["text"]["text"] == answer
        assert blocks[1]["type"] == "context"
        assert "🔧 Tools used:" in blocks[1]["elements"][0]["text"]
        assert "`search_docs`" in blocks[1]["elements"][0]["text"]
        assert "`get_user_info`" in blocks[1]["elements"][0]["text"]

    def test_answer_with_single_tool(self):
        """Test building blocks for an answer with a single tool used."""
        answer = "Here's the result."
        tools_used = ["calculator"]
        blocks = build_answer_blocks(answer, tools_used)

        assert len(blocks) == 2
        assert "🔧 Tools used: `calculator`" in blocks[1]["elements"][0]["text"]

    def test_answer_with_empty_tools_list(self):
        """Test that empty tools list doesn't add context block."""
        answer = "Just a simple answer."
        blocks = build_answer_blocks(answer, [])

        assert len(blocks) == 1
        assert blocks[0]["text"]["text"] == answer
