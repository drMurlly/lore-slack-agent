"""Tests for Block Kit builders."""

import pytest
from conduit.blocks import build_answer_blocks, build_home_view


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


class TestBuildHomeView:
    """Tests for build_home_view function."""
    
    def test_home_view_with_single_server(self):
        """Test building home view with a single server and tools."""
        servers = [
            {
                "name": "docs-server",
                "tools": [
                    {"name": "search_docs"},
                    {"name": "get_doc"},
                ],
            },
        ]
        view = build_home_view(servers)
        
        assert view["type"] == "home"
        assert "blocks" in view
        assert len(view["blocks"]) > 0
        
        # Check header exists
        header = view["blocks"][0]
        assert header["type"] == "header"
        assert "Conduit Agent" in header["text"]["text"]
        
        # Check server name is listed
        server_block = view["blocks"][2]  # After header and divider
        assert server_block["type"] == "section"
        assert "*docs-server*" in server_block["text"]["text"]
        
        # Check tools are listed
        tools_block = view["blocks"][3]
        assert "search_docs" in tools_block["text"]["text"]
        assert "get_doc" in tools_block["text"]["text"]
    
    def test_home_view_with_multiple_servers(self):
        """Test building home view with multiple servers."""
        servers = [
            {
                "name": "server-a",
                "tools": [{"name": "tool_a1"}],
            },
            {
                "name": "server-b",
                "tools": [{"name": "tool_b1"}, {"name": "tool_b2"}],
            },
        ]
        view = build_home_view(servers)
        
        assert view["type"] == "home"
        
        # Check both servers are listed
        blocks_text = "\n".join(
            str(block) for block in view["blocks"]
        )
        assert "server-a" in blocks_text
        assert "server-b" in blocks_text
        assert "tool_a1" in blocks_text
        assert "tool_b1" in blocks_text
        assert "tool_b2" in blocks_text
    
    def test_home_view_with_no_servers(self):
        """Test building home view with no servers."""
        servers = []
        view = build_home_view(servers)
        
        assert view["type"] == "home"
        assert view["blocks"][0]["type"] == "header"
        assert view["blocks"][1]["type"] == "divider"
    
    def test_home_view_with_server_no_tools(self):
        """Test building home view with a server that has no tools."""
        servers = [
            {
                "name": "empty-server",
                "tools": [],
            },
        ]
        view = build_home_view(servers)
        
        assert view["type"] == "home"
        
        # Check "No tools registered" message
        blocks_text = "\n".join(
            str(block) for block in view["blocks"]
        )
        assert "No tools registered" in blocks_text
    
    def test_home_view_server_count_footer(self):
        """Test that home view includes server count in footer."""
        servers = [
            {"name": "s1", "tools": []},
            {"name": "s2", "tools": []},
            {"name": "s3", "tools": []},
        ]
        view = build_home_view(servers)
        
        # Check footer context
        footer = view["blocks"][-1]
        assert footer["type"] == "context"
        assert "Total servers: 3" in footer["elements"][0]["text"]
