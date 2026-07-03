"""End-to-end tests for Conduit agent with real demo MCP servers."""

import json
import subprocess
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from conduit.agent import Agent, AgentResult, FakeLLMClient
from conduit.blocks import build_answer_blocks


class NotesServerProcess:
    """Context manager for running notes server as subprocess."""
    
    def __init__(self):
        self.process: subprocess.Popen | None = None
        self._buffer: list[str] = []
    
    def __enter__(self):
        self.process = subprocess.Popen(
            [sys.executable, "servers/notes_server.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=2)
    
    def call_tool(self, method: str, params: dict) -> dict[str, Any]:
        """Call a tool on the notes server."""
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": 1
        }
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        
        response_line = self.process.stdout.readline()
        return json.loads(response_line)


class WebFetchServerProcess:
    """Context manager for running web fetch server as subprocess."""
    
    def __init__(self):
        self.process: subprocess.Popen | None = None
    
    def __enter__(self):
        self.process = subprocess.Popen(
            [sys.executable, "servers/web_fetch_server.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=2)
    
    def call_tool(self, method: str, params: dict) -> dict[str, Any]:
        """Call a tool on the web fetch server."""
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": 1
        }
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        
        response_line = self.process.stdout.readline()
        return json.loads(response_line)


class MockToolManager:
    """Mock tool manager that uses real server processes."""
    
    def __init__(self):
        self._notes_server: NotesServerProcess | None = None
        self._web_server: WebFetchServerProcess | None = None
        self._execution_log: list[tuple[str, dict]] = []
    
    def start_servers(self):
        """Start all server processes."""
        self._notes_server = NotesServerProcess()
        self._notes_server.__enter__()
        self._web_server = WebFetchServerProcess()
        self._web_server.__enter__()
    
    def stop_servers(self):
        """Stop all server processes."""
        if self._notes_server:
            self._notes_server.__exit__(None, None, None)
        if self._web_server:
            self._web_server.__exit__(None, None, None)
    
    def get_tools(self) -> list[dict]:
        """Get available tools from all servers."""
        return [
            {
                "name": "notes__add_note",
                "description": "Add a new note with the given content",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "The note content"}
                    },
                    "required": ["content"]
                }
            },
            {
                "name": "notes__list_notes",
                "description": "List all notes",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "web_fetch__fetch_url",
                "description": "Fetch content from a URL",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "The URL to fetch"},
                        "max_length": {"type": "integer", "description": "Max length of content", "default": 2000}
                    },
                    "required": ["url"]
                }
            }
        ]
    
    def execute(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool call."""
        self._execution_log.append((tool_name, arguments))
        
        if tool_name == "notes__add_note":
            result = self._notes_server.call_tool("add_note", arguments)
            return result.get("result", json.dumps({"error": "No result"}))
        elif tool_name == "notes__list_notes":
            result = self._notes_server.call_tool("list_notes", {})
            return result.get("result", json.dumps({"error": "No result"}))
        elif tool_name == "web_fetch__fetch_url":
            result = self._web_server.call_tool("fetch_url", arguments)
            return result.get("result", json.dumps({"error": "No result"}))
        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
    
    def get_execution_log(self) -> list[tuple[str, dict]]:
        """Get the execution log."""
        return self._execution_log.copy()


class ReActFakeLLM(FakeLLMClient):
    """Fake LLM that simulates ReAct loop behavior."""
    
    def __init__(self, sequence: list[dict[str, Any]]):
        self._sequence = sequence
        self._index = 0
    
    def chat(self, messages: list[dict[str, str]], tools: list[dict] | None = None) -> dict[str, Any]:
        if self._index < len(self._sequence):
            response = self._sequence[self._index]
            self._index += 1
            return response
        # Default to final answer if sequence exhausted
        return {
            "content": "I've completed the requested actions.",
            "tool_calls": None
        }


class TestE2E:
    """End-to-end tests with real MCP servers."""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        self.tool_manager = MockToolManager()
        self.tool_manager.start_servers()
        yield
        self.tool_manager.stop_servers()
    
    def test_agent_adds_note_and_confirms(self):
        """Test agent adds a note via the notes server and confirms to user."""
        # Simulate ReAct sequence: tool call -> final answer
        llm_sequence = [
            {
                "content": None,
                "tool_calls": [{"name": "notes__add_note", "arguments": {"content": "Meeting notes from today"}}]
            },
            {
                "content": "I've added your note: 'Meeting notes from today'. It has been saved with ID 1.",
                "tool_calls": None
            }
        ]
        
        llm_client = ReActFakeLLM(llm_sequence)
        agent = Agent(
            llm_client=llm_client,
            tool_manager=self.tool_manager,
            model="test-model",
            max_rounds=4
        )
        
        result = agent.run("Add a note: Meeting notes from today")
        
        assert isinstance(result, AgentResult)
        assert "Meeting notes" in result.answer
        assert len(self.tool_manager.get_execution_log()) == 1
        tool_name, args = self.tool_manager.get_execution_log()[0]
        assert tool_name == "notes__add_note"
        assert args["content"] == "Meeting notes from today"
    
    def test_agent_lists_notes(self):
        """Test agent lists all notes via the notes server."""
        llm_sequence = [
            {
                "content": None,
                "tool_calls": [{"name": "notes__list_notes", "arguments": {}}]
            },
            {
                "content": "Here are your notes: [list of notes]",
                "tool_calls": None
            }
        ]
        
        llm_client = ReActFakeLLM(llm_sequence)
        agent = Agent(
            llm_client=llm_client,
            tool_manager=self.tool_manager,
            model="test-model",
            max_rounds=4
        )
        
        result = agent.run("Show me all my notes")
        
        assert isinstance(result, AgentResult)
        assert len(self.tool_manager.get_execution_log()) == 1
        tool_name, args = self.tool_manager.get_execution_log()[0]
        assert tool_name == "notes__list_notes"
    
    def test_agent_fetches_url(self):
        """Test agent fetches URL content via web fetch server.

        Serves a local HTML fixture over loopback (no external network) so we
        can assert the web-fetch tool call actually returns non-error content.
        Regression test for P1-11: the server previously died on import
        (NameError: Optional) and the agent's blanket except swallowed the
        resulting BrokenPipeError, so a log-only assertion still passed.
        """
        import http.server
        import threading

        class FixtureHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                body = b"<html><body><p>Hello from local fixture</p></body></html>"
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass  # keep test output quiet

        httpd = http.server.HTTPServer(("127.0.0.1", 0), FixtureHandler)
        server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        server_thread.start()
        url = f"http://127.0.0.1:{httpd.server_address[1]}/"

        try:
            llm_sequence = [
                {
                    "content": None,
                    "tool_calls": [{"name": "web_fetch__fetch_url", "arguments": {"url": url}}]
                },
                {
                    "content": "I fetched the page and here's what I found...",
                    "tool_calls": None
                }
            ]

            llm_client = ReActFakeLLM(llm_sequence)
            agent = Agent(
                llm_client=llm_client,
                tool_manager=self.tool_manager,
                model="test-model",
                max_rounds=4
            )

            result = agent.run(f"Fetch {url} for me")

            assert isinstance(result, AgentResult)
            assert len(self.tool_manager.get_execution_log()) == 1
            tool_name, args = self.tool_manager.get_execution_log()[0]
            assert tool_name == "web_fetch__fetch_url"
            assert args["url"] == url
            # tools_used is only populated when execution did not raise;
            # a dead server process (BrokenPipeError) would leave it empty.
            assert "web_fetch__fetch_url" in result.tools_used

            # The tool call must return actual non-error content.
            fetch_result = json.loads(
                self.tool_manager.execute("web_fetch__fetch_url", {"url": url})
            )
            assert fetch_result.get("status") == "success", fetch_result
            assert "Hello from local fixture" in fetch_result.get("content", "")
        finally:
            httpd.shutdown()
            server_thread.join(timeout=2)
            httpd.server_close()
    
    def test_agent_multiple_tool_calls(self):
        """Test agent makes multiple tool calls in sequence."""
        llm_sequence = [
            {
                "content": None,
                "tool_calls": [{"name": "notes__add_note", "arguments": {"content": "First note"}}]
            },
            {
                "content": None,
                "tool_calls": [{"name": "notes__add_note", "arguments": {"content": "Second note"}}]
            },
            {
                "content": "I've added both notes for you.",
                "tool_calls": None
            }
        ]
        
        llm_client = ReActFakeLLM(llm_sequence)
        agent = Agent(
            llm_client=llm_client,
            tool_manager=self.tool_manager,
            model="test-model",
            max_rounds=4
        )
        
        result = agent.run("Add two notes: 'First note' and 'Second note'")
        
        assert isinstance(result, AgentResult)
        assert len(self.tool_manager.get_execution_log()) == 2
        assert self.tool_manager.get_execution_log()[0][0] == "notes__add_note"
        assert self.tool_manager.get_execution_log()[1][0] == "notes__add_note"
    
    def test_block_kit_response_includes_tools_used(self):
        """Test that Block Kit response includes tools used information."""
        llm_sequence = [
            {
                "content": None,
                "tool_calls": [{"name": "notes__add_note", "arguments": {"content": "Test note"}}]
            },
            {
                "content": "Note added successfully!",
                "tool_calls": None
            }
        ]
        
        llm_client = ReActFakeLLM(llm_sequence)
        agent = Agent(
            llm_client=llm_client,
            tool_manager=self.tool_manager,
            model="test-model",
            max_rounds=4
        )
        
        result = agent.run("Add a test note")
        
        # Build Block Kit response
        blocks = build_answer_blocks(
            answer=result.answer,
            tools_used=result.tools_used
        )
        
        assert isinstance(blocks, list)
        assert len(blocks) > 0
        # Should have a section mentioning tools used
        assert any("notes__add_note" in str(block) for block in blocks)
    
    def test_agent_handles_tool_error_gracefully(self):
        """Test agent handles tool execution errors gracefully."""
        # Create a tool manager that returns an error
        class ErrorToolManager:
            def get_tools(self):
                return [{"name": "notes__add_note", "description": "Add note", "parameters": {"type": "object"}}]
            
            def execute(self, tool_name: str, arguments: dict) -> str:
                return json.dumps({"error": "Server unavailable"})
            
            def get_execution_log(self):
                return []
        
        llm_sequence = [
            {
                "content": None,
                "tool_calls": [{"name": "notes__add_note", "arguments": {"content": "Test"}}]
            },
            {
                "content": "I'm sorry, I couldn't complete that action. The service is currently unavailable.",
                "tool_calls": None
            }
        ]
        
        llm_client = ReActFakeLLM(llm_sequence)
        agent = Agent(
            llm_client=llm_client,
            tool_manager=ErrorToolManager(),
            model="test-model",
            max_rounds=4
        )
        
        result = agent.run("Add a note")
        
        assert isinstance(result, AgentResult)
        assert "unavailable" in result.answer.lower() or "sorry" in result.answer.lower()
