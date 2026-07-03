"""Tests for the Conduit Agent."""
import pytest

from conduit.agent import Agent, AgentResult, FakeLLMClient


class FakeToolManager:
    """Fake tool manager for testing."""

    def __init__(self):
        self._tools = [
            {
                "name": "stub__echo",
                "description": "Echoes back the input message",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "Message to echo"}
                    },
                    "required": ["message"],
                },
            }
        ]
        self._execution_log: list[tuple[str, dict]] = []

    def get_tools(self) -> list[dict]:
        """Return available tools."""
        return self._tools

    def execute(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool and log the call."""
        self._execution_log.append((tool_name, arguments))
        if tool_name == "stub__echo":
            return f"Echo: {arguments.get('message', '')}"
        raise ValueError(f"Unknown tool: {tool_name}")

    def get_execution_log(self) -> list[tuple[str, dict]]:
        """Return the execution log."""
        return self._execution_log


class TestAgent:
    """Tests for the Agent class."""

    def test_agent_calls_tool_and_returns_answer(self):
        """Test that agent calls the tool and returns correct answer with tools_used."""
        # Track call count to simulate ReAct loop
        call_count = [0]

        def create_response():
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: request tool
                return {
                    "content": None,
                    "tool_calls": [
                        {
                            "name": "stub__echo",
                            "arguments": {"message": "Hello, world!"},
                        }
                    ],
                }
            else:
                # Second call: return final answer
                return {
                    "content": "The echoed message is: Hello, world!",
                    "tool_calls": [],
                }

        class ReActFakeLLM(FakeLLMClient):
            def chat(self, messages, tools=None):
                return create_response()

        fake_llm = ReActFakeLLM()
        tool_manager = FakeToolManager()
        agent = Agent(llm_client=fake_llm, tool_manager=tool_manager)

        # Run the agent
        result = agent.run("Can you echo 'Hello, world!'?")

        # Assert the result
        assert isinstance(result, AgentResult)
        assert "Hello, world!" in result.answer
        assert "stub__echo" in result.tools_used

        # Assert the tool was actually called
        execution_log = tool_manager.get_execution_log()
        assert len(execution_log) == 1
        assert execution_log[0][0] == "stub__echo"
        assert execution_log[0][1] == {"message": "Hello, world!"}

    def test_agent_multiple_tool_calls(self):
        """Test agent with multiple tool calls in sequence."""
        call_count = [0]

        def create_response():
            call_count[0] += 1
            if call_count[0] == 1:
                return {
                    "content": None,
                    "tool_calls": [{"name": "stub__echo", "arguments": {"message": "First"}}],
                }
            else:
                return {
                    "content": "I've echoed both messages for you.",
                    "tool_calls": [],
                }

        class MultiCallFakeLLM(FakeLLMClient):
            def chat(self, messages, tools=None):
                return create_response()

        fake_llm = MultiCallFakeLLM()
        tool_manager = FakeToolManager()
        agent = Agent(llm_client=fake_llm, tool_manager=tool_manager)

        result = agent.run("Echo 'First' and then 'Second'")

        assert "stub__echo" in result.tools_used
        assert len(tool_manager.get_execution_log()) >= 1

    def test_agent_no_tools_needed(self):
        """Test agent when no tools are needed."""
        fake_llm = FakeLLMClient(
            scripted_response={
                "content": "The answer is 42.",
                "tool_calls": [],
            }
        )

        tool_manager = FakeToolManager()
        agent = Agent(llm_client=fake_llm, tool_manager=tool_manager)

        result = agent.run("What is 6 times 7?")

        assert result.answer == "The answer is 42."
        assert result.tools_used == []
        assert len(tool_manager.get_execution_log()) == 0

    def test_agent_max_rounds_reached(self):
        """Test agent when max rounds are reached."""
        # Fake LLM that keeps requesting tools without final answer
        class InfiniteToolFakeLLM(FakeLLMClient):
            def chat(self, messages, tools=None):
                return {
                    "content": None,
                    "tool_calls": [{"name": "stub__echo", "arguments": {"message": "test"}}],
                }

        fake_llm = InfiniteToolFakeLLM()
        tool_manager = FakeToolManager()
        agent = Agent(llm_client=fake_llm, tool_manager=tool_manager, max_rounds=2)

        result = agent.run("Keep echoing forever")

        assert "maximum number of tool calls" in result.answer
        assert len(result.tools_used) == 2

    def test_agent_timeout(self):
        """Test agent timeout handling."""
        class SlowFakeLLM(FakeLLMClient):
            def chat(self, messages, tools=None):
                import time
                time.sleep(0.1)  # Simulate slow response
                return {
                    "content": None,
                    "tool_calls": [{"name": "stub__echo", "arguments": {"message": "test"}}],
                }

        fake_llm = SlowFakeLLM()
        tool_manager = FakeToolManager()
        agent = Agent(llm_client=fake_llm, tool_manager=tool_manager, timeout=0.05)

        result = agent.run("This should timeout")

        assert "timed out" in result.answer.lower()
