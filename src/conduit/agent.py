"""Conduit Agent - ReAct loop with local LLM tool selection."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class AgentResult:
    """Result from an agent execution."""
    answer: str
    tools_used: list[str]


class LLMClient:
    """Protocol for LLM clients - allows dependency injection for testing."""

    def chat(self, messages: list[dict[str, str]], tools: Optional[list[dict]] = None) -> dict[str, Any]:
        """Send chat messages and get response with potential tool calls."""
        raise NotImplementedError


class OllamaLLMClient(LLMClient):
    """LLM client using Ollama's OpenAI-compatible API."""

    def __init__(self, model: str = "llama3.2", api_base: Optional[str] = None, timeout: float = 30.0,
                 max_tokens: Optional[int] = None):
        self.model = model
        self.api_base = api_base or os.environ.get("OLLAMA_API_BASE", "http://localhost:11434/v1")
        self.timeout = timeout
        # Cap generation length so a verbose model can't run for a minute (env-overridable).
        self.max_tokens = max_tokens if max_tokens is not None else int(os.environ.get("LORE_MAX_TOKENS", "700"))
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "The 'openai' package is required to use OllamaLLMClient. "
                "Install it with: pip install openai"
            )
        self._client = OpenAI(base_url=self.api_base, api_key="ollama", timeout=self.timeout)

    def chat(self, messages: list[dict[str, str]], tools: Optional[list[dict]] = None) -> dict[str, Any]:
        """Send chat messages to Ollama and get response."""
        # Only send tools/tool_choice when tools are actually provided:
        # OpenAI-compatible endpoints reject tool_choice without tools.
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        response = self._client.chat.completions.create(**kwargs)
        return {
            "content": response.choices[0].message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
                for tc in response.choices[0].message.tool_calls
            ] if response.choices[0].message.tool_calls else [],
        }


class FakeLLMClient(LLMClient):
    """Fake LLM client for testing - returns scripted responses."""

    def __init__(self, scripted_response: Optional[dict[str, Any]] = None):
        """
        Initialize with a scripted response.
        
        scripted_response should be:
        {
            "content": "final answer text",
            "tool_calls": [{"name": "tool_name", "arguments": {"arg": "value"}}]
        }
        """
        self.scripted_response = scripted_response or {"content": "", "tool_calls": []}

    def chat(self, messages: list[dict[str, str]], tools: Optional[list[dict]] = None) -> dict[str, Any]:
        """Return the scripted response."""
        return self.scripted_response


class Agent:
    """Conduit Agent - ReAct loop with tool selection via local LLM."""

    MAX_ROUNDS = 4
    DEFAULT_TIMEOUT = 30.0

    def __init__(
        self,
        llm_client: LLMClient,
        tool_manager: Any,  # MCPManager - type hinted loosely for flexibility
        model: str = "llama3.2",
        max_rounds: int = 4,
        timeout: float = 30.0,
    ):
        """
        Initialize the agent.
        
        Args:
            llm_client: LLM client for generating tool calls (dependency-injected)
            tool_manager: MCPManager instance for executing tools
            model: Model name for Ollama (used if creating default client)
            max_rounds: Maximum number of tool-call rounds
            timeout: Timeout in seconds for LLM calls
        """
        self.llm_client = llm_client
        self.tool_manager = tool_manager
        self.model = model
        self.max_rounds = max_rounds
        self.timeout = timeout

    def _get_available_tools(self) -> list[dict]:
        """Get available tools from the tool manager."""
        # Assuming tool_manager has a get_tools() method that returns OpenAI-style tool definitions
        if hasattr(self.tool_manager, "get_tools"):
            return self.tool_manager.get_tools()
        return []

    def _execute_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool and return the result."""
        if hasattr(self.tool_manager, "execute"):
            return self.tool_manager.execute(tool_name, arguments)
        raise ValueError(f"Tool manager does not support execute method: {tool_name}")

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the agent."""
        tools = self._get_available_tools()
        tool_descriptions = "\n".join(
            f"- {t.get('name', 'unknown')}: {t.get('description', '')}"
            for t in tools
        )
        return f"""You are a helpful assistant that can use tools to answer questions.

Available tools:
{tool_descriptions}

Follow this process:
1. Think about what tool(s) you need to call
2. Call the tool(s) with appropriate arguments
3. Use the results to formulate your final answer

When you have your final answer, respond with just the answer text (no tool calls)."""

    def run(self, user_message: str) -> AgentResult:
        """
        Run the agent loop to answer a user message.
        
        Args:
            user_message: The user's question or request
            
        Returns:
            AgentResult with the final answer and list of tools used
        """
        start_time = time.time()
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": user_message},
        ]
        tools_used: list[str] = []
        tools = self._get_available_tools()

        for round_num in range(self.max_rounds):
            # Check timeout
            if time.time() - start_time > self.timeout:
                return AgentResult(
                    answer="Sorry, I timed out while processing your request.",
                    tools_used=tools_used,
                )

            # Get LLM response
            response = self.llm_client.chat(messages, tools=tools if tools else None)

            # Check if there are tool calls
            if response.get("tool_calls"):
                tool_calls = response["tool_calls"]

                # Append the assistant message that carries the tool calls so
                # each subsequent role:"tool" message is linked to it via
                # tool_call_id (strict endpoints reject unlinked tool messages).
                # Tolerate clients (e.g. FakeLLMClient) that omit "id".
                call_ids = [
                    tool_call.get("id") or f"call_{i}"
                    for i, tool_call in enumerate(tool_calls)
                ]
                messages.append({
                    "role": "assistant",
                    "content": response.get("content") or "",
                    "tool_calls": [
                        {
                            "id": call_ids[i],
                            "type": "function",
                            "function": {
                                "name": tool_call.get("name", ""),
                                "arguments": (
                                    tool_call.get("arguments")
                                    if isinstance(tool_call.get("arguments"), str)
                                    else json.dumps(tool_call.get("arguments") or {})
                                ),
                            },
                        }
                        for i, tool_call in enumerate(tool_calls)
                    ],
                })

                for i, tool_call in enumerate(tool_calls):
                    tool_name = tool_call.get("name", "")
                    arguments = tool_call.get("arguments", {})
                    # The API returns arguments as a JSON string; tools expect a dict.
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except (json.JSONDecodeError, ValueError):
                            arguments = {}

                    # Execute the tool
                    try:
                        result = self._execute_tool(tool_name, arguments)
                        tools_used.append(tool_name)
                        # Add tool result to messages, linked to its tool call
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call_ids[i],
                            "content": str(result),
                            "name": tool_name,
                        })
                    except Exception as e:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call_ids[i],
                            "content": f"Error executing {tool_name}: {str(e)}",
                            "name": tool_name,
                        })
            else:
                # No tool calls - this is the final answer (guard None → "")
                final_answer = response.get("content") or ""
                return AgentResult(answer=final_answer, tools_used=tools_used)

        # Max rounds reached without final answer
        return AgentResult(
            answer="I've reached the maximum number of tool calls. Please try a different approach.",
            tools_used=tools_used,
        )
