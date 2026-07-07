"""LLM clients for Lore — an injectable ``LLMClient`` protocol with an Ollama-backed
implementation (OpenAI-compatible) and a deterministic fake for tests/offline demos.

Used across the research pipeline: decomposition + follow-up (``research.py``) and cited
synthesis (``citations.py``). Backend selection lives in ``slack_app._build_llm``.
"""
from __future__ import annotations

import os
from typing import Any, Optional


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
