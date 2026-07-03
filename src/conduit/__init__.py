"""Conduit — an MCP Agent for Slack."""
__version__ = "0.1.0"

from conduit.agent import Agent, AgentResult, FakeLLMClient, LLMClient, OllamaLLMClient
from conduit.rts_client import SearchHit, RTSClient
from conduit.fake_rts import FakeRTS, CorpusMessage

__all__ = [
    "Agent",
    "AgentResult",
    "FakeLLMClient",
    "LLMClient",
    "OllamaLLMClient",
    "SearchHit",
    "RTSClient",
    "FakeRTS",
    "CorpusMessage",
]
