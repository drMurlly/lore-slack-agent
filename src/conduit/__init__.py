"""Lore — deep research over your team's Slack memory (package name ``conduit`` is retained
for import stability)."""
__version__ = "0.1.0"

from conduit.agent import FakeLLMClient, LLMClient, OllamaLLMClient
from conduit.rts_client import SearchHit, RTSClient
from conduit.fake_rts import FakeRTS, CorpusMessage

__all__ = [
    "FakeLLMClient",
    "LLMClient",
    "OllamaLLMClient",
    "SearchHit",
    "RTSClient",
    "FakeRTS",
    "CorpusMessage",
]
