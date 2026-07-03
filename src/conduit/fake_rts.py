"""Fake RTS client for testing the research loop."""
from dataclasses import dataclass
from typing import Any, Optional
import hashlib

from conduit.rts_client import SearchHit


@dataclass
class CorpusMessage:
    """A message in the fake RTS corpus."""
    text: str
    channel: str
    ts: str
    author: Optional[str] = None


class FakeRTS:
    """Fake RTS client for testing.
    
    Provides keyword-based and naive semantic search over a corpus of messages.
    """
    
    # Default corpus with test messages that match common test queries
    DEFAULT_CORPUS: dict[str, list[CorpusMessage]] = {
        "general": [
            CorpusMessage(
                text="The quarterly review meeting is scheduled for Friday at 3pm.",
                channel="general",
                ts="1234567890.123456",
                author="alice",
            ),
            CorpusMessage(
                text="Welcome to the team! Please read the onboarding docs.",
                channel="general",
                ts="1234567891.123456",
                author="bob",
            ),
            CorpusMessage(
                text="Don't forget to submit your expense reports by the end of the month.",
                channel="general",
                ts="1234567892.123456",
                author="carol",
            ),
        ],
        "engineering": [
            CorpusMessage(
                text="The API deployment completed successfully. All tests passing.",
                channel="engineering",
                ts="1234567893.123456",
                author="dave",
            ),
            CorpusMessage(
                text="We need to review the pull request for the new feature branch.",
                channel="engineering",
                ts="1234567894.123456",
                author="eve",
            ),
        ],
        "random": [
            CorpusMessage(
                text="Anyone want to grab lunch tomorrow? There's a new sandwich place.",
                channel="random",
                ts="1234567895.123456",
                author="frank",
            ),
        ],
    }
    
    def __init__(self, corpus: Optional[dict[str, list[CorpusMessage]]] = None):
        """Initialize the fake RTS with a corpus.
        
        Args:
            corpus: A dict mapping channel names to lists of CorpusMessage.
                If None, uses the default test corpus.
        """
        self._corpus = corpus if corpus is not None else self.DEFAULT_CORPUS
    
    def search(self, query: str, limit: int = 10) -> list[SearchHit]:
        """Search the corpus for messages matching the query.
        
        Uses a simple keyword overlap scoring mechanism.
        
        Args:
            query: The search query string.
            limit: Maximum number of results to return.
            
        Returns:
            A list of SearchHit objects sorted by relevance score.
        """
        query_words = set(query.lower().split())
        results: list[tuple[CorpusMessage, float]] = []
        
        for channel, messages in self._corpus.items():
            for msg in messages:
                # Calculate keyword overlap score
                msg_words = set(msg.text.lower().split())
                overlap = len(query_words & msg_words)
                
                # Include results with at least 1 word overlap
                if overlap > 0:
                    # Score based on overlap ratio
                    score = overlap / max(len(query_words), 1)
                    results.append((msg, score))
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Convert to SearchHit objects
        hits: list[SearchHit] = []
        for msg, score in results[:limit]:
            permalink = self._make_permalink(msg.channel, msg.ts)
            hits.append(SearchHit(
                text=msg.text,
                channel=msg.channel,
                ts=msg.ts,
                permalink=permalink,
                score=score,
                author=msg.author,
            ))
        
        return hits
    
    def _make_permalink(self, channel: str, ts: str) -> str:
        """Create a stable permalink for a message."""
        # Use a hash to create a stable permalink
        hash_input = f"{channel}:{ts}"
        hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"https://slack.com/archives/{channel}/p{ts.replace('.', '')}?thread_ts={ts}&cid={hash_value}"
