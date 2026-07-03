"""Real-Time Search client for Slack API."""
from dataclasses import dataclass
from typing import Any, Optional
import json
import urllib.request
import urllib.error


@dataclass
class SearchHit:
    """A search result hit from Slack's Real-Time Search API."""
    text: str
    channel: str
    ts: str
    permalink: str
    score: float
    author: Optional[str] = None


class RTSClient:
    """Client for Slack's Real-Time Search API.
    
    Wraps the semantic + keyword search over channels, messages, and files.
    The HTTP call is isolated behind _http seam for easy mocking in tests.
    """
    
    def __init__(self, token: str, api_base: str = "https://slack.com/api"):
        """Initialize the RTS client with a Slack bot token.
        
        Args:
            token: Slack bot token for authentication.
            api_base: Base URL for the Slack API.
        """
        self.token = token
        self.api_base = api_base
    
    def _http(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Make an HTTP request to the Slack API.
        
        This is the seam for mocking in tests.
        
        Args:
            method: API method name (e.g., 'search.messages').
            params: Query parameters for the request.
            
        Returns:
            The JSON response from the API.
        """
        from urllib.parse import urlencode
        url = f"{self.api_base}/{method}"
        query_string = urlencode(params)
        full_url = f"{url}?{query_string}"
        req = urllib.request.Request(full_url, headers={"Authorization": f"Bearer {self.token}"})
        
        try:
            with urllib.request.urlopen(req) as resp:
                data = resp.read().decode('utf-8')
                return json.loads(data)
        except urllib.error.HTTPError as e:
            # urlopen already raises HTTPError for non-2xx responses, mimicking raise_for_status()
            raise
    
    def search(self, query: str, limit: int = 10) -> list[SearchHit]:
        """Search Slack messages, channels, and files.
        
        Args:
            query: The search query string.
            limit: Maximum number of results to return.
            
        Returns:
            A list of SearchHit objects ranked by relevance.

        Raises:
            RuntimeError: if the Slack API responds with ``ok: false`` (e.g. the legacy
                ``search.messages`` endpoint rejects bot tokens with ``not_allowed_token_type``).
        """
        # Legacy search.messages paginates with `count`, not `limit`.
        result = self._http("search.messages", {
            "query": query,
            "count": limit,
        })

        if not result.get("ok"):
            raise RuntimeError(f"search failed: {result.get('error')}")

        hits = []
        if "messages" in result and "matches" in result["messages"]:
            for msg in result["messages"]["matches"][:limit]:
                hit = SearchHit(
                    text=msg.get("text", ""),
                    channel=msg.get("channel", {}).get("name", ""),
                    ts=msg.get("ts", ""),
                    permalink=msg.get("permalink", ""),
                    score=float(msg.get("score", 0)),
                    author=msg.get("user", None),
                )
                hits.append(hit)
        
        return hits
