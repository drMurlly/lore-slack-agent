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
    """The **official Slack Search API** backend for the retrieval seam.

    Calls Slack's first-party ``search.messages`` full-text search across every channel the
    caller can see, and exposes the same ``search(query, limit) -> list[SearchHit]`` seam as
    :class:`conduit.live_rts.SlackHistoryRTS` and :class:`conduit.fake_rts.FakeRTS` — so the
    whole research pipeline runs on it with zero changes, selected at startup by
    ``_build_rts`` (see slack_app.py). The HTTP call is isolated behind the ``_http`` seam for
    easy mocking in tests.

    Token type (important): ``search.messages`` is a **user-scoped** method — it needs a Slack
    **user token** (``xoxp-…``) carrying the ``search:read`` scope. Slack rejects **bot tokens**
    (``xoxb-…``) with ``not_allowed_token_type`` by platform rule (not a Lore limitation), so
    construct this with a user token (``SLACK_USER_TOKEN``). When only a bot token is available,
    Lore uses the ``SlackHistoryRTS`` backend (``conversations.history`` + a local ranker)
    behind this same seam — both are real; only the substrate differs.
    """

    def __init__(self, token: str, api_base: str = "https://slack.com/api"):
        """Initialize the RTS client.

        Args:
            token: Slack **user** token (``xoxp-…``) with the ``search:read`` scope.
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

    @staticmethod
    def _channel_name(match: dict[str, Any]) -> str:
        """search.messages returns ``channel`` as ``{"id","name",…}``; older shapes as a str."""
        chan = match.get("channel")
        if isinstance(chan, dict):
            return chan.get("name") or chan.get("id") or ""
        return chan or ""

    def search(self, query: str, limit: int = 10) -> list[SearchHit]:
        """Search Slack messages via the official ``search.messages`` API.

        Args:
            query: The search query string.
            limit: Maximum number of results to return.

        Returns:
            A list of SearchHit objects ranked by Slack's relevance score.

        Raises:
            RuntimeError: if the Slack API responds with ``ok: false``. The common case is
                ``not_allowed_token_type`` — a **bot** token was used; ``search.messages``
                requires a **user** token (``xoxp-…``) with ``search:read``. The message spells
                this out so the failure is actionable rather than silently returning ``[]``.
        """
        # search.messages paginates with `count` (not `limit`); sort by relevance score.
        result = self._http("search.messages", {
            "query": query,
            "count": limit,
            "sort": "score",
            "sort_dir": "desc",
        })

        if not result.get("ok"):
            error = result.get("error", "unknown_error")
            hint = ""
            if error == "not_allowed_token_type":
                hint = (" — search.messages needs a USER token (xoxp-…) with search:read; "
                        "set SLACK_USER_TOKEN, or omit it to use the SlackHistoryRTS backend")
            raise RuntimeError(f"search.messages failed: {error}{hint}")

        hits = []
        messages = result.get("messages") or {}
        for msg in (messages.get("matches") or [])[:limit]:
            hit = SearchHit(
                text=msg.get("text", ""),
                channel=self._channel_name(msg),
                ts=msg.get("ts", ""),
                permalink=msg.get("permalink", ""),
                score=float(msg.get("score", 0) or 0),
                author=msg.get("username") or msg.get("user") or None,
            )
            hits.append(hit)

        return hits
