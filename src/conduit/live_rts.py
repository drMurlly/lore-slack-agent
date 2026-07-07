"""Live search over real Slack workspace history — Lore's production retrieval substrate.

Why this exists
---------------
Slack's brand-new Real-Time Search (RTS) API is allowlisted and needs a ``search`` scope
our sandbox app doesn't hold. Rather than fake it, this client does the honest thing: it
reads actual channel history via ``conversations.history`` (the bot holds ``channels:history``
/ ``groups:history`` / ``im:history``), indexes it, and ranks messages by lexical relevance
+ recency — exposing the SAME ``.search(query, limit) -> list[SearchHit]`` seam the research
loop already consumes. So the whole pipeline (decompose → multi-hop → cited synthesis →
Canvas) runs on real workspace data with zero pipeline changes, and swaps to the official RTS
API by replacing this one class the day the scope is granted.

Permalinks are built in Slack's canonical archive form
``<team_url>/archives/<channel_id>/p<ts-without-dot>`` so every citation deep-links to the
exact source message with no extra API round-trip.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional, Union

from slack_sdk.errors import SlackApiError

from conduit.rts_client import SearchHit

logger = logging.getLogger(__name__)

# Lightweight stopword set so ranking keys on content words, not glue.
_STOP = {
    "the", "a", "an", "and", "or", "to", "of", "is", "are", "was", "were", "be", "been",
    "we", "our", "you", "i", "it", "on", "in", "for", "with", "about", "did", "do", "does",
    "what", "when", "why", "who", "how", "which", "that", "this", "at", "as", "by", "from",
    "any", "anything", "since", "so", "if", "but", "not", "have", "has", "had", "will",
}


def _tokens(text: str) -> list[str]:
    # ASCII words/values as before, PLUS runs of non-ASCII (CJK / Cyrillic / Arabic / Hebrew / …)
    # so non-Latin messages are indexable and searchable instead of silently tokenizing to nothing
    # (a Cyrillic query used to score 0 against every doc → empty results, no error). This is
    # additive on ASCII: pure-English text tokenizes exactly as before. casefold() folds case
    # across scripts. Leading/trailing '.' are stripped per token so a sentence-final word or value
    # ('price.', '$49.') matches the same clean term in a query — interior dots survive ('$10.50').
    out: list[str] = []
    for w in re.findall(r"[a-zA-Z0-9$%€£.]+|[^\x00-\x7f]+", (text or "").casefold()):
        w = w.strip(".")
        if w:
            out.append(w)
    return out


def _content_tokens(text: str) -> set[str]:
    return {t for t in _tokens(text) if t not in _STOP and len(t) > 1}


# DISTINCTIVE signatures of Lore's OWN Slack posts (answers, trace, canvas buttons). These get
# posted into channels and would otherwise be re-indexed as "evidence" — a feedback loop that
# pollutes retrieval with Lore's own prior answers. Each phrase here essentially never appears in
# an ordinary human message. (Removed the over-broad "current value is" substring and the bare
# [n]-bracket rule: humans legitimately write "tier [2] pricing" or "ticket [123]", and excluding
# those silently dropped real evidence — including, potentially, the decisive reversal message.)
_LORE_MARKERS = ("🔦 *Researching", "📄 *Final Answer*", "Research Trace", "View Full Canvas",
                 "It was later changed to", "An earlier value was",
                 "Lore resolves to the most recent decision")
_CITE_RE = re.compile(r"\[\d+\]")


def _is_lore_output(text: str) -> bool:
    if any(m in text for m in _LORE_MARKERS):  # a distinctive Lore signature is decisive on its own
        return True
    # A bare [n] bracket is NOT proof of Lore output (humans write "tier [2]", "ticket [123]"); only
    # treat it as ours when it co-occurs with the Decision-Graph badge / conflicting-signals framing.
    return bool(_CITE_RE.search(text)) and ("🕸️" in text or "Conflicting signals" in text)


class SlackHistoryRTS:
    """RTS-compatible search over live Slack channel history.

    Args:
        slack: a Slack WebClient exposing ``conversations_history`` (and optionally
            ``users_info`` for author display names).
        channels: channels to index — a list of channel IDs, or a ``{id: "#name"}`` map so
            citations render the human channel name (the bot can't ``conversations.list`` in
            this sandbox, so names are supplied by the caller that created the channels).
        team_url: workspace base URL (e.g. ``https://simon-ca18831.slack.com``) used to build
            archive permalinks.
        max_per_channel: history depth per channel to index.
    """

    def __init__(
        self,
        slack: Any,
        channels: Union[list[str], dict[str, str]],
        team_url: str = "",
        max_per_channel: int = 300,
    ):
        self.slack = slack
        if isinstance(channels, dict):
            self._channel_ids = list(channels.keys())
            self._channel_names = dict(channels)
        else:
            self._channel_ids = list(channels)
            self._channel_names = {}
        self.team_url = team_url.rstrip("/")
        self.max_per_channel = max_per_channel
        self._index: Optional[list[dict[str, Any]]] = None
        self._user_names: dict[str, str] = {}
        # Populated by refresh() so callers can surface "index degraded" traces.
        self.index_stats: dict[str, Any] = {"messages": 0, "channels": 0, "empty_channels": []}

    # -- indexing ---------------------------------------------------------- #
    def refresh(self) -> "SlackHistoryRTS":
        """(Re)load the in-memory index from live channel history. Returns self."""
        docs: list[dict[str, Any]] = []
        empty_channels: list[str] = []
        for cid in self._channel_ids:
            before = len(docs)
            for msg in self._history(cid):
                text = msg.get("text") or ""
                ts = msg.get("ts") or ""
                if not text or not ts or msg.get("subtype") == "channel_join":
                    continue
                if _is_lore_output(text):
                    continue  # never index Lore's own answers — that's a feedback loop
                docs.append({
                    "text": text,
                    "channel": self._channel_names.get(cid, cid),
                    "channel_id": cid,
                    "ts": ts,
                    "permalink": self._permalink(cid, ts, msg.get("thread_ts")),
                    "author": self._author(msg.get("user")),
                    "tokens": _content_tokens(text),
                })
            if len(docs) == before:
                empty_channels.append(cid)
        self._index = docs
        self.index_stats = {
            "messages": len(docs),
            "channels": len(self._channel_ids),
            "empty_channels": empty_channels,
        }
        logger.info(
            "indexed %d messages across %d channels; %d yielded zero messages%s",
            len(docs),
            len(self._channel_ids),
            len(empty_channels),
            f" (check scopes/membership: {empty_channels})" if empty_channels else "",
        )
        return self

    def _history(self, channel_id: str) -> list[dict[str, Any]]:
        """Fetch up to ``max_per_channel`` messages, paging through cursors. Best-effort."""
        out: list[dict[str, Any]] = []
        cursor = None
        try:
            while len(out) < self.max_per_channel:
                resp = self.slack.conversations_history(
                    channel=channel_id,
                    limit=min(200, self.max_per_channel - len(out)),
                    **({"cursor": cursor} if cursor else {}),
                )
                out.extend(resp.get("messages", []) or [])
                cursor = (resp.get("response_metadata") or {}).get("next_cursor")
                if not cursor:
                    break
        except SlackApiError as e:
            # Best-effort: one bad channel (missing_scope, not_in_channel, …) must not
            # kill indexing — but the failure has to be visible, not swallowed.
            error = e.response.get("error") if e.response is not None else "unknown_error"
            logger.warning(
                "conversations.history failed for channel %s: %s — indexed %d messages before failure",
                channel_id, error, len(out),
            )
        except Exception:
            logger.warning(
                "conversations.history failed for channel %s — indexed %d messages before failure",
                channel_id, len(out), exc_info=True,
            )
        return out

    def _permalink(self, channel_id: str, ts: str, thread_ts: Optional[str]) -> str:
        base = self.team_url or "https://slack.com"
        p = "p" + ts.replace(".", "")
        url = f"{base}/archives/{channel_id}/{p}"
        if thread_ts and thread_ts != ts:
            url += f"?thread_ts={thread_ts}&cid={channel_id}"
        return url

    def _author(self, user_id: Optional[str]) -> Optional[str]:
        if not user_id:
            return None
        if user_id in self._user_names:
            return self._user_names[user_id]
        name = user_id
        try:
            info = self.slack.users_info(user=user_id)
            u = info.get("user", {}) or {}
            name = u.get("real_name") or (u.get("profile") or {}).get("display_name") or user_id
        except Exception:
            pass
        self._user_names[user_id] = name
        return name

    # -- search ------------------------------------------------------------ #
    def search(self, query: str, limit: int = 10) -> list[SearchHit]:
        """Rank indexed messages against ``query`` by lexical overlap + recency."""
        if self._index is None:
            self.refresh()
        q_tokens = _content_tokens(query)
        # Raw fallback exists only to catch value tokens ($10, 20%) that the stopword filter
        # drops — restrict it to tokens with a digit so it never matches on glue words like
        # "we"/"the" (which would let recent, unrelated messages out-rank relevant old ones).
        raw_q = {t for t in _tokens(query) if any(c.isdigit() for c in t)}
        scored: list[tuple[float, dict[str, Any]]] = []
        for doc in self._index or []:
            score = self._score(q_tokens, raw_q, doc)
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda x: (x[0], x[1]["ts"]), reverse=True)
        hits = [self._to_hit(doc, score) for score, doc in scored[:limit]]
        return hits

    def _score(self, q_tokens: set[str], raw_q: set[str], doc: dict[str, Any]) -> float:
        d_tokens = doc["tokens"]
        if not q_tokens:
            return 0.0
        overlap = len(q_tokens & d_tokens)
        if overlap == 0:
            # exact value tokens ($10, 20%) may be filtered as short/glue — check raw too
            overlap = len(raw_q & set(_tokens(doc["text"])))
            if overlap == 0:
                return 0.0
        base = overlap / len(q_tokens)
        # small recency nudge so, all else equal, the latest message wins (matters for
        # "what's the CURRENT answer" — the reversal resolution reads the newest value).
        try:
            recency = min(float(doc["ts"]) / 1e12, 0.05)
        except (TypeError, ValueError):
            recency = 0.0
        return base + recency

    def _to_hit(self, doc: dict[str, Any], score: float) -> SearchHit:
        return SearchHit(
            text=doc["text"],
            channel=doc["channel"],
            ts=doc["ts"],
            permalink=doc["permalink"],
            score=round(score, 4),
            author=doc["author"],
        )
