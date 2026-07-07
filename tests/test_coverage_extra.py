"""Supplemental coverage: rts_client HTTP seam, dedup guard, MCP text unwrap,
notify throttle-parse failure, live_rts generic history error, and blocks edges."""
import json
import types
import urllib.error

import pytest

import conduit.rts_client as rts_client
from conduit.rts_client import RTSClient


# --------------------------------------------------------------------------- #
# rts_client._http — the real HTTP seam (mocked transport)
# --------------------------------------------------------------------------- #
class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


def test_http_builds_request_and_parses_json(monkeypatch):
    seen = {}

    def _fake_urlopen(req):
        seen["url"] = req.full_url
        seen["auth"] = req.headers.get("Authorization")
        return _FakeResp({"ok": True, "messages": {"matches": []}})

    monkeypatch.setattr(rts_client.urllib.request, "urlopen", _fake_urlopen)
    client = RTSClient(token="xoxp-user")
    out = client._http("search.messages", {"query": "pricing", "count": 5})
    assert out == {"ok": True, "messages": {"matches": []}}
    assert "search.messages?" in seen["url"] and "query=pricing" in seen["url"]
    assert seen["auth"] == "Bearer xoxp-user"


def test_search_end_to_end_through_http(monkeypatch):
    payload = {"ok": True, "messages": {"matches": [
        {"text": "priced at $20", "channel": {"id": "C1", "name": "pricing"},
         "ts": "1.0", "permalink": "https://x/p1", "score": 9.0, "username": "priya"},
    ]}}
    monkeypatch.setattr(rts_client.urllib.request, "urlopen", lambda req: _FakeResp(payload))
    hits = RTSClient(token="xoxp-user").search("pricing")
    assert hits[0].channel == "pricing" and hits[0].author == "priya"


def test_http_reraises_http_error(monkeypatch):
    def _boom(req):
        raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", {}, None)
    monkeypatch.setattr(rts_client.urllib.request, "urlopen", _boom)
    with pytest.raises(urllib.error.HTTPError):
        RTSClient(token="xoxp-user")._http("search.messages", {"query": "q"})


def test_channel_name_string_shape():
    # Older API shapes return `channel` as a bare string, not a {"id","name"} dict.
    assert RTSClient._channel_name({"channel": "general"}) == "general"
    assert RTSClient._channel_name({}) == ""


# --------------------------------------------------------------------------- #
# dedup — empty id is never a duplicate
# --------------------------------------------------------------------------- #
def test_dedup_empty_id_never_duplicate():
    from conduit.dedup import EventDedup
    d = EventDedup()
    assert d.is_seen("") is False
    assert d.is_seen("") is False  # still not a duplicate — empty ids don't collide


def test_dedup_evicts_and_flags_repeat():
    from conduit.dedup import EventDedup
    d = EventDedup(maxsize=2)
    assert d.is_seen("a") is False
    assert d.is_seen("a") is True      # repeat flagged
    d.is_seen("b")
    d.is_seen("c")                     # over capacity -> oldest ("a") evicted
    assert d.is_seen("a") is False     # "a" is fresh again after eviction


# --------------------------------------------------------------------------- #
# mcp_manager._unwrap_tool_result — text-content fallbacks
# --------------------------------------------------------------------------- #
def test_unwrap_single_text_json_decodes():
    from conduit.mcp_manager import _unwrap_tool_result
    item = types.SimpleNamespace(text='{"term": "ARR"}')
    result = types.SimpleNamespace(structuredContent=None, content=[item], isError=False)
    assert _unwrap_tool_result(result) == {"term": "ARR"}


def test_unwrap_single_text_non_json_returns_text():
    from conduit.mcp_manager import _unwrap_tool_result
    item = types.SimpleNamespace(text="just a string")
    result = types.SimpleNamespace(structuredContent=None, content=[item], isError=False)
    assert _unwrap_tool_result(result) == "just a string"


def test_unwrap_multiple_texts_returns_list():
    from conduit.mcp_manager import _unwrap_tool_result
    items = [types.SimpleNamespace(text="a"), types.SimpleNamespace(text="b")]
    result = types.SimpleNamespace(structuredContent=None, content=items, isError=False)
    assert _unwrap_tool_result(result) == ["a", "b"]


def test_unwrap_error_result_raises():
    from conduit.mcp_manager import _unwrap_tool_result
    item = types.SimpleNamespace(text="tool exploded")
    result = types.SimpleNamespace(structuredContent=None, content=[item], isError=True)
    with pytest.raises(RuntimeError):
        _unwrap_tool_result(result)


# --------------------------------------------------------------------------- #
# notify — a malformed throttle interval must not escape _dispatch
# --------------------------------------------------------------------------- #
def test_dispatch_swallows_bad_min_interval(monkeypatch):
    import conduit.notify as notify
    notify._last = 0.0
    notify._suppressed = 0
    monkeypatch.setenv("LORE_NOTIFY_SLACK_USER", "UOP")
    monkeypatch.setenv("LORE_NOTIFY_MIN_INTERVAL", "not-a-float")
    # float("not-a-float") raises inside the try -> caught, no exception escapes.
    notify._dispatch("/lore", user="U2", text="q", channel="c", client=None)


# --------------------------------------------------------------------------- #
# live_rts — a non-SlackApiError during history is caught too
# --------------------------------------------------------------------------- #
def test_history_survives_generic_exception():
    from conduit.live_rts import SlackHistoryRTS

    class _Slack:
        def conversations_history(self, channel, limit=200, **kwargs):
            raise ValueError("unexpected boom")

    rts = SlackHistoryRTS(_Slack(), channels=["C1"]).refresh()
    assert rts.index_stats["messages"] == 0  # generic exception swallowed, index still built


# --------------------------------------------------------------------------- #
# blocks — money-shot with a raising graph, error card last-step, detail-less trace
# --------------------------------------------------------------------------- #
def test_money_shot_blocks_survive_raising_graph():
    from conduit.blocks import build_money_shot_blocks
    from conduit.citations import Answer

    class _BadGraph:
        def decision_rows(self, question):
            raise RuntimeError("graph down")

    answer = Answer(text="Answer [1].", citations=[])
    blocks = build_money_shot_blocks(answer, graph=_BadGraph(), question="q")
    assert isinstance(blocks, list)  # rows fell back to [] without raising


def test_error_blocks_include_last_step():
    from conduit.blocks import build_error_blocks
    blocks = build_error_blocks("ValueError", last_step="searching #pricing")
    assert any("searching #pricing" in str(b) for b in blocks)


def test_trace_block_without_detail():
    from conduit.blocks import trace_block, TraceStep
    block = trace_block(TraceStep("decompose", ""))
    assert "Decompose" in block["text"]["text"]
    assert ":" not in block["text"]["text"].split("Decompose")[-1]  # no "detail" suffix
