"""Coverage-hardening tests for the retrieval / LLM backends.

Targets the previously-uncovered branches in agent.py (the Ollama client),
mcp_manager.py (running-loop sync bridge, spec/None guards, glossary helpers),
and live_rts.py (SlackApiError handling, permalinks, author resolution, scoring
edge cases). All fakes — no network, no subprocess.
"""
import sys
import types

import pytest

import conduit.mcp_manager as mcp_manager
from conduit.agent import LLMClient, FakeLLMClient
from conduit.mcp_manager import (
    MCPManager,
    MCPServerSpec,
    _run_sync,
    _unwrap_tool_result,
    default_glossary_manager,
    find_glossary_server,
    lookup_glossary_terms,
)


# --------------------------------------------------------------------------- #
# agent.py — the LLMClient protocol + the Ollama-backed client
# --------------------------------------------------------------------------- #
def _install_fake_openai(monkeypatch, *, tool_calls=True):
    """Install a fake ``openai`` module so OllamaLLMClient runs with no SDK/network."""
    captured: dict = {}

    class _FakeCompletions:
        def create(self, **kwargs):
            captured["kwargs"] = kwargs
            msg = types.SimpleNamespace(content="the answer")
            if tool_calls:
                tc = types.SimpleNamespace(
                    id="call_1",
                    function=types.SimpleNamespace(name="lookup_terms", arguments='{"text": "x"}'),
                )
                msg.tool_calls = [tc]
            else:
                msg.tool_calls = None
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            captured["init"] = kwargs
            self.chat = types.SimpleNamespace(completions=_FakeCompletions())

    fake = types.ModuleType("openai")
    fake.OpenAI = _FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake)
    return captured


def test_llmclient_protocol_is_abstract():
    with pytest.raises(NotImplementedError):
        LLMClient().chat([{"role": "user", "content": "hi"}])


def test_fake_llm_returns_scripted_and_defaults():
    assert FakeLLMClient().chat([]) == {"content": "", "tool_calls": []}
    scripted = {"content": "hello", "tool_calls": []}
    assert FakeLLMClient(scripted).chat([]) is scripted


def test_ollama_client_chat_with_tools(monkeypatch):
    from conduit.agent import OllamaLLMClient
    captured = _install_fake_openai(monkeypatch, tool_calls=True)
    client = OllamaLLMClient(model="gemma3:27b", api_base="http://host/v1",
                             timeout=7.0, max_tokens=123)
    out = client.chat([{"role": "user", "content": "q"}], tools=[{"type": "function"}])

    assert out["content"] == "the answer"
    assert out["tool_calls"][0] == {"id": "call_1", "name": "lookup_terms",
                                    "arguments": '{"text": "x"}'}
    # tools present -> tool_choice auto is added, and max_tokens/model flow through
    assert captured["kwargs"]["tool_choice"] == "auto"
    assert captured["kwargs"]["max_tokens"] == 123
    assert captured["kwargs"]["model"] == "gemma3:27b"
    assert captured["init"]["base_url"] == "http://host/v1"


def test_ollama_client_chat_without_tools(monkeypatch):
    from conduit.agent import OllamaLLMClient
    captured = _install_fake_openai(monkeypatch, tool_calls=False)
    # No explicit max_tokens -> read from env default (LORE_MAX_TOKENS unset -> 700)
    monkeypatch.delenv("LORE_MAX_TOKENS", raising=False)
    monkeypatch.delenv("OLLAMA_API_BASE", raising=False)
    client = OllamaLLMClient(model="llama3.2")
    out = client.chat([{"role": "user", "content": "q"}])

    assert out["tool_calls"] == []          # no tool calls -> empty list
    assert "tools" not in captured["kwargs"]  # omitted when none supplied
    assert captured["kwargs"]["max_tokens"] == 700
    assert client.api_base.endswith("/v1")   # default api base applied


def test_ollama_client_requires_openai(monkeypatch):
    from conduit.agent import OllamaLLMClient
    # Force `from openai import OpenAI` to fail.
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(ImportError) as exc:
        OllamaLLMClient(model="llama3.2")
    assert "openai" in str(exc.value)


# --------------------------------------------------------------------------- #
# mcp_manager.py — sync bridge + guards + glossary helpers
# --------------------------------------------------------------------------- #
def test_unwrap_structured_non_result_dict():
    """A structuredContent dict that isn't the FastMCP {"result": ...} wrapper is returned as-is."""
    result = types.SimpleNamespace(structuredContent={"term": "ARR", "definition": "x"},
                                   content=[], isError=False)
    assert _unwrap_tool_result(result) == {"term": "ARR", "definition": "x"}


async def test_run_sync_inside_running_loop_returns_value():
    """When a loop is already running, _run_sync bridges via a worker thread."""
    async def coro():
        return 42
    assert _run_sync(coro(), timeout=5) == 42


async def test_run_sync_inside_running_loop_propagates_error():
    async def coro():
        raise ValueError("boom")
    with pytest.raises(ValueError):
        _run_sync(coro(), timeout=5)


async def test_list_tools_async_thread_path():
    """list_tools() called from inside a running loop exercises the thread bridge end-to-end."""
    from contextlib import asynccontextmanager

    class _Tool:
        def __init__(self, name):
            self.name = name
            self.description = "d"

    class _Session:
        async def list_tools(self):
            return types.SimpleNamespace(tools=[_Tool("a"), _Tool("b")])

    @asynccontextmanager
    async def factory(spec):
        yield _Session()

    mgr = MCPManager(servers=[MCPServerSpec(name="s", command="x")], session_factory=factory)
    tools = mgr.list_tools()
    assert {t["name"] for t in tools} == {"a", "b"}


def test_spec_selects_named_server():
    a = MCPServerSpec(name="a", command="x")
    b = MCPServerSpec(name="b", command="y")
    mgr = MCPManager(servers=[a, b])
    assert mgr._spec("b") is b
    assert mgr._spec(None) is a  # first configured


async def test_list_tools_async_raises_on_unknown_server():
    mgr = MCPManager(servers=[MCPServerSpec(name="a", command="x")], session_factory=lambda s: None)
    with pytest.raises(ValueError):
        await mgr.list_tools_async("nope")


async def test_call_tool_async_raises_on_unknown_server():
    mgr = MCPManager(servers=[MCPServerSpec(name="a", command="x")], session_factory=lambda s: None)
    with pytest.raises(ValueError):
        await mgr.call_tool_async("t", {}, server="nope")


def test_find_glossary_server_returns_none(monkeypatch):
    monkeypatch.setenv("LORE_GLOSSARY_SERVER", "/no/such/glossary.py")
    monkeypatch.setattr(mcp_manager.Path, "exists", lambda self: False)
    assert find_glossary_server() is None


def test_default_glossary_manager_without_server(monkeypatch):
    monkeypatch.setattr(mcp_manager, "find_glossary_server", lambda: None)
    mgr = default_glossary_manager()
    assert mgr.servers == {}


def test_lookup_glossary_terms_wraps_single_dict():
    class _DictMgr:
        def call_tool(self, name, args=None, **kwargs):
            return {"term": "ARR", "definition": "Annual Recurring Revenue"}
    assert lookup_glossary_terms("ARR", manager=_DictMgr()) == [
        {"term": "ARR", "definition": "Annual Recurring Revenue"}
    ]


def test_lookup_glossary_terms_non_list_returns_empty():
    class _StrMgr:
        def call_tool(self, name, args=None, **kwargs):
            return "not a list"
    assert lookup_glossary_terms("ARR", manager=_StrMgr()) == []


# --------------------------------------------------------------------------- #
# live_rts.py — indexing / history / author / scoring edge cases
# --------------------------------------------------------------------------- #
from conduit.live_rts import SlackHistoryRTS, _is_lore_output  # noqa: E402
from slack_sdk.errors import SlackApiError  # noqa: E402


class _FakeSlack:
    """Minimal Slack WebClient stand-in for SlackHistoryRTS."""

    def __init__(self, history_by_channel, users=None, history_error=None):
        self._history = history_by_channel
        self._users = users or {}
        self._history_error = history_error
        self.users_info_calls = 0

    def conversations_history(self, channel, limit=200, **kwargs):
        if self._history_error is not None:
            raise self._history_error
        return {"messages": self._history.get(channel, [])}

    def users_info(self, user):
        self.users_info_calls += 1
        if user in self._users:
            return {"user": self._users[user]}
        raise SlackApiError("boom", {"ok": False, "error": "user_not_found"})


def test_is_lore_output_flags_multi_citation():
    assert _is_lore_output("We set it [1] then changed [2].") is True
    assert _is_lore_output("adjacent [3][4] cite") is True
    assert _is_lore_output("View Full Canvas") is True
    # a single human bracket is NOT Lore output
    assert _is_lore_output("the tier [2] applies") is False


def test_refresh_skips_lore_output_join_and_empty_channels():
    history = {
        "C1": [
            {"text": "We priced it at $10 for launch.", "ts": "1000.000100", "user": "U1"},
            {"text": "Lore answer [1] and [2].", "ts": "1000.000200", "user": "U1"},  # skipped
            {"text": "joined", "ts": "1000.000300", "subtype": "channel_join", "user": "U2"},
            {"text": "", "ts": "1000.000400"},          # no text -> skipped
        ],
        "C2": [],  # empty channel -> recorded in empty_channels
    }
    rts = SlackHistoryRTS(
        _FakeSlack(history, users={"U1": {"real_name": "Maya"}}),
        channels={"C1": "pricing", "C2": "infra"},
        team_url="https://team.slack.com/",
    ).refresh()

    assert rts.index_stats["messages"] == 1          # only the one real human message
    assert rts.index_stats["channels"] == 2
    assert "C2" in rts.index_stats["empty_channels"]

    hits = rts.search("pricing launch")
    assert hits and hits[0].channel == "pricing"
    assert hits[0].author == "Maya"                  # users_info real_name resolved
    assert hits[0].permalink == "https://team.slack.com/archives/C1/p1000000100"


def test_history_survives_slack_api_error():
    err = SlackApiError("nope", {"ok": False, "error": "missing_scope"})
    rts = SlackHistoryRTS(_FakeSlack({}, history_error=err), channels=["C1"]).refresh()
    assert rts.index_stats["messages"] == 0
    assert rts.index_stats["empty_channels"] == ["C1"]


def test_permalink_threaded_message_carries_thread_params():
    history = {"C1": [{"text": "reply about pricing", "ts": "2000.000200",
                       "thread_ts": "2000.000100", "user": "U1"}]}
    rts = SlackHistoryRTS(_FakeSlack(history), channels={"C1": "pricing"},
                          team_url="https://t.slack.com").refresh()
    hit = rts.search("pricing")[0]
    assert "thread_ts=2000.000100" in hit.permalink and "cid=C1" in hit.permalink


def test_author_caches_and_falls_back_to_user_id():
    history = {"C1": [
        {"text": "first pricing note", "ts": "10.0001", "user": "U9"},
        {"text": "second pricing note", "ts": "10.0002", "user": "U9"},
        {"text": "third pricing note", "ts": "10.0003"},  # no user -> author None
    ]}
    slack = _FakeSlack(history)  # users_info raises -> author falls back to the id
    rts = SlackHistoryRTS(slack, channels=["C1"]).refresh()
    # U9 looked up once despite appearing twice (cached), and resolves to the id on failure.
    assert slack.users_info_calls == 1
    authors = {d["author"] for d in rts._index}
    assert "U9" in authors and None in authors


def test_search_empty_query_returns_nothing():
    history = {"C1": [{"text": "some content here", "ts": "1.0", "user": "U1"}]}
    rts = SlackHistoryRTS(_FakeSlack(history), channels=["C1"]).refresh()
    assert rts.search("the a an of") == []  # all-stopword query -> no q_tokens -> no hits


def test_score_tolerates_nonnumeric_ts():
    """A non-numeric ts must not crash scoring (recency degrades to 0)."""
    history = {"C1": [{"text": "pricing decision made", "ts": "not-a-number", "user": "U1"}]}
    rts = SlackHistoryRTS(_FakeSlack(history), channels=["C1"]).refresh()
    hits = rts.search("pricing decision")
    assert hits and hits[0].text == "pricing decision made"


def test_search_refreshes_lazily_when_index_missing():
    history = {"C1": [{"text": "lazy pricing index", "ts": "1.0", "user": "U1"}]}
    rts = SlackHistoryRTS(_FakeSlack(history), channels=["C1"])  # no refresh() yet
    hits = rts.search("pricing")                                  # triggers refresh internally
    assert hits and hits[0].text == "lazy pricing index"
