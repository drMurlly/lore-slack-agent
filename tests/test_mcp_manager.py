"""Tests for the MCP client manager, the glossary MCP server, and the
research-pipeline glossary consult.

Covers:
  * unit round-trip with an injected fake session (no process spawn),
  * a REAL stdio integration test that launches servers/glossary_server.py
    via the official MCP SDK (skipped only if the server can't be spawned),
  * graceful no-op degradation on bogus/missing servers,
  * research.run() wiring: glossary entries attach to ResearchResult and a
    trace step is emitted, while the consult stays off by default.
"""

import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from conduit.mcp_manager import (
    MCPManager,
    MCPServerSpec,
    default_glossary_manager,
    find_glossary_server,
    glossary_enabled,
    lookup_glossary_terms,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
GLOSSARY_SERVER = REPO_ROOT / "servers" / "glossary_server.py"


# --------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------- #
class _FakeTool:
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description


class _FakeToolsResult:
    def __init__(self, tools):
        self.tools = tools


class _FakeCallResult:
    def __init__(self, structured=None, is_error=False):
        self.structuredContent = structured
        self.content = []
        self.isError = is_error


class _FakeSession:
    """Duck-typed stand-in for mcp.ClientSession."""

    def __init__(self):
        self.calls = []

    async def list_tools(self):
        return _FakeToolsResult(
            [_FakeTool("lookup_terms", "Find glossary terms"), _FakeTool("define", "Define one term")]
        )

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return _FakeCallResult(
            structured={"result": [{"term": "ARR", "definition": "Annual Recurring Revenue"}]}
        )


def _manager_with_fake_session(session):
    @asynccontextmanager
    async def factory(spec):
        yield session

    spec = MCPServerSpec(name="glossary", command="unused", args=[])
    return MCPManager(servers=[spec], session_factory=factory)


# --------------------------------------------------------------------- #
# (a) Unit: injected fake session round-trip
# --------------------------------------------------------------------- #
class TestFakeSessionRoundTrip:
    def test_list_tools_round_trip(self):
        manager = _manager_with_fake_session(_FakeSession())
        tools = manager.list_tools()
        assert {t["name"] for t in tools} == {"lookup_terms", "define"}
        assert all("description" in t for t in tools)

    def test_call_tool_round_trip_unwraps_structured_result(self):
        session = _FakeSession()
        manager = _manager_with_fake_session(session)
        result = manager.call_tool("lookup_terms", {"text": "What is our ARR?"})
        # FastMCP-style {"result": ...} wrapper is unwrapped to plain data
        assert result == [{"term": "ARR", "definition": "Annual Recurring Revenue"}]
        # The exact tool name + arguments made it through the session
        assert session.calls == [("lookup_terms", {"text": "What is our ARR?"})]

    def test_call_tool_error_result_degrades_to_default(self):
        class _ErrSession(_FakeSession):
            async def call_tool(self, name, arguments):
                return _FakeCallResult(is_error=True)

        manager = _manager_with_fake_session(_ErrSession())
        assert manager.call_tool("lookup_terms", {"text": "x"}, default=[]) == []


# --------------------------------------------------------------------- #
# (b) Integration: REAL stdio server via the official MCP SDK
# --------------------------------------------------------------------- #
def _can_spawn_glossary_server() -> bool:
    if not GLOSSARY_SERVER.exists():
        return False
    try:
        proc = subprocess.run(
            [sys.executable, "-c", "import mcp.server.fastmcp"],
            capture_output=True,
            timeout=60,
        )
        return proc.returncode == 0
    except Exception:
        return False


class TestRealStdioIntegration:
    def test_glossary_server_stdio_round_trip(self):
        """Launch servers/glossary_server.py over stdio and call its tools."""
        pytest.importorskip("mcp.client.stdio")
        if not _can_spawn_glossary_server():
            pytest.skip("environment cannot spawn the glossary MCP server")

        manager = MCPManager(
            servers=[
                MCPServerSpec(
                    name="glossary", command=sys.executable, args=[str(GLOSSARY_SERVER)]
                )
            ],
            timeout=60,
        )

        tools = manager.list_tools()
        assert {t["name"] for t in tools} >= {"lookup_terms", "define"}

        entries = manager.call_tool(
            "lookup_terms", {"text": "What did we decide about ARR and the SSO rollout?"}
        )
        by_term = {e["term"]: e["definition"] for e in entries}
        assert "Annual Recurring Revenue" in by_term["ARR"]
        assert "Single Sign-On" in by_term["SSO"]

        definition = manager.call_tool("define", {"term": "mau"})
        assert "Monthly Active Users" in definition

    def test_default_glossary_manager_convenience(self):
        """lookup_glossary_terms() end-to-end against the bundled server."""
        pytest.importorskip("mcp.client.stdio")
        if not _can_spawn_glossary_server():
            pytest.skip("environment cannot spawn the glossary MCP server")

        assert find_glossary_server() == GLOSSARY_SERVER
        entries = lookup_glossary_terms(
            "pricing tier change and churn", manager=default_glossary_manager(timeout=60)
        )
        assert {e["term"] for e in entries} == {"pricing tier", "churn"}


# --------------------------------------------------------------------- #
# (c) Degradation: never crash the Slack app
# --------------------------------------------------------------------- #
class TestGracefulDegradation:
    def test_bogus_server_command_is_a_noop(self):
        manager = MCPManager(
            servers=[MCPServerSpec(name="bogus", command="/nonexistent/not-a-binary-xyz")],
            timeout=5,
        )
        assert manager.list_tools() == []
        assert manager.call_tool("lookup_terms", {"text": "ARR"}) is None
        assert manager.call_tool("lookup_terms", {"text": "ARR"}, default=[]) == []

    def test_no_servers_configured_is_a_noop(self):
        manager = MCPManager(servers=[])
        assert manager.available() is False
        assert manager.list_tools() == []
        assert manager.call_tool("anything") is None

    def test_lookup_glossary_terms_swallows_raising_manager(self):
        class _Exploding:
            def call_tool(self, name, args=None, **kwargs):
                raise RuntimeError("boom")

        assert lookup_glossary_terms("ARR", manager=_Exploding()) == []

    def test_lookup_glossary_terms_filters_non_dict_junk(self):
        class _Junk:
            def call_tool(self, name, args=None, **kwargs):
                return ["not-a-dict", {"term": "ARR", "definition": "ok"}, 42]

        assert lookup_glossary_terms("ARR", manager=_Junk()) == [
            {"term": "ARR", "definition": "ok"}
        ]

    def test_glossary_flag_defaults_off(self, monkeypatch):
        monkeypatch.delenv("LORE_MCP_GLOSSARY", raising=False)
        assert glossary_enabled() is False
        monkeypatch.setenv("LORE_MCP_GLOSSARY", "1")
        assert glossary_enabled() is True


# --------------------------------------------------------------------- #
# (d) Pipeline wiring: research.run consults the glossary via MCP
# --------------------------------------------------------------------- #
from conduit.research import run  # noqa: E402
from conduit.rts_client import SearchHit  # noqa: E402


class _StubRTS:
    def search(self, query, limit=5):
        return [
            SearchHit(
                text="We set the ARR target in the Q3 planning thread.",
                channel="general",
                ts="1700000000.000100",
                permalink="https://example.slack.com/archives/C1/p1700000000000100",
                score=0.9,
                author="alice",
            )
        ]


class _StubLLM:
    def chat(self, messages, tools=None):
        return {"content": "ARR target Q3\nrevenue planning thread", "tool_calls": None}


class _StubAssistant:
    def __init__(self):
        self.traces = []
        self.statuses = []

    def set_status(self, status):
        self.statuses.append(status)

    def emit_trace(self, phase, detail):
        self.traces.append((phase, detail))


class _StubGlossaryManager:
    def __init__(self):
        self.calls = []

    def call_tool(self, name, args=None, **kwargs):
        self.calls.append((name, args))
        return [{"term": "ARR", "definition": "Annual Recurring Revenue"}]


class TestResearchPipelineWiring:
    def test_run_attaches_glossary_and_emits_trace(self, monkeypatch):
        monkeypatch.delenv("LORE_MCP_GLOSSARY", raising=False)
        manager = _StubGlossaryManager()
        assistant = _StubAssistant()

        result = run(
            "What is our ARR target?",
            _StubRTS(),
            _StubLLM(),
            follow_up_threshold=0,
            assistant=assistant,
            glossary=manager,
        )

        assert result.glossary == [{"term": "ARR", "definition": "Annual Recurring Revenue"}]
        assert manager.calls == [("lookup_terms", {"text": "What is our ARR target?"})]
        glossary_traces = [t for t in assistant.traces if t[0] == "glossary"]
        assert len(glossary_traces) == 1
        assert "1 term(s) via MCP" in glossary_traces[0][1]
        assert "ARR" in glossary_traces[0][1]

    def test_run_defaults_to_no_glossary_consult(self, monkeypatch):
        monkeypatch.delenv("LORE_MCP_GLOSSARY", raising=False)
        result = run("What is our ARR target?", _StubRTS(), _StubLLM(), follow_up_threshold=0)
        assert result.glossary == []

    def test_env_flag_enables_consult_in_live_use(self, monkeypatch):
        monkeypatch.setenv("LORE_MCP_GLOSSARY", "1")
        # Patch the MCP lookup so this stays a fast unit test (no spawn);
        # research._consult_glossary re-imports it per call, so this is seen.
        monkeypatch.setattr(
            "conduit.mcp_manager.lookup_glossary_terms",
            lambda text, manager=None: [{"term": "SSO", "definition": "Single Sign-On"}],
        )
        result = run("When is the SSO rollout?", _StubRTS(), _StubLLM(), follow_up_threshold=0)
        assert result.glossary == [{"term": "SSO", "definition": "Single Sign-On"}]

    def test_run_survives_broken_glossary_manager(self, monkeypatch):
        monkeypatch.delenv("LORE_MCP_GLOSSARY", raising=False)

        class _Exploding:
            def call_tool(self, name, args=None, **kwargs):
                raise RuntimeError("MCP down")

        result = run(
            "What is our ARR target?",
            _StubRTS(),
            _StubLLM(),
            follow_up_threshold=0,
            glossary=_Exploding(),
        )
        # Research still completes with evidence; glossary quietly empty.
        assert result.glossary == []
        assert len(result.evidence) >= 1
