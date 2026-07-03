"""Minimal MCP client manager for Lore, built on the official ``mcp`` SDK.

Connects to stdio MCP servers (e.g. ``servers/glossary_server.py``), lists
their tools, and calls them. The SDK is async; the research loop and Slack
handlers are sync, so this module exposes synchronous ``list_tools`` /
``call_tool`` wrappers that manage the async session under the hood (the
async core stays available as ``*_async`` methods).

Design constraints honored here:

* **Never crash the Slack app.** If the ``mcp`` import fails, a server binary
  is missing, or a call times out, the sync wrappers log a warning and return
  a benign default (``[]`` / ``None``) instead of raising.
* **Dependency-injectable.** Tests can pass ``session_factory`` — any async
  context manager factory yielding an object with ``list_tools()`` /
  ``call_tool()`` — to exercise the round-trip without spawning a process.
* **Connection per call.** Each operation opens a fresh stdio session and
  tears it down. For Lore that's one glossary consult per question (~a
  subprocess spawn), which keeps lifecycle management trivial and robust.
"""

import asyncio
import json
import logging
import os
import sys
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:  # Degrade gracefully if the official SDK is unavailable.
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    MCP_AVAILABLE = True
except Exception as _import_error:  # pragma: no cover - only without the SDK
    ClientSession = None  # type: ignore[assignment]
    StdioServerParameters = None  # type: ignore[assignment]
    stdio_client = None  # type: ignore[assignment]
    MCP_AVAILABLE = False
    logger.warning("mcp SDK unavailable (%s) — MCP tools disabled", _import_error)

#: Env var holding an explicit path to the glossary MCP server script.
GLOSSARY_SERVER_ENV = "LORE_GLOSSARY_SERVER"
#: Env var gating the glossary consult in live use (set to "1"/"true" to enable).
GLOSSARY_FLAG_ENV = "LORE_MCP_GLOSSARY"

_TRUTHY = {"1", "true", "yes", "on"}


@dataclass
class MCPServerSpec:
    """Launch spec for one stdio MCP server (mirrors config.ServerConfig)."""

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: Optional[dict[str, str]] = None


def _unwrap_tool_result(result: Any) -> Any:
    """Convert an SDK ``CallToolResult`` into plain Python data.

    Prefers ``structuredContent`` (FastMCP wraps non-object returns as
    ``{"result": ...}``); falls back to JSON-decoding text content.
    """
    if getattr(result, "isError", False):
        texts = [getattr(c, "text", "") for c in (getattr(result, "content", None) or [])]
        raise RuntimeError("MCP tool returned an error: " + " ".join(t for t in texts if t))

    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        if set(structured) == {"result"}:
            return structured["result"]
        return structured

    texts = []
    for item in getattr(result, "content", None) or []:
        text = getattr(item, "text", None)
        if text is not None:
            texts.append(text)
    if len(texts) == 1:
        try:
            return json.loads(texts[0])
        except (ValueError, TypeError):
            return texts[0]
    return texts


def _run_sync(coro: Any, timeout: float) -> Any:
    """Run an async coroutine from sync code, even if a loop is already running."""

    async def _with_timeout() -> Any:
        return await asyncio.wait_for(coro, timeout)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_with_timeout())

    # Already inside an event loop (e.g. async Slack handler): use a fresh
    # thread with its own loop rather than blocking the running one.
    box: dict[str, Any] = {}

    def _target() -> None:
        try:
            box["value"] = asyncio.run(_with_timeout())
        except BaseException as exc:  # relayed to the caller below
            box["error"] = exc

    thread = threading.Thread(target=_target, name="mcp-sync-call", daemon=True)
    thread.start()
    thread.join(timeout + 5)
    if "error" in box:
        raise box["error"]
    if "value" not in box:
        raise TimeoutError(f"MCP call did not complete within {timeout}s")
    return box["value"]


class MCPManager:
    """Client manager for one or more configured stdio MCP servers.

    Args:
        servers: launch specs (``MCPServerSpec`` or config ``ServerConfig`` —
            anything with ``name``/``command``/``args``/``env``).
        session_factory: optional async-context-manager factory
            ``factory(spec) -> session`` for tests; bypasses process spawning.
        timeout: per-operation wall-clock budget in seconds.
    """

    def __init__(
        self,
        servers: Optional[list[Any]] = None,
        session_factory: Any = None,
        timeout: float = 20.0,
    ) -> None:
        self.servers: dict[str, Any] = {s.name: s for s in (servers or [])}
        self._session_factory = session_factory
        self.timeout = timeout

    def available(self) -> bool:
        """True if at least one server is configured and a transport exists."""
        return bool(self.servers) and (MCP_AVAILABLE or self._session_factory is not None)

    def _spec(self, server: Optional[str]) -> Any:
        if server is not None:
            return self.servers.get(server)
        return next(iter(self.servers.values()), None)

    @asynccontextmanager
    async def _open_session(self, spec: Any):
        """Open an initialized MCP session to ``spec`` (real stdio or injected)."""
        if self._session_factory is not None:
            async with self._session_factory(spec) as session:
                yield session
            return
        params = StdioServerParameters(
            command=spec.command, args=list(spec.args or []), env=spec.env
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session

    # ------------------------------------------------------------------ #
    # Async core
    # ------------------------------------------------------------------ #
    async def list_tools_async(self, server: Optional[str] = None) -> list[dict[str, str]]:
        """List tools on a configured server as ``[{"name", "description"}]``."""
        spec = self._spec(server)
        if spec is None:
            raise ValueError(f"No MCP server configured matching {server!r}")
        async with self._open_session(spec) as session:
            result = await session.list_tools()
        return [
            {"name": t.name, "description": getattr(t, "description", None) or ""}
            for t in result.tools
        ]

    async def call_tool_async(
        self, name: str, args: Optional[dict[str, Any]] = None, server: Optional[str] = None
    ) -> Any:
        """Call a tool on a configured server and return plain Python data."""
        spec = self._spec(server)
        if spec is None:
            raise ValueError(f"No MCP server configured matching {server!r}")
        async with self._open_session(spec) as session:
            result = await session.call_tool(name, args or {})
        return _unwrap_tool_result(result)

    # ------------------------------------------------------------------ #
    # Sync convenience wrappers (never raise — the Slack app must not crash)
    # ------------------------------------------------------------------ #
    def list_tools(self, server: Optional[str] = None) -> list[dict[str, str]]:
        """Sync ``list_tools``; returns ``[]`` on any failure."""
        if not self.available():
            logger.warning("MCP unavailable or no servers configured; list_tools -> []")
            return []
        try:
            return _run_sync(self.list_tools_async(server), self.timeout)
        except (Exception, BaseExceptionGroup) as exc:
            logger.warning("MCP list_tools failed (server=%s): %s", server, exc)
            return []

    def call_tool(
        self,
        name: str,
        args: Optional[dict[str, Any]] = None,
        server: Optional[str] = None,
        default: Any = None,
    ) -> Any:
        """Sync ``call_tool``; returns ``default`` on any failure."""
        if not self.available():
            logger.warning("MCP unavailable or no servers configured; call_tool(%s) -> default", name)
            return default
        try:
            return _run_sync(self.call_tool_async(name, args, server), self.timeout)
        except (Exception, BaseExceptionGroup) as exc:
            logger.warning("MCP call_tool %r failed (server=%s): %s", name, server, exc)
            return default


# ---------------------------------------------------------------------- #
# Glossary conveniences used by the research pipeline
# ---------------------------------------------------------------------- #
def glossary_enabled() -> bool:
    """Whether the live glossary consult is switched on (LORE_MCP_GLOSSARY=1)."""
    return os.environ.get(GLOSSARY_FLAG_ENV, "").strip().lower() in _TRUTHY


def find_glossary_server() -> Optional[Path]:
    """Locate ``servers/glossary_server.py`` (env override, repo layout, cwd)."""
    override = os.environ.get(GLOSSARY_SERVER_ENV)
    candidates = [Path(override)] if override else []
    candidates += [
        Path(__file__).resolve().parents[2] / "servers" / "glossary_server.py",
        Path.cwd() / "servers" / "glossary_server.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def default_glossary_manager(timeout: float = 8.0) -> MCPManager:
    """An MCPManager wired to the bundled glossary server (empty if not found)."""
    servers: list[MCPServerSpec] = []
    path = find_glossary_server()
    if path is not None:
        servers.append(
            MCPServerSpec(name="glossary", command=sys.executable, args=[str(path)])
        )
    else:
        logger.warning("glossary_server.py not found; glossary MCP disabled")
    return MCPManager(servers=servers, timeout=timeout)


def lookup_glossary_terms(text: str, manager: Any = None) -> list[dict]:
    """Resolve org glossary terms in ``text`` via MCP. Returns ``[]`` on failure.

    ``manager`` may be any object with a ``call_tool(name, args)`` method
    (an ``MCPManager`` or a test double); defaults to the bundled server.
    """
    mgr = manager if manager is not None else default_glossary_manager()
    try:
        entries = mgr.call_tool("lookup_terms", {"text": text})
    except Exception as exc:  # injected managers may raise; ours does not
        logger.warning("glossary lookup failed: %s", exc)
        return []
    if isinstance(entries, dict):
        entries = [entries]
    if not isinstance(entries, list):
        return []
    return [e for e in entries if isinstance(e, dict)]
