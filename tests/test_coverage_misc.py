"""Coverage-hardening tests for canvas.py, config.py, citations.py and the
assistant-surface streaming/formatting edge cases."""
import types

import pytest
import yaml

from conduit.citations import Answer, Citation, _validate_citation_markers
from conduit.contradiction import TimelineDrift


# --------------------------------------------------------------------------- #
# canvas.py
# --------------------------------------------------------------------------- #
def _drift(older_link="https://x/p1", newer_link="https://x/p2"):
    older = types.SimpleNamespace(permalink=older_link, channel="pricing")
    newer = types.SimpleNamespace(permalink=newer_link, channel="decisions")
    return TimelineDrift(old_value="$10", new_value="$20", current_value="$20",
                         older=older, newer=newer, summary="pricing $10 -> $20")


def test_create_canvas_sends_markdown():
    from conduit.canvas import create_canvas
    client = types.SimpleNamespace(canvases_create=lambda **kw: kw)
    answer = Answer(text="Pricing is $20 [1].",
                    citations=[Citation(index=1, permalink="https://x/p1",
                                        channel="pricing", quote="set to $20")])
    resp = create_canvas(client, "Lore report", answer, "pricing?")
    assert resp["title"] == "Lore report"
    assert resp["document_content"]["type"] == "markdown"
    md = resp["document_content"]["markdown"]
    assert "Pricing is $20 [1](https://x/p1)." in md  # the [n] marker becomes the deep-link
    assert "## Sources" in md


def test_build_report_markdown_renders_drift_with_links():
    from conduit.canvas import build_report_markdown
    answer = Answer(text="Current price is $20 [1].",
                    citations=[Citation(index=1, permalink="https://x/p2",
                                        channel="decisions", quote="$20")],
                    drift=_drift())
    md = build_report_markdown(answer, "what is the price?")
    assert "## ⚠️ Conflicting signals over time" in md
    assert "Earlier: **$10** ([#pricing](https://x/p1))" in md
    assert "Later / current: **$20** ([#decisions](https://x/p2))" in md
    assert "Lore resolves to the most recent decision: **$20**" in md


def test_build_report_markdown_drift_without_permalinks():
    from conduit.canvas import build_report_markdown
    answer = Answer(text="Price is $20 [1].",
                    citations=[Citation(index=1, permalink="", channel="decisions", quote="$20")],
                    drift=_drift(older_link="", newer_link=""))
    md = build_report_markdown(answer, "price?")
    assert "Earlier: **$10** (#pricing)" in md         # no link form
    assert "Later / current: **$20** (#decisions)" in md


def test_timeline_rows_handles_none_and_raising_graph():
    from conduit.canvas import _timeline_rows
    assert _timeline_rows(None, "q") == []

    class _BadGraph:
        def decision_rows(self, question):
            raise RuntimeError("graph exploded")
    assert _timeline_rows(_BadGraph(), "q") == []


def test_build_report_markdown_renders_timeline_rows():
    from conduit.canvas import build_report_markdown

    class _Graph:
        def decision_rows(self, question):
            return [
                {"value": "$10", "channel": "pricing", "permalink": "https://x/p1"},
                {"value": "$20", "channel": "decisions", "permalink": ""},
            ]
    answer = Answer(text="Price now $20 [1].",
                    citations=[Citation(index=1, permalink="https://x/p2",
                                        channel="decisions", quote="$20")])
    md = build_report_markdown(answer, "price?", graph=_Graph())
    assert "## 🕸️ Decision timeline" in md
    assert "- **$10** — [#pricing](https://x/p1)" in md
    assert "- **$20** — #decisions" in md               # empty permalink -> plain channel
    assert "- **Current: $20**" in md


# --------------------------------------------------------------------------- #
# config.py — validation edges
# --------------------------------------------------------------------------- #
def _write(tmp_path, text):
    p = tmp_path / "conduit.yaml"
    p.write_text(text)
    return str(p)


def test_load_config_missing_file():
    from conduit.config import load_config
    with pytest.raises(FileNotFoundError):
        load_config("/no/such/config.yaml")


def test_load_config_empty_file_uses_defaults(tmp_path):
    from conduit.config import load_config
    cfg = load_config(_write(tmp_path, ""))  # yaml.safe_load -> None -> {}
    assert cfg.model == "llama3.2"
    assert cfg.servers == []


def test_load_config_rejects_non_mapping_root(tmp_path):
    from conduit.config import load_config
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, "- a\n- b\n"))


def test_load_config_rejects_bad_model(tmp_path):
    from conduit.config import load_config
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, "model: ''\n"))


def test_load_config_rejects_non_string_ollama_base(tmp_path):
    from conduit.config import load_config
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, "ollama_api_base: 123\n"))


def test_load_config_rejects_bad_servers():
    import os
    import tempfile

    from conduit.config import load_config

    def _load(text):
        fd, path = tempfile.mkstemp(suffix=".yaml")
        os.close(fd)
        with open(path, "w") as f:
            f.write(text)
        try:
            return load_config(path)
        finally:
            os.unlink(path)

    with pytest.raises(ValueError):
        _load("servers: not-a-list\n")
    with pytest.raises(ValueError):        # server entry not a dict
        _load("servers:\n  - just-a-string\n")
    with pytest.raises(ValueError):        # missing name
        _load("servers:\n  - command: run\n")
    with pytest.raises(ValueError):        # missing command
        _load("servers:\n  - name: s\n")
    with pytest.raises(ValueError):        # empty name
        _load("servers:\n  - name: ''\n    command: run\n")
    with pytest.raises(ValueError):        # empty command
        _load("servers:\n  - name: s\n    command: ''\n")
    with pytest.raises(ValueError):        # args not a list
        _load("servers:\n  - name: s\n    command: run\n    args: nope\n")
    with pytest.raises(ValueError):        # env not a dict
        _load("servers:\n  - name: s\n    command: run\n    env: nope\n")


def test_load_config_accepts_valid_servers(tmp_path):
    from conduit.config import load_config
    cfg = load_config(_write(tmp_path,
        "model: gemma3\n"
        "ollama_api_base: http://host/v1\n"
        "servers:\n"
        "  - name: glossary\n"
        "    command: python\n"
        "    args: [server.py]\n"
        "    env: {FOO: bar}\n"))
    assert cfg.model == "gemma3"
    assert cfg.ollama_api_base == "http://host/v1"
    assert cfg.servers[0].name == "glossary"
    assert cfg.servers[0].env == {"FOO": "bar"}


def test_load_config_rejects_malformed_yaml(tmp_path):
    from conduit.config import load_config
    with pytest.raises(yaml.YAMLError):
        load_config(_write(tmp_path, "model: [unclosed\n"))


# --------------------------------------------------------------------------- #
# citations.py — dangling-marker validation
# --------------------------------------------------------------------------- #
def test_validate_citation_markers_removes_dangling():
    cites = [Citation(index=1, permalink="p", channel="c", quote="q")]
    # [1] resolves, [9] does not -> the [9] marker is stripped.
    out = _validate_citation_markers("First [1] and second [9].", cites)
    assert "[1]" in out and "[9]" not in out


# --------------------------------------------------------------------------- #
# assistant_surface.py — streaming / formatting edge cases
# --------------------------------------------------------------------------- #
from conduit.assistant_surface import (  # noqa: E402
    AssistantContext, ResearchAssistant, _extract_ts,
)


def test_extract_ts_variants():
    assert _extract_ts(None) is None
    assert _extract_ts({"ts": "1.0"}) == "1.0"
    assert _extract_ts({}) is None

    class _Boom:
        def __getitem__(self, k):
            raise TypeError("no subscript")
    assert _extract_ts(_Boom()) is None


def test_set_status_skipped_off_assistant_container():
    client = types.SimpleNamespace(assistant_threads_setStatus=lambda **k: 1 / 0)
    a = ResearchAssistant(client, AssistantContext("C1", "1.0"), assistant_container=False)
    a.set_status("thinking")  # skipped entirely -> the /0 is never hit


def test_set_status_swallows_api_error():
    class _Client:
        def assistant_threads_setStatus(self, **k):
            raise RuntimeError("setStatus 400")
    a = ResearchAssistant(_Client(), AssistantContext("C1", "1.0"), assistant_container=True)
    a.set_status("thinking")  # must not raise


def test_stream_step_disables_when_no_ts():
    """If the first post yields no ts, streaming disables (never floods with new cards)."""
    from unittest.mock import MagicMock
    client = MagicMock()
    client.chat_postMessage.return_value = {}  # no ts
    a = ResearchAssistant(client, AssistantContext("C1", "1.0"), stream=True)
    a.emit_trace("decompose", "x")
    a.emit_trace("search", "y")
    a.emit_trace("synth", "z")
    client.chat_postMessage.assert_called_once()   # posted once, then disabled
    client.chat_update.assert_not_called()


def test_stream_step_swallows_post_exception():
    from unittest.mock import MagicMock
    client = MagicMock()
    client.chat_postMessage.side_effect = RuntimeError("post blew up")
    a = ResearchAssistant(client, AssistantContext("C1", "1.0"), stream=True)
    a.emit_trace("decompose", "x")  # exception caught; _posted already set so no re-post
    a.emit_trace("search", "y")
    assert a.trace_log == ["decompose: x", "search: y"]


def test_post_result_without_canvas_truncates_long_answer():
    from unittest.mock import MagicMock
    client = MagicMock()
    a = ResearchAssistant(client, AssistantContext("C1", ""))  # no thread_ts -> omitted
    long_text = "x" * 4000
    answer = Answer(text=long_text, citations=[])
    a.post_result(answer, "")  # no canvas url -> plain final-answer block, truncated
    blocks = client.chat_postMessage.call_args.kwargs["blocks"]
    dumped = str(blocks)
    assert "📄 *Final Answer*" in dumped
    assert "…" in dumped                    # truncated with an ellipsis
    assert "thread_ts" not in client.chat_postMessage.call_args.kwargs


def test_post_result_caps_citations_with_more_note():
    from unittest.mock import MagicMock
    client = MagicMock()
    a = ResearchAssistant(client, AssistantContext("C1", "1.0"))
    cites = [Citation(index=i, permalink=f"https://x/p{i}", channel=f"c{i}",
                      quote=f"quote {i}") for i in range(1, 8)]  # 7 citations
    answer = Answer(text="ans " + " ".join(f"[{i}]" for i in range(1, 8)), citations=cites)
    a.post_result(answer, "https://canvas/url")
    blocks = client.chat_postMessage.call_args.kwargs["blocks"]
    context_notes = [b for b in blocks if b.get("type") == "context"]
    assert any("and 2 more" in str(b) for b in context_notes)  # 7 - 5 capped = 2 more
