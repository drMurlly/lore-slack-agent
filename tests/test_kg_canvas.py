"""M16: the knowledge graph is visible in the Canvas report + demo output.

These assert the judge-facing proof of deep research — a Decision timeline rendered
straight from the graph's supersedes chain, oldest→newest, with the current value.
"""
import json
import subprocess
import sys
from pathlib import Path

from conduit.citations import Answer, Citation
from conduit.canvas import build_report, build_report_markdown
from conduit.knowledge_graph import build_graph


def _reversal_graph():
    """A graph over a genuine pricing reversal ($10 → $20)."""
    from types import SimpleNamespace as S
    evidence = [
        S(text="We set the pricing tier to $10 for launch.",
          ts="1000.000000", channel="pricing", permalink="https://x/p1", author="a"),
        S(text="After review we changed the pricing tier to $20.",
          ts="2000.000000", channel="decisions", permalink="https://x/p2", author="b"),
    ]
    return build_graph(evidence, question="What did we decide about pricing?")


def _answer():
    return Answer(
        text="Pricing was $10 [1], later changed to $20 [2]. The current value is $20.",
        citations=[
            Citation(index=1, permalink="https://x/p1", channel="pricing", quote="set to $10"),
            Citation(index=2, permalink="https://x/p2", channel="decisions", quote="changed to $20"),
        ],
        graph_summary=_reversal_graph().summary(),
    )


def test_build_report_no_graph_is_unchanged_shape():
    """The 2-arg call still works (no crash) and omits the timeline section."""
    result = build_report(_answer(), "What about pricing?")
    assert "document_content" in result
    blocks = result["document_content"]["blocks"]
    assert blocks[0]["type"] == "header"
    dumped = json.dumps(blocks)
    assert "Decision timeline" not in dumped  # no graph passed -> no timeline section


def test_build_report_with_graph_renders_timeline():
    g = _reversal_graph()
    result = build_report(_answer(), "What did we decide about pricing?", graph=g)
    dumped = json.dumps(result["document_content"]["blocks"])
    assert "Decision timeline" in dumped
    assert "$10" in dumped and "$20" in dumped
    assert "https://x/p2" in dumped        # newest permalink present
    assert "Current: $20" in dumped        # resolved current value


def test_build_report_markdown_with_graph_renders_timeline():
    g = _reversal_graph()
    md = build_report_markdown(_answer(), "What did we decide about pricing?", graph=g)
    assert "## 🕸️ Decision timeline" in md
    assert "$10" in md and "$20" in md
    assert "](https://x/p2)" in md          # markdown link to newest source
    assert "**Current: $20**" in md


def test_demo_output_contains_graph_and_timeline():
    """run_demo.py writes graph summary + resolved timeline into demo_output.json."""
    repo = Path(__file__).resolve().parent.parent
    proc = subprocess.run(
        [sys.executable, "scripts/run_demo.py"],
        cwd=repo, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads((repo / "demo_output.json").read_text())
    assert "graph" in data and data["graph"]["reversals"] >= 1
    assert data.get("current_value") == "$20"
    assert isinstance(data.get("timeline"), list) and len(data["timeline"]) >= 2


def _ev(text, ts, channel="pricing"):
    from types import SimpleNamespace
    return SimpleNamespace(text=text, ts=ts, channel=channel, permalink=f"https://x/{ts}")


def test_stray_count_does_not_poison_timeline_or_drift():
    """A bare number in a pricing message must not join the price decision chain (class-anchored)."""
    evs = [
        _ev("We set pricing to $10 per seat.", "1000.0"),
        _ev("After review we changed pricing to $20 for the 3 enterprise seats.", "2000.0"),
    ]
    g = build_graph(evs, question="What did we decide about pricing?")
    rows = g.decision_rows("What did we decide about pricing?")
    assert [r["value"] for r in rows] == ["$10", "$20"]  # NOT $10, $20, 3
    d = g.drift_for_question("What did we decide about pricing?")
    assert d and d.old_value == "$10" and d.current_value == "$20"


def test_graph_drift_fallback_never_crosses_value_class():
    """detect_drift returns None here; the graph fallback must NOT invent a $10 -> 3 change."""
    from conduit.contradiction import detect_drift
    evs = [
        _ev("Pricing decided at $10 per seat.", "1000.0"),
        _ev("3 customers asked about pricing today.", "2000.0"),
    ]
    assert detect_drift(evs, question="What is our pricing?") is None
    g = build_graph(evs, question="What is our pricing?")
    assert g.drift_for_question("What is our pricing?") is None


def test_detect_drift_current_is_value_after_last_to():
    """When the newest message says 'from $10 to $20', current is $20 (LAST same-class token,
    not the first) — the old first-token logic would have missed this drift."""
    from conduit.contradiction import detect_drift
    evs = [
        _ev("We set pricing at $10.", "1000.0"),
        _ev("We changed pricing from $10 to $20.", "2000.0"),
    ]
    d = detect_drift(evs, question="pricing?")
    assert d and d.old_value == "$10" and d.current_value == "$20"


def test_duplicate_value_does_not_make_fake_timeline():
    """Two messages asserting the SAME value are not a 'change' — one row, no drift section."""
    evs = [
        _ev("Pricing is $10.", "1000.0"),
        _ev("Confirming pricing stays $10.", "2000.0"),
    ]
    g = build_graph(evs, question="pricing?")
    rows = g.decision_rows("pricing?")
    assert len({r["value"] for r in rows}) == 1
    assert g.drift_for_question("pricing?") is None
