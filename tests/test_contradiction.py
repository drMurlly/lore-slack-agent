"""Tests for the evidence-grounded contradiction / timeline-drift resolver."""
from types import SimpleNamespace

from conduit.contradiction import (
    timeline_sort,
    extract_values,
    detect_drift,
    resolve_answer_text,
    conflict_canvas_section,
    TimelineDrift,
)


def _ev(text, ts, channel="pricing", permalink=None):
    return SimpleNamespace(
        text=text, ts=ts, channel=channel,
        permalink=permalink or f"https://slack.com/{channel}/p{ts}",
    )


def test_timeline_sort_oldest_first_and_newest_first():
    a, b, c = _ev("x", "300"), _ev("y", "100"), _ev("z", "200")
    assert [e.ts for e in timeline_sort([a, b, c])] == ["100", "200", "300"]
    assert [e.ts for e in timeline_sort([a, b, c], newest_first=True)] == ["300", "200", "100"]


def test_extract_values_handles_currency_percent_number():
    assert extract_values("we decided on $10 for the tier") == ["$10"]
    assert extract_values("changed to $ 20 after review") == ["$20"]   # whitespace normalised
    assert extract_values("the rate limit is 20% now") == ["20%"]
    assert "1,200" in extract_values("budget of 1,200 dollars")


def test_detect_drift_positive_dollar_reversal():
    """The exact money-shot the old (\\w+) regex could NOT catch: $10 -> $20."""
    ev = [
        _ev("We decided on $10 for the pricing tier.", "1.0", channel="pricing"),
        _ev("Update: we changed pricing to $20 after review.", "200.0", channel="leadership"),
    ]
    drift = detect_drift(ev, question="What did we decide about pricing, and did it change?")
    assert isinstance(drift, TimelineDrift)
    assert drift.old_value == "$10"
    assert drift.new_value == "$20"
    assert drift.current_value == "$20"
    assert drift.newer.channel == "leadership"          # latest = current source
    assert drift.current_permalink == drift.newer.permalink


def test_detect_drift_negative_single_value():
    ev = [
        _ev("We decided on $10 for pricing.", "1.0"),
        _ev("Reminder: pricing is $10, unchanged.", "200.0"),
    ]
    assert detect_drift(ev, question="what is the pricing") is None


def test_detect_drift_ignores_unrelated_evidence():
    ev = [
        _ev("Lunch is at $10 today.", "1.0", channel="random"),
        _ev("The new sandwich place costs $25.", "200.0", channel="random"),
    ]
    # keywords are about pricing policy, not lunch -> no relevant drift
    assert detect_drift(ev, question="what did we decide about the API pricing policy") is None


def test_resolve_answer_text_is_deterministic_and_idempotent():
    drift = TimelineDrift("$10", "$20", "$20", _ev("a", "1"), _ev("b", "2"), "sum")
    bare = resolve_answer_text("Pricing was discussed.", drift)
    assert "$10" in bare and "$20" in bare and "current value is $20" in bare.lower()
    # already-complete answer is left essentially unchanged (idempotent)
    good = "We decided $10 then changed to $20. The current value is $20."
    assert resolve_answer_text(good, drift) == good
    # no drift -> untouched
    assert resolve_answer_text("no change here", None) == "no change here"


def test_conflict_canvas_section_shape():
    drift = TimelineDrift(
        "$10", "$20", "$20",
        _ev("a", "1", channel="pricing", permalink="https://slack.com/p1"),
        _ev("b", "2", channel="leadership", permalink="https://slack.com/p2"),
        "sum",
    )
    sec = conflict_canvas_section(drift)
    assert sec["type"] == "section"
    txt = sec["text"]["text"]
    assert "https://slack.com/p1" in txt and "https://slack.com/p2" in txt
    assert "$10" in txt and "$20" in txt
    assert conflict_canvas_section(None) is None


def test_wording_split_pricing_vs_price_still_detects_reversal():
    """Robustness (light-stem): the question says 'pricing' but the evidence says 'price'.
    Previously the plural-only normaliser missed this and could HIDE the reversal; the light
    stem unifies price/pricing so the $10 -> $20 change is still resolved to the current value."""
    from conduit.knowledge_graph import build_graph
    q = "What did we decide about pricing?"
    evidence = [
        _ev("We set the price at $10 for launch.", "1000.000100", channel="pricing"),
        _ev("We later changed the price to $20.", "2000.000200", channel="decisions"),
    ]
    drift = detect_drift(evidence, question=q)
    assert drift is not None
    assert drift.old_value == "$10"
    assert drift.new_value == "$20"
    assert drift.current_value == "$20"

    # The knowledge graph anchors both values on ONE topic and yields a 2-step timeline.
    graph = build_graph(evidence, question=q)
    rows = graph.decision_rows(q)
    assert [r["value"] for r in rows] == ["$10", "$20"]


def test_light_stem_does_not_overmerge_distinct_words():
    """The light stem must keep genuinely different words apart (no crude 4-char collisions)."""
    from conduit.contradiction import _light_stem
    assert _light_stem("pricing") == _light_stem("price") == _light_stem("priced")
    assert _light_stem("hiring") == _light_stem("hire")
    assert _light_stem("required") != _light_stem("requests")
    assert _light_stem("company") != _light_stem("competitors")
    assert _light_stem("policy") != _light_stem("police")
