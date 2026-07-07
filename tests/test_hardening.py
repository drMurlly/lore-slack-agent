"""Regression tests for the robustness/hardening pass — each pins a fixed weakness."""
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# --- #2 EventDedup thread-safety ------------------------------------------------------------ #
def test_dedup_is_thread_safe_single_winner():
    from conduit.dedup import EventDedup
    dd = EventDedup()
    results, errors = [], []

    def worker():
        try:
            results.append(dd.is_seen("evt-shared"))
        except Exception as e:  # OrderedDict-mutated-during-iteration etc.
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(24)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"is_seen raised under concurrency: {errors}"
    assert results.count(False) == 1  # exactly one caller sees it as new
    assert results.count(True) == 23


# --- #1 detect_drift is topic-anchored (no cross-topic false reversal) ---------------------- #
def _ev(text, ts, channel="c"):
    return SimpleNamespace(text=text, ts=ts, channel=channel, permalink=f"http://x/{ts}")


def test_detect_drift_rejects_cross_topic_values():
    from conduit.contradiction import detect_drift
    ev = [
        _ev("pricing is $10", "1", channel="pricing"),
        _ev("marketing budget is $99", "2", channel="marketing"),
    ]
    drift = detect_drift(ev, question="what did we decide on pricing and the marketing budget?")
    assert drift is None  # $10 and $99 are unrelated topics — no fabricated reversal


def test_detect_drift_still_fires_single_topic():
    from conduit.contradiction import detect_drift
    ev = [_ev("pricing is $10", "1"), _ev("pricing changed to $20", "2")]
    drift = detect_drift(ev, question="pricing")
    assert drift is not None and drift.old_value == "$10" and drift.current_value == "$20"


# --- #4 resolve_answer_text reconciles a wrong 'current' claim ------------------------------- #
def test_resolve_answer_text_strips_contradictory_current():
    from conduit.contradiction import resolve_answer_text, TimelineDrift
    d = TimelineDrift(old_value="$10", new_value="$20", current_value="$20",
                      older=SimpleNamespace(channel="p", permalink="x"),
                      newer=SimpleNamespace(channel="d", permalink="y"), summary="s")
    out = resolve_answer_text("The current price is $15 [2].", d)
    assert "$15" not in out            # the wrong 'current' claim is removed, not contradicted
    assert "current value is $20" in out.lower()   # deterministic current value stands alone


# --- #5 evidence dedup keeps distinct messages with empty permalinks ------------------------- #
def test_gather_evidence_keeps_distinct_empty_permalink_messages():
    from conduit.research import run
    from conduit.agent import LLMClient
    from conduit.rts_client import SearchHit

    class RTS:
        def search(self, query, limit=10):
            return [
                SearchHit(text="pricing is $10", channel="pricing", ts="1.0", permalink="", score=1.0, author=None),
                SearchHit(text="pricing is $20", channel="decisions", ts="2.0", permalink="", score=0.9, author=None),
            ]

    class LLM(LLMClient):
        def chat(self, messages, tools=None):
            return {"content": "pricing"}

    res = run("what is our pricing?", RTS(), LLM(), glossary=False)
    texts = {e.text for e in res.evidence}
    assert "pricing is $10" in texts and "pricing is $20" in texts  # neither collapsed away


# --- #6 rank drops off-topic prose but keeps value-bearing + glossary evidence --------------- #
def test_rank_by_question_drops_offtopic_keeps_value_and_glossary():
    from conduit.research import _rank_by_question
    ev = [
        SimpleNamespace(text="pricing tier is $10", ts="3"),        # on-topic
        SimpleNamespace(text="admins must enable MFA in Okta", ts="9"),  # off-topic prose, no value
        SimpleNamespace(text="changed it to $20", ts="4"),          # value-bearing, terse
        SimpleNamespace(text="Annual Recurring Revenue crossed $2M", ts="5"),  # glossary expansion
    ]
    kept = _rank_by_question("what is our pricing?", ev, top_k=8,
                             expansions=["Annual Recurring Revenue"])
    kept_texts = {e.text for e in kept}
    assert "admins must enable MFA in Okta" not in kept_texts        # off-topic prose dropped
    assert "pricing tier is $10" in kept_texts
    assert "changed it to $20" in kept_texts                         # value-bearing protected
    assert "Annual Recurring Revenue crossed $2M" in kept_texts      # glossary recall preserved


# --- #3 money-shot timeline section is capped + clipped under Slack's 3000-char limit -------- #
def test_money_shot_timeline_is_clipped_and_keeps_current():
    from conduit.blocks import build_money_shot_blocks

    rows = [{"value": f"${i}", "channel": "pricing", "ts": str(i),
             "permalink": f"http://slack/archives/C1/p{i}"} for i in range(300)]
    graph = SimpleNamespace(decision_rows=lambda q: rows)
    answer = SimpleNamespace(graph_summary=None, drift=None)

    blocks = build_money_shot_blocks(answer, graph=graph, question="pricing")
    for b in blocks:
        if b.get("type") == "section":
            assert len(b["text"]["text"]) <= 3000
    dumped = str(blocks)
    assert "Current: $299" in dumped   # the current-value line always survives the cap


# --- #7 untrusted quote sanitizers neutralize link/markup injection -------------------------- #
def test_textsafe_neutralizes_injection():
    from conduit.textsafe import mrkdwn_safe, markdown_safe, oneline
    # Slack mrkdwn: the <url|label> link and the code span are defused (escaped / removed).
    m = mrkdwn_safe("Payroll <https://phish|Portal> `code`")
    assert "<https://phish" not in m and "&lt;" in m and "`" not in m
    # Canvas markdown: link/image/code escaped so they render as literal text, not active markup.
    md = markdown_safe("[x](http://phish) ![](http://track) `c`")
    assert "](" not in md            # link/image close-open sequence broken
    assert "[x](http" not in md      # the raw link pattern is gone
    assert "\\`" in md               # backtick escaped → literal, not a code span
    assert "\n" not in oneline("a\nb\nc")


# --- #8 streaming trace never floods: at most one post when ts is uncapturable --------------- #
def test_stream_trace_does_not_flood_when_ts_missing():
    from conduit.assistant_surface import ResearchAssistant, AssistantContext
    client = MagicMock()
    client.chat_postMessage.return_value = {}  # no 'ts' → _extract_ts None
    a = ResearchAssistant(client, AssistantContext(channel="C1", thread_ts="1.0"), stream=True)
    for i in range(8):
        a.emit_trace("search", f"step {i}")
    assert client.chat_postMessage.call_count == 1   # posted once, then gave up (no flood)
    client.chat_update.assert_not_called()


# --- #9 non-Latin messages are indexable/searchable ----------------------------------------- #
def test_unicode_message_is_searchable():
    from conduit.live_rts import SlackHistoryRTS, _content_tokens
    rts = SlackHistoryRTS(MagicMock(), channels={"C1": "pricing"}, team_url="http://x")
    rts._index = [{
        "text": "Цена изменилась на 20", "channel": "pricing", "channel_id": "C1",
        "ts": "100.0", "permalink": "http://x/p100", "author": None,
        "tokens": _content_tokens("Цена изменилась на 20"),
    }]
    hits = rts.search("цена", limit=5)
    assert hits and "Цена" in hits[0].text   # Cyrillic query finds the Cyrillic message


# --- #11 research_and_respond never raises on empty question even if the post fails ---------- #
def test_empty_question_never_raises():
    import conduit.slack_app as slack_app
    client = MagicMock()
    client.chat_postMessage.side_effect = RuntimeError("transient 5xx")
    # Must not raise despite the failing post.
    assert slack_app.research_and_respond(client, "C1", None, "   ") is None


# --- #12 load_config gives ValueError (not AttributeError) on a non-dict root ---------------- #
def test_load_config_non_dict_raises_valueerror(tmp_path):
    from conduit.config import load_config
    p = tmp_path / "bad.yaml"
    p.write_text("- foo\n- bar\n")
    with pytest.raises(ValueError):
        load_config(str(p))
