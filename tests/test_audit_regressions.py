"""Regression tests for the 2026-07-07 adversarial-audit fixes.

Each test pins one verified finding so a green-CI regression can't silently reintroduce a
wrong/missing money-shot, a recall drop, an injection, or a broken deliverable in front of a
live judge. Grouped by finding number (see the audit remediation plan).
"""
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from conduit.contradiction import detect_drift, extract_typed_values


def _ev(text, ts, channel="pricing", permalink=None):
    return SimpleNamespace(text=text, ts=ts, channel=channel,
                           permalink=permalink or f"https://slack.com/{channel}/p{ts}")


# --- #1 currency regex must not swallow the next word's first letter ------------------------- #
def test_currency_suffix_does_not_eat_following_word():
    assert extract_typed_values("changed to $49 monthly going forward") == [("money", "$49")]
    assert extract_typed_values("valued at $50 billion now") == [("money", "$50")]
    # a currency followed by a b/m/k word is NOT turned into $50b / $50 million
    assert extract_typed_values("close the deal at $50 by summer") == [("money", "$50")]
    # an ADJACENT magnitude suffix is still recognised
    assert extract_typed_values("raised a $2M seed") == [("money", "$2M")]
    assert extract_typed_values("moved to $49m ARR") == [("money", "$49m")]


def test_currency_suffix_bug_no_longer_corrupts_current_value():
    ev = [_ev("we set the Pro pricing tier to $29 per seat", "1"),
          _ev("we changed the Pro pricing tier to $49 monthly going forward", "2", channel="leadership")]
    d = detect_drift(ev, question="What did we decide about pricing, and did it change?")
    assert d is not None and d.current_value == "$49"   # not "$49m"


# --- #5 same magnitude written two ways is not a reversal ------------------------------------ #
def test_restated_cents_is_not_a_reversal():
    assert extract_typed_values("confirmed the pricing at $49.00") == [("money", "$49")]
    ev = [_ev("pricing tier set to $49 per seat", "1"),
          _ev("confirmed the pricing model at $49.00 going forward", "2")]
    assert detect_drift(ev, question="pricing") is None


# --- #4 "current is X, up from Y" resolves to X (not the trailing historical Y) -------------- #
def test_up_from_does_not_invert_current():
    ev = [_ev("we set pricing at $20", "1"),
          _ev("current pricing is $20, up from the $50 we floated", "2")]
    assert detect_drift(ev, question="pricing") is None   # stable $20, no fabricated reversal


def test_from_to_still_resolves_to_the_to_value():
    ev = [_ev("we set the pricing tier to $29 per seat", "1"),
          _ev("changed the pricing tier from $29 to $49 per seat", "2", channel="leadership")]
    d = detect_drift(ev, question="pricing")
    assert d is not None and d.old_value == "$29" and d.current_value == "$49"


def test_revert_from_back_to_resolves_to_original():
    ev = [_ev("we set the pricing tier to $20", "1"),
          _ev("reverted the pricing tier from $20 back to $10", "3")]
    d = detect_drift(ev, question="pricing")
    assert d is not None and d.current_value == "$10"


# --- #2 bare counts / percentages are not fabricated into a reversal ------------------------- #
def test_planned_vs_actual_count_is_not_a_reversal():
    ev = [_ev("Hiring plan: we want 3 engineers in Q1", "1", channel="hiring"),
          _ev("Hiring update: onboarded 2 engineers", "2", channel="hiring")]
    assert detect_drift(ev, question="what did we decide about hiring") is None


def test_two_uptime_percentages_without_change_word_are_not_a_reversal():
    ev = [_ev("target uptime is 99%", "1"), _ev("measured uptime came in at 97%", "2")]
    assert detect_drift(ev, question="uptime") is None


def test_percentage_reversal_with_change_word_still_fires():
    ev = [_ev("SLA uptime target is 99%", "1"),
          _ev("we dropped the SLA uptime target to 97%", "2")]
    d = detect_drift(ev, question="uptime SLA")
    assert d is not None and d.old_value == "99%" and d.current_value == "97%"


# --- #15 3-value chain: current is the NEWEST value (not first-differing) -------------------- #
def test_three_value_chain_resolves_to_newest():
    from conduit.knowledge_graph import build_graph
    q = "pricing"
    ev = [_ev("pricing set to $10", "1000"),
          _ev("pricing raised to $15", "2000"),
          _ev("pricing changed to $20", "3000")]
    d = detect_drift(ev, question=q)
    assert d.old_value == "$10" and d.current_value == "$20"
    g = build_graph(ev, question=q)
    assert [r["value"] for r in g.decision_rows(q)] == ["$10", "$15", "$20"]
    assert g.drift_for_question(q).current_value == "$20"


# --- #17 revert-to-original chain is coherent (no fabricated current, badge counts supersede) - #
def test_revert_chain_is_coherent():
    from conduit.knowledge_graph import build_graph
    q = "pricing"
    ev = [_ev("pricing set to $10", "1000"),
          _ev("pricing changed to $20", "2000"),
          _ev("pricing reverted back to $10", "3000")]
    g = build_graph(ev, question=q)
    assert [r["value"] for r in g.decision_rows(q)] == ["$10", "$20", "$10"]
    assert detect_drift(ev, question=q) is None      # net no-change -> no fabricated reversal
    assert g.drift_for_question(q) is None
    assert g.summary()["reversals"] >= 1             # the supersedes chain is still recorded


# --- #6 sentence-final keywords/values keep recall ------------------------------------------ #
def test_tokenizer_strips_trailing_period():
    from conduit.live_rts import _tokens
    assert _tokens("we agreed to raise the price.")[-1] == "price"
    assert "$10.50" in _tokens("it costs $10.50 today")   # interior dot preserved


def test_search_finds_sentence_final_keyword():
    from conduit.live_rts import SlackHistoryRTS, _content_tokens
    rts = SlackHistoryRTS(MagicMock(), channels={"C1": "pricing"}, team_url="http://x")
    rts._index = [{
        "text": "We agreed to raise the price.", "channel": "pricing", "channel_id": "C1",
        "ts": "100.0", "permalink": "http://x/p100", "author": None,
        "tokens": _content_tokens("We agreed to raise the price."),
    }]
    hits = rts.search("what is our current price", limit=5)
    assert hits and "price" in hits[0].text


# --- #11 the feedback-loop filter no longer excludes ordinary human messages ---------------- #
def test_is_lore_output_keeps_human_messages():
    from conduit.live_rts import _is_lore_output
    assert _is_lore_output("We moved to tier [2] pricing, now $49") is False
    assert _is_lore_output("Ticket [123] tracks the pricing change to $49") is False
    assert _is_lore_output("the current value is unclear, maybe $49") is False
    # but Lore's OWN distinctive posts are still filtered out
    assert _is_lore_output("📄 *Final Answer*\nPricing is $20 [1]") is True
    assert _is_lore_output("An earlier value was $10.") is True


# --- #9 Canvas markdown neutralizes angle-bracket autolinks --------------------------------- #
def test_markdown_safe_escapes_autolinks():
    from conduit.textsafe import markdown_safe
    out = markdown_safe("Payroll <https://phish.example> and <admin@phish.example>")
    assert "<https://phish" not in out and "&lt;" in out


# --- #8 the answer body can't @-broadcast or plant live links under Lore's identity --------- #
def test_neutralize_answer_body_strips_slack_control_but_keeps_citations():
    from conduit.textsafe import neutralize_answer_body
    out = neutralize_answer_body("Price is $20 [1]. <!channel> see <https://evil.example|here>")
    assert "<!channel>" not in out
    assert "https://evil.example" not in out    # url dropped; label kept
    assert "here" in out
    assert "[1]" in out                          # citation marker preserved for deep-linking


def test_synthesized_answer_is_neutralized():
    from conduit.research import ResearchResult, Evidence
    from conduit.rts_client import SearchHit
    from conduit.agent import FakeLLMClient
    from conduit.citations import synthesize
    hit = SearchHit(text="pricing $20", channel="pricing", ts="2.0", permalink="http://x/p2",
                    score=1.0, author=None)
    ev = Evidence(text="pricing $20", channel="pricing", ts="2.0", permalink="http://x/p2",
                  score=1.0, author=None, citation_index=1, source_hit=hit)
    result = ResearchResult(question="pricing", evidence=[ev])
    llm = FakeLLMClient({"content": "The price is $20 [1]. <!channel> ping <http://evil|x>"})
    ans = synthesize(result, llm)
    assert "<!channel>" not in ans.text and "http://evil" not in ans.text


# --- #16 live Canvas-creation contract (the seam a judge actually sees) ---------------------- #
def _answer_stub():
    return SimpleNamespace(text="ans", citations=[], drift=None, graph_summary=None)


def test_create_canvas_channel_share_and_url(monkeypatch):
    import conduit.slack_app as slack_app
    monkeypatch.setattr(slack_app, "_team_info",
                        lambda c: {"team_url": "https://x.slack.com", "team_id": "T1"})
    client = MagicMock()
    client.canvases_create.return_value = {"canvas_id": "F123"}
    url = slack_app._create_canvas(client, _answer_stub(), "what about pricing?", "C9", user_id="U1")
    assert url == "https://x.slack.com/docs/T1/F123"
    ck = client.canvases_create.call_args.kwargs
    assert ck["title"].startswith("Lore — ")
    assert ck["document_content"]["type"] == "markdown"
    assert isinstance(ck["document_content"]["markdown"], str)
    client.canvases_access_set.assert_any_call(canvas_id="F123", access_level="read",
                                               channel_ids=["C9"])


def test_create_canvas_alternate_response_shape(monkeypatch):
    import conduit.slack_app as slack_app
    monkeypatch.setattr(slack_app, "_team_info",
                        lambda c: {"team_url": "https://x.slack.com", "team_id": "T1"})
    client = MagicMock()
    client.canvases_create.return_value = {"canvas": {"id": "F9"}}
    assert slack_app._create_canvas(client, _answer_stub(), "q", "C1").endswith("/docs/T1/F9")


def test_create_canvas_missing_id_returns_empty(monkeypatch):
    import conduit.slack_app as slack_app
    monkeypatch.setattr(slack_app, "_team_info",
                        lambda c: {"team_url": "https://x.slack.com", "team_id": "T1"})
    client = MagicMock()
    client.canvases_create.return_value = {}
    assert slack_app._create_canvas(client, _answer_stub(), "q", "C1") == ""
    client.canvases_access_set.assert_not_called()


def test_create_canvas_no_team_returns_empty(monkeypatch):
    import conduit.slack_app as slack_app
    monkeypatch.setattr(slack_app, "_team_info", lambda c: {})
    client = MagicMock()
    client.canvases_create.return_value = {"canvas_id": "F1"}
    assert slack_app._create_canvas(client, _answer_stub(), "q", "C1") == ""


def test_create_canvas_dm_grants_user_not_channel(monkeypatch):
    import conduit.slack_app as slack_app
    monkeypatch.setattr(slack_app, "_team_info",
                        lambda c: {"team_url": "https://x.slack.com", "team_id": "T1"})
    client = MagicMock()
    client.canvases_create.return_value = {"canvas_id": "F5"}
    url = slack_app._create_canvas(client, _answer_stub(), "q", "D123", user_id="U9")
    assert url.endswith("/docs/T1/F5")
    client.canvases_access_set.assert_called_once_with(canvas_id="F5", access_level="read",
                                                       user_ids=["U9"])


# --- #10 the View-Canvas button is omitted when no read grant succeeded --------------------- #
def test_create_canvas_returns_empty_when_no_grant(monkeypatch):
    import conduit.slack_app as slack_app
    monkeypatch.setattr(slack_app, "_team_info",
                        lambda c: {"team_url": "https://x.slack.com", "team_id": "T1"})
    client = MagicMock()
    client.canvases_create.return_value = {"canvas_id": "F7"}
    client.canvases_access_set.side_effect = RuntimeError("not_in_channel")
    assert slack_app._create_canvas(client, _answer_stub(), "q", "C1", user_id="") == ""


# --- #18 off-corpus question posts the friendly empty-state and no Canvas ------------------- #
def test_off_corpus_question_posts_empty_state(monkeypatch):
    import conduit.slack_app as slack_app
    from conduit.fake_rts import FakeRTS
    from conduit.agent import FakeLLMClient
    client = MagicMock()
    monkeypatch.setattr(slack_app, "_build_rts", lambda c: FakeRTS(corpus={}))
    monkeypatch.setattr(slack_app, "_build_llm", lambda: FakeLLMClient({"content": "sub"}))
    with patch.dict(os.environ, {"LORE_STREAM_TRACE": "0", "LORE_MCP_GLOSSARY": "0"}):
        out = slack_app.research_and_respond(client, "C1", None, "totally off-corpus question")
    assert out == ""   # delivered (empty-state), not None
    posts = [c for c in client.chat_postMessage.call_args_list if c.kwargs.get("blocks")]
    assert posts and "couldn't find anything" in str(posts[0].kwargs["blocks"])
    client.canvases_create.assert_not_called()


def test_build_empty_state_blocks_shape():
    from conduit.blocks import build_empty_state_blocks
    b = build_empty_state_blocks("pricing?", None)
    assert len(b) == 1 and b[0]["type"] == "section"
    assert "couldn't find anything" in b[0]["text"]["text"]
    b2 = build_empty_state_blocks("pricing?", ["alpha", "beta"])
    assert "#alpha" in b2[0]["text"]["text"]


# --- #12 raw exception text never reaches the workspace ------------------------------------- #
def test_error_card_shows_class_name_not_raw_message(monkeypatch):
    import conduit.slack_app as slack_app

    def _boom(_c):
        raise RuntimeError("secret endpoint http://internal-host:9000/x")

    client = MagicMock()
    monkeypatch.setattr(slack_app, "_build_rts", _boom)
    monkeypatch.setattr(slack_app, "_build_llm", lambda: None)
    with patch.dict(os.environ, {"LORE_STREAM_TRACE": "0"}):
        out = slack_app.research_and_respond(client, "C1", None, "q")
    assert out == ""   # an error card was delivered
    dumped = str(client.chat_postMessage.call_args_list)
    assert "internal-host:9000" not in dumped and "secret endpoint" not in dumped
    assert "RuntimeError" in dumped


def test_handle_query_error_is_generic():
    from conduit.slack_app import handle_query
    from conduit.agent import LLMClient
    from conduit.fake_rts import FakeRTS, CorpusMessage

    class _BoomLLM(LLMClient):
        def chat(self, messages, tools=None):
            raise RuntimeError("leaky http://internal:1234 detail")

    out = handle_query("q", rts=FakeRTS({"c": [CorpusMessage("pricing $10", "c", "1.0")]}),
                       llm=_BoomLLM())
    assert "internal:1234" not in out and "leaky" not in out
    assert "error" in out.lower()


# --- #13 /lore falls back to an ephemeral answer only when nothing was delivered ------------ #
def test_lore_ephemeral_fallback_when_delivery_fails():
    import conduit.slack_app as slack_app
    slack_app._DEDUP._seen.clear()
    respond = MagicMock()
    with patch.object(slack_app, "research_and_respond", return_value=None), \
         patch.object(slack_app, "handle_query", return_value="fallback answer"):
        slack_app.handle_lore(body={"text": "q", "trigger_id": "t-fb", "channel_id": "C1"},
                              ack=MagicMock(), say=MagicMock(), client=MagicMock(), respond=respond)
    assert any(c.args and c.args[0] == "fallback answer" for c in respond.call_args_list)


def test_lore_no_double_post_when_delivered():
    import conduit.slack_app as slack_app
    slack_app._DEDUP._seen.clear()
    with patch.object(slack_app, "research_and_respond", return_value=""), \
         patch.object(slack_app, "handle_query") as hq:
        slack_app.handle_lore(body={"text": "q", "trigger_id": "t-ok", "channel_id": "C1"},
                              ack=MagicMock(), say=MagicMock(), client=MagicMock(), respond=MagicMock())
    hq.assert_not_called()   # delivered "" -> no fallback -> no double post


# --- #3 the flagship demo cites the correct source for each claimed value ------------------- #
def test_demo_citations_are_correctly_grounded():
    """Every '<value> [n]' claim in the demo answer must deep-link to a source that asserts it."""
    import importlib.util
    import pathlib
    import re
    root = pathlib.Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location("_rd", root / "scripts" / "run_demo.py")
    rd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rd)
    from conduit.research import run
    from conduit.citations import synthesize

    result = run(rd.QUESTION, rd.seed_corpus(), rd.DemoLLMClient())
    answer = synthesize(result, rd.DemoLLMClient())
    by_index = {c.index: c for c in answer.citations}
    claims = list(re.finditer(r"(\$\d[\d,]*)\s*\[(\d+)\]", answer.text))
    assert claims, "expected cited value claims in the demo answer"
    for m in claims:
        value, idx = m.group(1), int(m.group(2))
        cite = by_index.get(idx)
        assert cite is not None and value in cite.quote, \
            f"claim {value} [{idx}] must link to a source containing {value}"
