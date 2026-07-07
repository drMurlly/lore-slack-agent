from unittest.mock import MagicMock, patch

import conduit.slack_app as slack_app


def test_duplicate_event_skipped():
    # Isolate from other tests by clearing module-level dedup state
    slack_app._DEDUP._seen.clear()

    body = {"event_id": "evt-dup-001", "event": {"text": "What is Lore?", "user": "U123"}}
    event = body["event"]
    say = MagicMock()
    client = MagicMock()
    mock_logger = MagicMock()

    with patch.object(slack_app, "research_and_respond", return_value="") as mock_research:
        slack_app.handle_mention(body=body, event=event, say=say, client=client, logger=mock_logger)
        slack_app.handle_mention(body=body, event=event, say=say, client=client, logger=mock_logger)

    assert mock_research.call_count == 1


def test_lore_command_repeatable():
    """P0-3: /lore must answer every time — dedup keys on trigger_id (unique per call), so
    two invocations that carry NO event_id (as real slash payloads don't) both reply."""
    slack_app._DEDUP._seen.clear()
    ack, say, client = MagicMock(), MagicMock(), MagicMock()

    with patch.object(slack_app, "research_and_respond", return_value="") as mock_research:
        slack_app.handle_lore(body={"text": "q1", "trigger_id": "trig-1"},
                              ack=ack, say=say, client=client)
        slack_app.handle_lore(body={"text": "q2", "trigger_id": "trig-2"},
                              ack=ack, say=say, client=client)

    assert mock_research.call_count == 2
    assert ack.call_count == 2


def test_lore_command_true_duplicate_skipped():
    """Same trigger_id (an actual Slack retry) is deduplicated."""
    slack_app._DEDUP._seen.clear()
    ack, say, client = MagicMock(), MagicMock(), MagicMock()

    with patch.object(slack_app, "research_and_respond", return_value="") as mock_research:
        slack_app.handle_lore(body={"text": "q", "trigger_id": "trig-same"},
                              ack=ack, say=say, client=client)
        slack_app.handle_lore(body={"text": "q", "trigger_id": "trig-same"},
                              ack=ack, say=say, client=client)

    assert mock_research.call_count == 1


def test_message_listener_ignores_bot_and_channel_chatter():
    """P0-4: the generic message listener must not answer bot echoes, edits/joins, or
    ordinary channel messages — only direct messages (channel_type == 'im')."""
    slack_app._DEDUP._seen.clear()
    say, client = MagicMock(), MagicMock()

    with patch.object(slack_app, "research_and_respond", return_value="") as mock_research:
        # bot message -> ignored
        slack_app.handle_thread_message(
            body={"event_id": "e1"},
            event={"text": "hi", "bot_id": "B999", "channel_type": "im"},
            say=say, client=client)
        # subtype (edit/join) -> ignored
        slack_app.handle_thread_message(
            body={"event_id": "e2"},
            event={"text": "hi", "subtype": "message_changed", "channel_type": "im"},
            say=say, client=client)
        # ordinary public-channel message -> ignored
        slack_app.handle_thread_message(
            body={"event_id": "e3"},
            event={"text": "hi", "channel_type": "channel"},
            say=say, client=client)

    assert mock_research.call_count == 0


def test_message_listener_answers_direct_message():
    slack_app._DEDUP._seen.clear()
    say, client = MagicMock(), MagicMock()

    with patch.object(slack_app, "research_and_respond", return_value="") as mock_research:
        slack_app.handle_thread_message(
            body={"event_id": "dm-1"},
            event={"text": "what did we decide?", "channel_type": "im", "user": "U1"},
            say=say, client=client)

    # A DM is answered via the full money-shot path (posts its own Block Kit + Canvas), not say().
    assert mock_research.call_count == 1


def test_channel_surface_posts_full_money_shot():
    """The win: a non-assistant surface (/lore, @mention, DM) now posts the FULL money-shot —
    Decision-Graph badge, decision timeline, conflicting-signals, and a View-Canvas button —
    not a plain wall of text. Previously only the Assistant split-view got this."""
    import os
    from conduit.rts_client import SearchHit
    from conduit.agent import LLMClient

    class FakeRTS:
        def search(self, query, limit=10):
            return [
                SearchHit(text="We set the pricing tier to $10 for launch.", channel="pricing",
                          ts="1000.000100", permalink="https://x/archives/C1/p1000000100",
                          score=0.9, author="maya"),
                SearchHit(text="After review we changed the pricing tier to $20.", channel="decisions",
                          ts="2000.000200", permalink="https://x/archives/C2/p2000000200",
                          score=0.8, author="priya"),
            ]

    class FakeLLM(LLMClient):
        def chat(self, messages, tools=None):
            if "Decompose" in messages[0]["content"]:
                return {"content": "pricing decision"}
            return {"content": "We set pricing at $10 [1], then changed it to $20 [2]. "
                               "The current price is $20 [2]."}

    client = MagicMock()
    slack_app._DEDUP._seen.clear()
    with patch.dict(os.environ, {"LORE_STREAM_TRACE": "0"}), \
         patch.object(slack_app, "_build_rts", return_value=FakeRTS()), \
         patch.object(slack_app, "_build_llm", return_value=FakeLLM()), \
         patch.object(slack_app, "_create_canvas", return_value="https://slack.com/docs/T1/CANVAS1"):
        canvas_url = slack_app.research_and_respond(client, "C999", None,
                                                    "What did we decide about pricing?")

    assert canvas_url == "https://slack.com/docs/T1/CANVAS1"
    posts = [c for c in client.chat_postMessage.call_args_list if c.kwargs.get("blocks")]
    assert posts, "expected a Block Kit post"
    dumped = str([c.kwargs["blocks"] for c in posts])
    assert "Decision Graph" in dumped            # graph badge (proof of deep research)
    assert "Decision timeline" in dumped         # the timeline
    assert "$10" in dumped and "$20" in dumped   # both values, current bolded
    assert "Conflicting signals" in dumped       # the reversal, resolved
    assert "View Full Canvas" in dumped          # the Canvas deep-link button
    # A slash command has no thread — thread_ts must be omitted, not sent empty.
    for c in posts:
        assert "thread_ts" not in c.kwargs


def test_app_home_opened_publishes_lore_view():
    client = MagicMock()
    slack_app.handle_app_home_opened(event={"tab": "home", "user": "U42"}, client=client)
    client.views_publish.assert_called_once()
    view = client.views_publish.call_args.kwargs["view"]
    assert view["type"] == "home"
    # It's the Lore home (not the old Conduit MCP list).
    dumped = str(view)
    assert "Lore" in dumped and "Conduit Agent - MCP Servers" not in dumped
