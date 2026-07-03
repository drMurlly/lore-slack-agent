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

    with patch.object(slack_app, "handle_query", return_value="mocked answer") as mock_research:
        slack_app.handle_mention(body=body, event=event, say=say, client=client, logger=mock_logger)
        slack_app.handle_mention(body=body, event=event, say=say, client=client, logger=mock_logger)

    assert mock_research.call_count == 1


def test_lore_command_repeatable():
    """P0-3: /lore must answer every time — dedup keys on trigger_id (unique per call), so
    two invocations that carry NO event_id (as real slash payloads don't) both reply."""
    slack_app._DEDUP._seen.clear()
    ack, say, client = MagicMock(), MagicMock(), MagicMock()

    with patch.object(slack_app, "handle_query", return_value="answer") as mock_research:
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

    with patch.object(slack_app, "handle_query", return_value="answer") as mock_research:
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

    with patch.object(slack_app, "handle_query", return_value="answer") as mock_research:
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

    with patch.object(slack_app, "handle_query", return_value="answer") as mock_research:
        slack_app.handle_thread_message(
            body={"event_id": "dm-1"},
            event={"text": "what did we decide?", "channel_type": "im", "user": "U1"},
            say=say, client=client)

    assert mock_research.call_count == 1
    say.assert_called_once()


def test_app_home_opened_publishes_lore_view():
    client = MagicMock()
    slack_app.handle_app_home_opened(event={"tab": "home", "user": "U42"}, client=client)
    client.views_publish.assert_called_once()
    view = client.views_publish.call_args.kwargs["view"]
    assert view["type"] == "home"
    # It's the Lore home (not the old Conduit MCP list).
    dumped = str(view)
    assert "Lore" in dumped and "Conduit Agent - MCP Servers" not in dumped
