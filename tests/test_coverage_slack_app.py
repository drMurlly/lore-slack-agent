"""Coverage-hardening tests for slack_app.py — the backend construction helpers,
the Canvas builder, the Bolt handlers' defensive branches, app wiring, and main().

Everything is exercised with fakes/mocks; no Slack, model, or socket connection.
"""
import sys
from unittest.mock import MagicMock, patch

import pytest

import conduit.slack_app as slack_app
from conduit.citations import Answer, Citation


@pytest.fixture(autouse=True)
def _reset_module_state(monkeypatch):
    """Reset the module-level caches/sets between tests so ordering can't leak state."""
    slack_app._TEAM.clear()
    slack_app._RTS_CACHE.clear()
    slack_app._HOME_PUBLISHED.clear()
    slack_app._DEDUP._seen.clear()
    for var in ("LORE_CHANNELS", "SLACK_BOT_TOKEN", "SLACK_USER_TOKEN", "LORE_USE_RTS_API",
                "OLLAMA_API_BASE", "LORE_USE_OLLAMA", "LORE_INDEX_TTL", "LORE_STREAM_TRACE",
                "SLACK_APP_TOKEN", "LORE_MODEL"):
        monkeypatch.delenv(var, raising=False)
    yield


# --------------------------------------------------------------------------- #
# _team_info
# --------------------------------------------------------------------------- #
def test_team_info_caches_auth_test():
    client = MagicMock()
    client.auth_test.return_value = {"url": "https://t.slack.com/", "team_id": "T1", "user_id": "B1"}
    info = slack_app._team_info(client)
    assert info == {"team_url": "https://t.slack.com", "team_id": "T1", "bot_user_id": "B1"}
    # Cached: a second call returns the cache without another auth.test.
    client.auth_test.reset_mock()
    assert slack_app._team_info(client) is slack_app._TEAM
    client.auth_test.assert_not_called()


def test_team_info_none_client_and_exception():
    assert slack_app._team_info(None) == {}
    client = MagicMock()
    client.auth_test.side_effect = RuntimeError("auth failed")
    assert slack_app._team_info(client) == {}  # exception swallowed -> empty


# --------------------------------------------------------------------------- #
# _discover_channels
# --------------------------------------------------------------------------- #
def test_discover_channels_env_override(monkeypatch):
    monkeypatch.setenv("LORE_CHANNELS", "C1:general, C2:pricing , ,C3")
    out = slack_app._discover_channels(MagicMock())
    assert out == {"C1": "general", "C2": "pricing", "C3": "C3"}


def test_discover_channels_via_conversations_list():
    client = MagicMock()
    client.conversations_list.side_effect = [
        {"channels": [{"id": "C1", "name": "general", "is_member": True},
                      {"id": "C2", "name": "lurk", "is_member": False}],
         "response_metadata": {"next_cursor": "CUR"}},
        {"channels": [{"id": "C3", "name": "pricing", "is_member": True}],
         "response_metadata": {"next_cursor": ""}},
    ]
    out = slack_app._discover_channels(client)
    assert out == {"C1": "general", "C3": "pricing"}  # non-members excluded, pagination followed


def test_discover_channels_none_client_and_error():
    assert slack_app._discover_channels(None) == {}
    client = MagicMock()
    client.conversations_list.side_effect = RuntimeError("api down")
    assert slack_app._discover_channels(client) == {}


# --------------------------------------------------------------------------- #
# _build_rts backend selection
# --------------------------------------------------------------------------- #
def test_build_rts_official_api_falls_back_on_construct_error(monkeypatch):
    monkeypatch.setenv("SLACK_USER_TOKEN", "xoxp-user")
    monkeypatch.setenv("LORE_USE_RTS_API", "1")
    import conduit.rts_client as rts_client

    def _boom(*a, **k):
        raise RuntimeError("cannot build")
    monkeypatch.setattr(rts_client, "RTSClient", _boom)
    from conduit.fake_rts import FakeRTS
    assert isinstance(slack_app._build_rts(client=None), FakeRTS)  # falls back


def test_build_rts_history_backend_and_cache(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-real")
    monkeypatch.setenv("LORE_CHANNELS", "C1:general")
    monkeypatch.setenv("LORE_INDEX_TTL", "1000")
    client = MagicMock()
    client.auth_test.return_value = {"url": "https://t.slack.com", "team_id": "T1", "user_id": "B1"}

    built = {"count": 0}

    class _FakeHistoryRTS:
        def __init__(self, slack, channels=None, team_url=""):
            built["count"] += 1
            self._channel_names = channels

        def refresh(self):
            return self

    import conduit.live_rts as live_rts
    monkeypatch.setattr(live_rts, "SlackHistoryRTS", _FakeHistoryRTS)

    first = slack_app._build_rts(client)
    second = slack_app._build_rts(client)  # within TTL -> cached, not rebuilt
    assert isinstance(first, _FakeHistoryRTS)
    assert first is second
    assert built["count"] == 1


def test_build_rts_no_channels_falls_back_to_fake(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-real")
    client = MagicMock()
    client.conversations_list.return_value = {"channels": [], "response_metadata": {}}
    from conduit.fake_rts import FakeRTS
    assert isinstance(slack_app._build_rts(client), FakeRTS)


def test_build_rts_history_construct_error_falls_back(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-real")
    monkeypatch.setenv("LORE_CHANNELS", "C1:general")
    client = MagicMock()
    client.auth_test.return_value = {"url": "https://t.slack.com", "team_id": "T1"}
    import conduit.live_rts as live_rts

    def _boom(*a, **k):
        raise RuntimeError("index blew up")
    monkeypatch.setattr(live_rts, "SlackHistoryRTS", _boom)
    from conduit.fake_rts import FakeRTS
    assert isinstance(slack_app._build_rts(client), FakeRTS)


def test_index_channel_names_reads_channel_map():
    rts = MagicMock()
    rts._channel_names = {"C1": "general", "C2": "pricing"}
    assert set(slack_app._index_channel_names(rts)) == {"general", "pricing"}
    # No attribute -> empty list, never raises.
    assert slack_app._index_channel_names(object()) == []


# --------------------------------------------------------------------------- #
# _build_llm
# --------------------------------------------------------------------------- #
def test_build_llm_uses_ollama_when_configured(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_BASE", "http://host/v1")
    monkeypatch.setenv("LORE_MODEL", "gemma3:27b")
    sentinel = object()
    import conduit.agent as agent
    captured = {}

    def _fake_ollama(model="llama3.2", timeout=180):
        captured["model"] = model
        captured["timeout"] = timeout
        return sentinel
    monkeypatch.setattr(agent, "OllamaLLMClient", _fake_ollama)
    assert slack_app._build_llm() is sentinel
    assert captured["model"] == "gemma3:27b"


def test_build_llm_falls_back_when_ollama_errors(monkeypatch):
    monkeypatch.setenv("LORE_USE_OLLAMA", "1")
    import conduit.agent as agent
    monkeypatch.setattr(agent, "OllamaLLMClient",
                        lambda **k: (_ for _ in ()).throw(RuntimeError("no model")))
    from conduit.agent import FakeLLMClient
    assert isinstance(slack_app._build_llm(), FakeLLMClient)


def test_build_llm_live_mode_warns_and_uses_fake(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-real")  # live mode, but no Ollama config
    from conduit.agent import FakeLLMClient
    assert isinstance(slack_app._build_llm(), FakeLLMClient)


# --------------------------------------------------------------------------- #
# _create_canvas
# --------------------------------------------------------------------------- #
def _answer():
    return Answer(text="Pricing is $20 [1].",
                  citations=[Citation(index=1, permalink="https://x/p1", channel="pricing",
                                      quote="We set pricing to $20")])


def test_create_canvas_channel_grant_returns_url():
    client = MagicMock()
    client.canvases_create.return_value = {"canvas_id": "CAN1"}
    client.auth_test.return_value = {"url": "https://t.slack.com", "team_id": "T1"}
    url = slack_app._create_canvas(client, _answer(), "pricing?", "C123")
    assert url == "https://t.slack.com/docs/T1/CAN1"
    client.canvases_access_set.assert_called_once()
    assert client.canvases_access_set.call_args.kwargs["channel_ids"] == ["C123"]


def test_create_canvas_user_grant_on_dm():
    client = MagicMock()
    client.canvases_create.return_value = {"canvas": {"id": "CAN2"}}  # nested id shape
    client.auth_test.return_value = {"url": "https://t.slack.com", "team_id": "T1"}
    url = slack_app._create_canvas(client, _answer(), "q", "D123", user_id="U9")
    assert url.endswith("/CAN2")
    assert client.canvases_access_set.call_args.kwargs["user_ids"] == ["U9"]


def test_create_canvas_user_grant_failure_omits_button():
    client = MagicMock()
    client.canvases_create.return_value = {"canvas_id": "CAN3"}
    client.auth_test.return_value = {"url": "https://t.slack.com", "team_id": "T1"}
    client.canvases_access_set.side_effect = RuntimeError("no access")
    # DM surface, the only grant attempt fails -> no reader -> empty URL (no dead button).
    assert slack_app._create_canvas(client, _answer(), "q", "D1", user_id="U9") == ""


def test_create_canvas_no_team_url_returns_empty():
    client = MagicMock()
    client.canvases_create.return_value = {"canvas_id": "CAN4"}
    client.auth_test.return_value = {"url": "", "team_id": ""}
    assert slack_app._create_canvas(client, _answer(), "q", "C1") == ""


def test_create_canvas_no_id_and_create_error():
    client = MagicMock()
    client.canvases_create.return_value = {}
    client.auth_test.return_value = {"url": "https://t.slack.com", "team_id": "T1"}
    assert slack_app._create_canvas(client, _answer(), "q", "C1") == ""
    client.canvases_create.side_effect = RuntimeError("canvas api down")
    assert slack_app._create_canvas(client, _answer(), "q", "C1") == ""


def test_post_kwargs_includes_thread_ts_only_when_present():
    assert slack_app._post_kwargs("C1", None) == {"channel": "C1"}
    assert slack_app._post_kwargs("C1", "123.45") == {"channel": "C1", "thread_ts": "123.45"}


# --------------------------------------------------------------------------- #
# research_and_respond — empty question + error card fallbacks
# --------------------------------------------------------------------------- #
def test_research_and_respond_empty_question_posts_prompt():
    client = MagicMock()
    out = slack_app.research_and_respond(client, "C1", None, "   ")
    assert out == ""
    assert "Ask me a question" in client.chat_postMessage.call_args.kwargs["text"]


def test_research_and_respond_empty_question_post_failure_returns_none():
    client = MagicMock()
    client.chat_postMessage.side_effect = RuntimeError("post failed")
    assert slack_app.research_and_respond(client, "C1", None, "") is None


def test_research_and_respond_error_card(monkeypatch):
    client = MagicMock()
    monkeypatch.setattr(slack_app, "_build_rts",
                        lambda c: (_ for _ in ()).throw(RuntimeError("boom")))
    out = slack_app.research_and_respond(client, "C1", None, "a real question")
    assert out == ""  # an error card was delivered
    assert client.chat_postMessage.call_args.kwargs["text"] == "Research hit an error."


def test_research_and_respond_error_card_post_failure_returns_none(monkeypatch):
    client = MagicMock()
    client.chat_postMessage.side_effect = RuntimeError("cannot post")
    monkeypatch.setattr(slack_app, "_build_rts",
                        lambda c: (_ for _ in ()).throw(RuntimeError("boom")))
    assert slack_app.research_and_respond(client, "C1", None, "q") is None


def test_research_and_respond_empty_state(monkeypatch):
    """No evidence -> a Block Kit empty-state card, still counted as delivered ('')."""
    client = MagicMock()

    class _EmptyRTS:
        _channel_names = {"C1": "general"}

        def search(self, q, limit=10):
            return []
    monkeypatch.setattr(slack_app, "_build_rts", lambda c: _EmptyRTS())
    monkeypatch.setenv("LORE_STREAM_TRACE", "0")
    out = slack_app.research_and_respond(client, "C1", None, "obscure question")
    assert out == ""
    assert client.chat_postMessage.call_args.kwargs["text"] == "No relevant history found."


# --------------------------------------------------------------------------- #
# Bolt handlers — defensive branches
# --------------------------------------------------------------------------- #
def test_handle_thread_message_dedup():
    client = MagicMock()
    with patch.object(slack_app, "research_and_respond", return_value="") as rr:
        for _ in range(2):
            slack_app.handle_thread_message(
                body={"event_id": "dm-dup"},
                event={"text": "hi", "channel_type": "im", "user": "U1"},
                say=MagicMock(), client=client)
    assert rr.call_count == 1


def test_handle_lore_interim_message_failure_is_swallowed():
    client = MagicMock()
    respond = MagicMock(side_effect=RuntimeError("respond down"))
    with patch.object(slack_app, "research_and_respond", return_value=""):
        slack_app.handle_lore(body={"text": "q", "trigger_id": "t1"},
                              ack=MagicMock(), say=MagicMock(), client=client, respond=respond)


def test_handle_lore_ephemeral_fallback_when_nothing_delivered():
    client = MagicMock()
    respond = MagicMock()
    with patch.object(slack_app, "research_and_respond", return_value=None), \
         patch.object(slack_app, "handle_query", return_value="fallback answer"):
        slack_app.handle_lore(body={"text": "q", "trigger_id": "t2"},
                              ack=MagicMock(), say=MagicMock(), client=client, respond=respond)
    # interim + ephemeral fallback both routed through respond
    assert any("fallback answer" in str(c.args) for c in respond.call_args_list)


def test_handle_lore_fallback_failure_is_swallowed():
    client = MagicMock()
    respond = MagicMock(side_effect=[None, RuntimeError("gone")])  # interim ok, fallback raises
    with patch.object(slack_app, "research_and_respond", return_value=None), \
         patch.object(slack_app, "handle_query", return_value="x"):
        slack_app.handle_lore(body={"text": "q", "trigger_id": "t3"},
                              ack=MagicMock(), say=MagicMock(), client=client, respond=respond)


def test_publish_home_idempotent_and_swallows_failure():
    client = MagicMock()
    slack_app._publish_home(client, "U1")
    slack_app._publish_home(client, "U1")  # already published -> no second call
    assert client.views_publish.call_count == 1
    # Missing user id / client -> no-op.
    slack_app._publish_home(client, "")
    slack_app._publish_home(None, "U2")
    # A views_publish failure is swallowed.
    client.views_publish.side_effect = RuntimeError("nope")
    slack_app._publish_home(client, "U3")


def test_handle_app_home_opened_ignores_non_home_tab():
    client = MagicMock()
    slack_app.handle_app_home_opened(event={"tab": "messages", "user": "U1"}, client=client)
    client.views_publish.assert_not_called()


def test_handle_home_ask_happy_path(monkeypatch):
    client = MagicMock()
    client.conversations_open.return_value = {"channel": {"id": "D1"}}
    started = {}

    class _FakeThread:
        def __init__(self, target=None, name=None, daemon=None):
            started["target"] = target

        def start(self):
            started["started"] = True
    monkeypatch.setattr(slack_app.threading if hasattr(slack_app, "threading") else __import__("threading"),
                        "Thread", _FakeThread, raising=False)
    with patch("threading.Thread", _FakeThread):
        slack_app.handle_home_ask(
            ack=MagicMock(),
            body={"user": {"id": "U1"}, "actions": [{"value": "What about pricing?"}]},
            client=client)
    assert started.get("started")
    assert "Researching" in client.chat_postMessage.call_args.kwargs["text"]


def test_handle_home_ask_missing_user_or_question():
    client = MagicMock()
    slack_app.handle_home_ask(ack=MagicMock(), body={"user": {}, "actions": []}, client=client)
    client.conversations_open.assert_not_called()


def test_handle_home_ask_no_dm_channel():
    client = MagicMock()
    client.conversations_open.return_value = {"channel": {}}  # no id
    slack_app.handle_home_ask(
        ack=MagicMock(),
        body={"user": {"id": "U1"}, "actions": [{"value": "q"}]}, client=client)
    client.chat_postMessage.assert_not_called()


def test_handle_home_ask_interim_post_failure(monkeypatch):
    client = MagicMock()
    client.conversations_open.return_value = {"channel": {"id": "D1"}}
    client.chat_postMessage.side_effect = RuntimeError("post down")

    class _FakeThread:
        def __init__(self, **k):
            pass

        def start(self):
            pass
    with patch("threading.Thread", _FakeThread):
        # interim post fails but the handler still starts research without raising
        slack_app.handle_home_ask(
            ack=MagicMock(),
            body={"user": {"id": "U1"}, "actions": [{"value": "q"}]}, client=client)


def test_handle_home_ask_outer_failure_is_swallowed():
    # body.get on a non-dict raises inside the try -> caught by the outer handler.
    slack_app.handle_home_ask(ack=MagicMock(), body=None, client=MagicMock())


def test_handle_view_canvas_action_acks():
    ack = MagicMock()
    slack_app.handle_view_canvas_action(ack=ack)
    ack.assert_called_once()


def test_assistant_thread_started_swallows_failure():
    say = MagicMock(side_effect=RuntimeError("say down"))
    # exception is caught; no raise
    slack_app.assistant_thread_started(payload={"channel_id": "C1"},
                                       set_suggested_prompts=MagicMock(), say=say)


def test_assistant_user_message_routes_to_research(monkeypatch):
    client = MagicMock()
    monkeypatch.setattr(slack_app, "notify_usage" if hasattr(slack_app, "notify_usage") else "_x",
                        lambda *a, **k: None, raising=False)
    with patch.object(slack_app, "research_and_respond", return_value="") as rr, \
         patch("conduit.notify.notify_usage", lambda *a, **k: None):
        slack_app.assistant_user_message(
            payload={"channel": "C1", "ts": "1.0", "user": "U1", "text": "hi"},
            client=client, context={"channel_id": "C1"})
    rr.assert_called_once()
    assert rr.call_args.kwargs["is_assistant"] is True


# --------------------------------------------------------------------------- #
# build_app + middleware + main()
# --------------------------------------------------------------------------- #
def test_build_app_returns_none_without_slack_bolt(monkeypatch):
    monkeypatch.setitem(sys.modules, "slack_bolt", None)
    assert slack_app.build_app() is None


def test_build_app_wires_handlers_and_middleware():
    app = slack_app.build_app()
    assert app is not None
    # Find and drive the incoming-logging middleware directly.
    mws = [m for m in getattr(app, "_middleware_list", [])
           if getattr(getattr(m, "func", None), "__name__", "") == "_log_incoming"]
    assert mws, "expected the _log_incoming middleware to be registered"
    log_mw = mws[0].func
    called = {}

    def _next():
        called["next"] = True
        return "ok"
    assert log_mw(body={"command": "/lore", "user_id": "U1", "channel_id": "C1"},
                  next=_next) == "ok"
    assert called["next"] is True
    # A body that raises inside the try is swallowed; next() still runs.
    assert log_mw(body=None, next=lambda: "recovered") == "recovered"


def test_build_app_survives_missing_assistant(monkeypatch):
    import slack_bolt

    def _boom(*a, **k):
        raise RuntimeError("no Assistant in this bolt")
    monkeypatch.setattr(slack_bolt, "Assistant", _boom, raising=False)
    app = slack_app.build_app()  # exception caught; mention/command paths still wired
    assert app is not None


def test_main_requires_app_token(monkeypatch):
    monkeypatch.delenv("SLACK_APP_TOKEN", raising=False)
    monkeypatch.setattr(slack_app, "build_app", lambda: MagicMock())
    assert slack_app.main() == 1


def test_main_returns_1_without_app(monkeypatch):
    monkeypatch.setattr(slack_app, "build_app", lambda: None)
    assert slack_app.main() == 1


def test_main_starts_socket_mode(monkeypatch):
    monkeypatch.setenv("SLACK_APP_TOKEN", "xapp-token")
    fake_app = MagicMock()
    fake_app.client = MagicMock()
    fake_app.client.conversations_list.return_value = {"channels": [], "response_metadata": {}}
    monkeypatch.setattr(slack_app, "build_app", lambda: fake_app)
    monkeypatch.setattr(slack_app, "_build_llm", lambda: MagicMock())

    started = {}

    class _FakeHandler:
        def __init__(self, app, token):
            started["token"] = token

        def start(self):
            started["started"] = True

    import slack_bolt.adapter.socket_mode as sm
    monkeypatch.setattr(sm, "SocketModeHandler", _FakeHandler)

    # Run the background warmup/home-population threads synchronously & catch their bodies.
    import threading as _t
    real = _t.Thread

    class _InlineThread(real):
        def start(self):
            self.run()
    monkeypatch.setattr(_t, "Thread", _InlineThread)

    assert slack_app.main() == 0
    assert started["started"] is True
    assert started["token"] == "xapp-token"
