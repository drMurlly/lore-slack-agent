"""Coverage-hardening tests for the operator usage notifier (notify.py).

Drives ``_dispatch`` and the Slack/email helpers directly (bypassing the
background thread) so the throttle, the self-query skip, and both email
transports are exercised without any real network/SMTP/sendmail I/O.
"""
import subprocess
from unittest.mock import MagicMock

import pytest

import conduit.notify as notify


@pytest.fixture(autouse=True)
def _reset_throttle(monkeypatch):
    """Each test starts with a clean throttle window and no notifier env config."""
    notify._last = 0.0
    notify._suppressed = 0
    for var in ("LORE_NOTIFY_SLACK_USER", "LORE_NOTIFY_MIN_INTERVAL", "LORE_NOTIFY_TO",
                "LORE_SMTP_PASSWORD", "LORE_SMTP_USER", "LORE_SMTP_HOST", "LORE_SMTP_PORT",
                "LORE_MAIL_FROM", "SENDMAIL_PATH"):
        monkeypatch.delenv(var, raising=False)
    yield


def test_notify_usage_spawns_thread_and_returns(monkeypatch):
    seen = {}
    monkeypatch.setattr(notify, "_dispatch",
                        lambda *a, **k: seen.setdefault("args", a))
    # notify_usage runs _dispatch on a daemon thread; join it so the assertion is deterministic.
    real_thread = notify.threading.Thread
    threads = []

    def _capture(*args, **kwargs):
        t = real_thread(*args, **kwargs)
        threads.append(t)
        return t
    monkeypatch.setattr(notify.threading, "Thread", _capture)

    notify.notify_usage("/lore", user="U1", text="hi", channel="general", client=MagicMock())
    for t in threads:
        t.join(2)
    assert seen["args"][0] == "/lore"


def test_dispatch_skips_operator_own_query(monkeypatch):
    monkeypatch.setenv("LORE_NOTIFY_SLACK_USER", "UOP")
    client = MagicMock()
    notify._dispatch("/lore", user="UOP", text="hi", channel="general", client=client)
    client.conversations_open.assert_not_called()


def test_dispatch_notifies_slack_and_throttles(monkeypatch):
    monkeypatch.setenv("LORE_NOTIFY_SLACK_USER", "UOP")
    client = MagicMock()
    client.conversations_open.return_value = {"channel": {"id": "D1"}}

    # A different user (not the operator) -> a Slack DM is sent.
    notify._dispatch("@mention", user="U2", text="what changed?", channel="pricing", client=client)
    assert client.conversations_open.call_count == 1
    assert client.chat_postMessage.call_count == 1
    dm = client.chat_postMessage.call_args.kwargs
    assert dm["channel"] == "D1"
    assert "U2" in dm["text"] and "pricing" in dm["text"]

    # A second call inside the throttle window is suppressed (counter increments, no new DM).
    notify._dispatch("@mention", user="U3", text="again", channel="pricing", client=client)
    assert client.chat_postMessage.call_count == 1
    assert notify._suppressed == 1


def test_dispatch_reports_suppressed_tail(monkeypatch):
    monkeypatch.setenv("LORE_NOTIFY_SLACK_USER", "UOP")
    monkeypatch.setenv("LORE_NOTIFY_MIN_INTERVAL", "0")  # no throttling
    client = MagicMock()
    client.conversations_open.return_value = {"channel": {"id": "D1"}}
    notify._suppressed = 3  # pretend 3 were suppressed earlier
    notify._dispatch("/lore", user="U2", text="q", channel="", client=client)
    body = client.chat_postMessage.call_args.kwargs["text"]
    assert "+3 more" in body
    assert "a DM / the assistant" in body  # empty channel -> generic location
    assert notify._suppressed == 0


def test_dispatch_uses_no_text_placeholder(monkeypatch):
    monkeypatch.setenv("LORE_NOTIFY_SLACK_USER", "UOP")
    client = MagicMock()
    client.conversations_open.return_value = {"channel": {"id": "D1"}}
    notify._dispatch("/lore", user="", text="", channel="", client=client)
    body = client.chat_postMessage.call_args.kwargs["text"]
    assert "(no text)" in body and "someone" in body  # empty user -> "someone"


def test_dispatch_never_raises(monkeypatch):
    monkeypatch.setenv("LORE_NOTIFY_SLACK_USER", "UOP")
    client = MagicMock()
    client.conversations_open.side_effect = RuntimeError("slack down")
    # Must swallow the failure (Slack handlers must never crash).
    notify._dispatch("/lore", user="U2", text="q", channel="c", client=client)


def test_notify_slack_noop_without_uid_or_client(monkeypatch):
    monkeypatch.setenv("LORE_NOTIFY_SLACK_USER", "")
    client = MagicMock()
    notify._notify_slack(client, "subj", "body")  # uid empty -> early return
    client.conversations_open.assert_not_called()
    monkeypatch.setenv("LORE_NOTIFY_SLACK_USER", "UOP")
    notify._notify_slack(None, "subj", "body")    # client None -> early return


def test_notify_slack_swallows_api_failure(monkeypatch):
    monkeypatch.setenv("LORE_NOTIFY_SLACK_USER", "UOP")
    client = MagicMock()
    client.conversations_open.side_effect = RuntimeError("boom")
    notify._notify_slack(client, "subj", "body")  # logs a warning, does not raise


def test_notify_email_disabled_without_recipient(monkeypatch):
    monkeypatch.delenv("LORE_NOTIFY_TO", raising=False)
    # No recipient -> both transports are skipped (nothing to assert but the no-raise).
    notify._notify_email("subj", "body")


def test_notify_email_prefers_sendmail(monkeypatch):
    monkeypatch.setenv("LORE_NOTIFY_TO", "op@example.com")
    monkeypatch.delenv("LORE_SMTP_PASSWORD", raising=False)
    calls = {}
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: calls.update(args=a, input=k.get("input")))
    notify._notify_email("Subject", "Body text")
    assert calls, "expected sendmail subprocess.run to be invoked"
    assert b"op@example.com" in calls["input"]


def test_notify_email_sendmail_swallows_failure(monkeypatch):
    monkeypatch.setenv("LORE_NOTIFY_TO", "op@example.com")
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("no sendmail")))
    notify._notify_email_sendmail("s", "b", "op@example.com")  # must not raise


def test_notify_email_smtp_path(monkeypatch):
    monkeypatch.setenv("LORE_NOTIFY_TO", "op@example.com")
    monkeypatch.setenv("LORE_SMTP_PASSWORD", "app-password")
    monkeypatch.setenv("LORE_SMTP_USER", "sender@example.com")

    sent = {}

    class _FakeSMTP:
        def __init__(self, host, port, timeout=15):
            sent["host"] = host
            sent["port"] = port

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def ehlo(self):
            sent["ehlo"] = True

        def starttls(self):
            sent["tls"] = True

        def login(self, user, pw):
            sent["login"] = (user, pw)

        def sendmail(self, frm, to, msg):
            sent["sendmail"] = (frm, to)

    monkeypatch.setattr(notify.smtplib, "SMTP", _FakeSMTP)
    notify._notify_email("Subject", "Body")

    assert sent["host"] == "smtp.gmail.com" and sent["port"] == 587
    assert sent["login"] == ("sender@example.com", "app-password")
    assert sent["sendmail"][0] == "sender@example.com"
    assert sent["tls"] is True


def test_notify_email_smtp_swallows_failure(monkeypatch):
    monkeypatch.setenv("LORE_NOTIFY_TO", "op@example.com")

    def _boom(*a, **k):
        raise RuntimeError("smtp unreachable")

    monkeypatch.setattr(notify.smtplib, "SMTP", _boom)
    notify._notify_email_smtp("s", "b", "op@example.com", "pw")  # must not raise
