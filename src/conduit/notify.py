"""Operator usage notifications — Lore runs on local GPU, so tell the operator when it's used.

Two best-effort channels, both optional and configured via environment:
  * Slack DM  — set ``LORE_NOTIFY_SLACK_USER`` to the operator's Slack user id (works with the
    bot token already in use; no extra credential).
  * Email     — set ``LORE_SMTP_PASSWORD`` (+ ``LORE_NOTIFY_TO`` / ``LORE_SMTP_USER``); uses
    STARTTLS SMTP (e.g. smtp.gmail.com:587 with a Gmail App Password).

Notifications are dispatched on a background thread (never block a Slack handler), lightly
throttled, and never raise.
"""
from __future__ import annotations

import logging
import os
import smtplib
import threading
import time
from email.mime.text import MIMEText
from typing import Any

logger = logging.getLogger(__name__)

_last = 0.0
_suppressed = 0
_lock = threading.Lock()


def notify_usage(source: str, user: str = "", text: str = "", channel: str = "",
                 client: Any = None) -> None:
    """Fire-and-forget: tell the operator Lore was queried. Safe to call from any handler."""
    threading.Thread(target=_dispatch, args=(source, user, text, channel, client),
                     daemon=True).start()


def _dispatch(source: str, user: str, text: str, channel: str, client: Any) -> None:
    global _last, _suppressed
    try:
        # Don't notify the operator about their OWN queries — they already see the answer.
        if user and user == os.environ.get("LORE_NOTIFY_SLACK_USER", ""):
            return
        min_interval = float(os.environ.get("LORE_NOTIFY_MIN_INTERVAL", "6"))
        with _lock:
            now = time.monotonic()
            if now - _last < min_interval:
                _suppressed += 1
                return
            extra = _suppressed
            _suppressed = 0
            _last = now

        who = f"<@{user}>" if user else "someone"
        where = f"#{channel}" if channel else "a DM / the assistant"
        q = (text or "").strip().replace("\n", " ")[:300] or "(no text)"
        tail = f"\n(+{extra} more in the last few seconds)" if extra else ""
        subject = "🔔 Lore was just used (local GPU)"
        body = (f"{who} queried Lore via {source} in {where}.\n\n"
                f"Question: {q}{tail}\n\n"
                f"— Lore usage notifier")

        _notify_slack(client, subject, body)
        _notify_email(subject, body)
    except Exception:
        logger.debug("notify dispatch failed", exc_info=True)


def _notify_slack(client: Any, subject: str, body: str) -> None:
    uid = os.environ.get("LORE_NOTIFY_SLACK_USER", "").strip()
    if not client or not uid:
        return
    try:
        im = client.conversations_open(users=uid)
        cid = im["channel"]["id"]
        client.chat_postMessage(channel=cid, text=f"*{subject}*\n{body}")
        logger.info("usage Slack DM sent to %s", uid)
    except Exception:
        logger.warning("usage Slack DM failed", exc_info=True)


def _notify_email(subject: str, body: str) -> None:
    """Email the operator. Prefers the local ``sendmail`` MTA (no credential needed); falls
    back to authenticated SMTP if ``LORE_SMTP_PASSWORD`` is set."""
    to = os.environ.get("LORE_NOTIFY_TO", "").strip()
    if not to:
        return  # email disabled
    pw = os.environ.get("LORE_SMTP_PASSWORD", "").strip()
    if pw:
        _notify_email_smtp(subject, body, to, pw)
    else:
        _notify_email_sendmail(subject, body, to)


def _notify_email_sendmail(subject: str, body: str, to: str) -> None:
    import shutil
    import subprocess
    sm = os.environ.get("SENDMAIL_PATH") or shutil.which("sendmail") or "/usr/sbin/sendmail"
    frm = os.environ.get("LORE_MAIL_FROM", "lore@simon.local")
    msg = f"To: {to}\nFrom: Lore <{frm}>\nSubject: {subject}\nContent-Type: text/plain; charset=utf-8\n\n{body}\n"
    try:
        subprocess.run([sm, "-t", "-oi"], input=msg.encode("utf-8"), timeout=15, check=False)
        logger.info("usage email sent to %s via sendmail", to)
    except Exception:
        logger.warning("usage email (sendmail) failed", exc_info=True)


def _notify_email_smtp(subject: str, body: str, to: str, pw: str) -> None:
    user = os.environ.get("LORE_SMTP_USER", to).strip()
    host = os.environ.get("LORE_SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("LORE_SMTP_PORT", "587"))
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = user
        msg["To"] = to
        with smtplib.SMTP(host, port, timeout=15) as s:
            s.ehlo()
            s.starttls()
            s.login(user, pw)
            s.sendmail(user, [to], msg.as_string())
        logger.info("usage email sent to %s via SMTP", to)
    except Exception:
        logger.warning("usage email (SMTP) failed", exc_info=True)
