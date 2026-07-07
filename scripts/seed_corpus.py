#!/usr/bin/env python3
"""Seed the demo workspace with a realistic story arc for the Lore money-shot.

Creates (or reuses) a handful of channels, invites the Lore bot, and posts a corpus whose
centerpiece is a genuine pricing *reversal*: decided **$29** in an early #pricing thread,
then **reversed to $49** later in #leadership — plus deploy-pipeline, Q3-roadmap and noise
so the retrieval + contradiction resolver have something real to work through.

Requires a bot token with: channels:manage (create), channels:join, chat:write (post),
chat:write.customize (persona names). Run AFTER the app is (re)installed with those scopes:

    SLACK_BOT_TOKEN=xoxb-… .venv/bin/python scripts/seed_corpus.py

It is idempotent-ish: channels are looked up by name and reused; messages are appended.
Prints a LORE_CHANNELS line to paste into .env so live research indexes exactly these.
"""
from __future__ import annotations

import os
import sys
import time

try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
except ImportError:
    print("slack_sdk required: pip install -e .", file=sys.stderr)
    sys.exit(1)

# (channel, author persona, message). Order matters: earlier posts get earlier timestamps,
# so the reversal ($29 -> $49) is chronologically real.
STORY: list[tuple[str, str, str]] = [
    ("pricing", "Maya (PM)", "Kicking off pricing for launch. Proposal: a single Pro tier at $29 per seat / month."),
    ("pricing", "Devon (Finance)", "Modeled it — $29 covers infra + support at our target margin. I'm good with $29."),
    ("pricing", "Maya (PM)", "Decision: we set the Pro pricing tier to $29 per seat for launch. Shipping it."),
    ("engineering", "Raj (Eng)", "Deploy pipeline: merges to main auto-deploy to staging; prod is a manual gate + smoke tests."),
    ("engineering", "Sam (SRE)", "Rollback is one command: `lore deploy rollback <sha>`. Runbook is in the canvas."),
    ("product", "Maya (PM)", "Q3 roadmap: ship the assistant split-view first, then the App Home onboarding."),
    ("general", "Nadia (Ops)", "Welcome new folks! If you're new, ask Lore — it cites its sources from our history."),
    ("leadership", "Priya (CEO)", "Board pushed back on pricing — we're underpricing vs competitors at $29."),
    ("leadership", "Priya (CEO)", "Decision: after the market review we changed the Pro pricing tier from $29 to $49 per seat."),
    ("leadership", "Devon (Finance)", "Updated the model for $49 — this is the current pricing tier going forward."),
]

DEMO_CHANNELS = ["pricing", "engineering", "product", "general", "leadership"]


def _ensure_channel(client: WebClient, name: str) -> str:
    """Create the channel if needed, return its id; ensure the bot is a member."""
    # find existing
    cursor = None
    while True:
        resp = client.conversations_list(types="public_channel", exclude_archived=True,
                                         limit=200, **({"cursor": cursor} if cursor else {}))
        for c in resp["channels"]:
            if c["name"] == name:
                if not c.get("is_member"):
                    try:
                        client.conversations_join(channel=c["id"])
                    except SlackApiError:
                        pass
                return c["id"]
        cursor = (resp.get("response_metadata") or {}).get("next_cursor")
        if not cursor:
            break
    # create
    created = client.conversations_create(name=name, is_private=False)
    cid = created["channel"]["id"]
    try:
        client.conversations_join(channel=cid)
    except SlackApiError:
        pass
    return cid


def main() -> int:
    token = os.environ.get("SLACK_BOT_TOKEN", "")
    if not token.startswith("xoxb-"):
        print("Set SLACK_BOT_TOKEN=xoxb-… (bot must have channels:manage/join, chat:write[.customize]).",
              file=sys.stderr)
        return 1
    client = WebClient(token=token)

    ids: dict[str, str] = {}
    for name in DEMO_CHANNELS:
        try:
            ids[name] = _ensure_channel(client, name)
            print(f"channel #{name} -> {ids[name]}")
        except SlackApiError as e:
            print(f"! #{name}: {e.response.get('error')}", file=sys.stderr)

    for channel, persona, text in STORY:
        cid = ids.get(channel)
        if not cid:
            continue
        try:
            client.chat_postMessage(channel=cid, text=text, username=persona,
                                    icon_emoji=":speech_balloon:")
            print(f"  #{channel} <{persona}>: {text[:60]}…")
            time.sleep(1.1)  # keep timestamps monotonic + avoid rate limits
        except SlackApiError as e:
            print(f"  ! post to #{channel}: {e.response.get('error')}", file=sys.stderr)

    line = ",".join(f"{ids[n]}:{n}" for n in DEMO_CHANNELS if n in ids)
    print("\nPaste into .env so live research indexes exactly these channels:")
    print(f"LORE_CHANNELS={line}")
    print("\nMoney-shot question:")
    print('  "What did we decide about pricing, and did anything change since?"  -> $29 then $49 (current $49)')
    return 0


if __name__ == "__main__":
    sys.exit(main())
