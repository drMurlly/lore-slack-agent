#!/usr/bin/env python3
"""Seed MORE channels into Simon.Ltd to stress-test Lore: extra topics, reversals, acronyms,
and cross-channel decisions. Run with a bot token that has channels:manage/join + chat:write."""
from __future__ import annotations
import os, sys, time
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# (channel, persona, message) — order matters; earlier posts get earlier timestamps so the
# reversals below are chronologically real.
STORY = [
    # security — an SSO policy that gets reversed
    ("security", "Aria (Security)", "Proposal: require SSO for all users. Decision: SSO is mandatory for every account."),
    ("security", "Devon (Finance)", "SSO for free-tier users hurts signups. Can we revisit?"),
    ("security", "Aria (Security)", "Decision: after review we made SSO optional for the free tier; still mandatory for Business and Enterprise."),
    ("security", "Aria (Security)", "MFA stays required for all admins. No change there."),
    # infra — a rate limit that gets raised (numeric reversal, different value-class than $)
    ("infra", "Priya (Eng)", "Decision: API rate limit set to 100 requests/min per key for launch."),
    ("infra", "Sam (SRE)", "Load tests show we can handle much more. Recommend raising it."),
    ("infra", "Priya (Eng)", "Decision: after load testing we raised the API rate limit from 100 to 300 requests/min per key."),
    # hiring — headcount revised after funding
    ("hiring", "Nadia (Ops)", "Decision: Q3 plan is to hire 3 engineers."),
    ("hiring", "Priya (CEO)", "With the new funding round closed, we can scale faster."),
    ("hiring", "Nadia (Ops)", "Decision: revised Q3 hiring from 3 to 5 engineers after the funding round."),
    # support — SLA tightened
    ("support", "Maya (PM)", "Decision: support SLA is a 24-hour first response for all tiers."),
    ("support", "Priya (CEO)", "Enterprise customers need faster. Let's tighten it for them."),
    ("support", "Maya (PM)", "Decision: tightened the Enterprise SLA from 24 hours to 4 hours; other tiers stay at 24 hours."),
    # data — metrics + acronyms for the glossary
    ("data", "Devon (Finance)", "ARR crossed $1.2M this quarter; MAU is up 18% month over month."),
    ("data", "Devon (Finance)", "Decision: our north-star KPI for H2 is weekly active teams, not MAU."),
    ("data", "Aria (Security)", "Reminder: churn is trending down after the onboarding revamp."),
]
CHANNELS = ["security", "infra", "hiring", "support", "data"]


def ensure(client, name):
    cursor = None
    while True:
        r = client.conversations_list(types="public_channel", exclude_archived=True, limit=200,
                                      **({"cursor": cursor} if cursor else {}))
        for c in r["channels"]:
            if c["name"] == name:
                if not c.get("is_member"):
                    try: client.conversations_join(channel=c["id"])
                    except SlackApiError: pass
                return c["id"]
        cursor = (r.get("response_metadata") or {}).get("next_cursor")
        if not cursor: break
    cid = client.conversations_create(name=name, is_private=False)["channel"]["id"]
    try: client.conversations_join(channel=cid)
    except SlackApiError: pass
    return cid


def main():
    tok = os.environ.get("SLACK_BOT_TOKEN", "")
    if not tok.startswith("xoxb-"): print("need SLACK_BOT_TOKEN", file=sys.stderr); return 1
    c = WebClient(token=tok)
    ids = {}
    for n in CHANNELS:
        try:
            ids[n] = ensure(c, n); print(f"#{n} -> {ids[n]}")
        except SlackApiError as e:
            print(f"! #{n}: {e.response.get('error')}", file=sys.stderr)
    for chan, persona, text in STORY:
        cid = ids.get(chan)
        if not cid: continue
        try:
            c.chat_postMessage(channel=cid, text=text, username=persona, icon_emoji=":speech_balloon:")
            print(f"  #{chan} <{persona}>: {text[:55]}…"); time.sleep(1.1)
        except SlackApiError as e:
            print(f"  ! {chan}: {e.response.get('error')}", file=sys.stderr)
    print("\nnew channels:", ",".join(f"{ids[n]}:{n}" for n in CHANNELS if n in ids))
    return 0


if __name__ == "__main__":
    sys.exit(main())
