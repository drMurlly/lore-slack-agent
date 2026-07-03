#!/usr/bin/env python3
"""One-shot LIVE end-to-end smoke test against the real Slack workspace.

Proves the money-shot on real infrastructure without needing the Socket-Mode event loop:
indexes the seeded channels via `SlackHistoryRTS` (real `conversations.history`), runs the
real research pipeline with the local model, builds + creates a **Canvas** via the live
`canvases.create`, shares it, and posts the cited answer + a Canvas link to #general.

Prereqs (after reinstalling the app with the full scopes + running scripts/seed_corpus.py):

    export SLACK_BOT_TOKEN=xoxb-…
    export OLLAMA_API_BASE=http://192.168.10.52:11434/v1   # or your Ollama
    export LORE_MODEL=qwen3.5:35b-a3b
    export LORE_CHANNELS="C…:pricing,C…:leadership,…"       # from seed_corpus.py
    .venv/bin/python scripts/live_smoke.py

Prints the Canvas URL + the posted message ts. Exits non-zero on failure.
"""
from __future__ import annotations

import os
import sys

QUESTION = "What did we decide about pricing, and did anything change since?"


def main() -> int:
    token = os.environ.get("SLACK_BOT_TOKEN", "")
    if not token.startswith("xoxb-"):
        print("Set SLACK_BOT_TOKEN=xoxb-…", file=sys.stderr)
        return 1

    from slack_sdk import WebClient
    from conduit.live_rts import SlackHistoryRTS
    from conduit.research import run, synthesize
    from conduit.canvas import build_report_markdown
    from conduit.blocks import build_answer_blocks, final_block

    client = WebClient(token=token)
    auth = client.auth_test()
    team_url = (auth.get("url") or "").rstrip("/")
    team_id = auth.get("team_id", "")
    print(f"auth ok: team={auth.get('team')} bot={auth.get('user')}")

    # channels: from LORE_CHANNELS, else every channel the bot is a member of.
    raw = os.environ.get("LORE_CHANNELS", "").strip()
    if raw:
        channels = {}
        for part in raw.split(","):
            cid, _, name = part.partition(":")
            channels[cid.strip()] = (name.strip() or cid.strip())
    else:
        channels = {}
        resp = client.conversations_list(types="public_channel", exclude_archived=True, limit=200)
        for c in resp["channels"]:
            if c.get("is_member"):
                channels[c["id"]] = c["name"]
    if not channels:
        print("No channels to index — run scripts/seed_corpus.py and/or set LORE_CHANNELS.", file=sys.stderr)
        return 1
    print(f"indexing {len(channels)} channel(s): {', '.join('#'+n for n in channels.values())}")

    rts = SlackHistoryRTS(client, channels=channels, team_url=team_url).refresh()
    print(f"index: {rts.index_stats}")

    from conduit.agent import OllamaLLMClient, FakeLLMClient
    if os.environ.get("OLLAMA_API_BASE"):
        llm = OllamaLLMClient(model=os.environ.get("LORE_MODEL", "llama3.2"), timeout=150)
    else:
        print("! OLLAMA_API_BASE not set — using deterministic fake LLM (answer quality reduced)")
        llm = FakeLLMClient()

    print(f"\nQ: {QUESTION}")
    result = run(QUESTION, rts, llm)
    answer = synthesize(result, llm)
    print(f"evidence={len(result.evidence)} citations={len(answer.citations)} "
          f"drift={(answer.drift.old_value + ' -> ' + answer.drift.current_value) if answer.drift else None}")
    print(f"graph={answer.graph_summary}")
    print("\n--- ANSWER ---\n" + answer.text + "\n--- END ---")

    if not result.evidence:
        print("No evidence indexed — is the bot in the seeded channels?", file=sys.stderr)
        return 2

    # Canvas
    md = build_report_markdown(answer, QUESTION, graph=getattr(result, "graph", None))
    canv = client.canvases_create(title=f"Lore — {QUESTION[:60]}",
                                  document_content={"type": "markdown", "markdown": md})
    canvas_id = canv.get("canvas_id", "")
    canvas_url = f"{team_url}/docs/{team_id}/{canvas_id}" if canvas_id else ""
    print(f"\ncanvas: {canvas_url}")

    # Post the answer + Canvas link to #general (or the first channel).
    target = next((cid for cid, n in channels.items() if n == "general"), next(iter(channels)))
    try:
        client.canvases_access_set(canvas_id=canvas_id, access_level="read", channel_ids=[target])
    except Exception as e:
        print(f"! canvases.access.set: {e}")
    blocks = build_answer_blocks(answer.text[:1500])
    if canvas_url:
        blocks += final_block(answer.text[:280], canvas_url)
    posted = client.chat_postMessage(channel=target, blocks=blocks, text=answer.text[:1500])
    print(f"posted to #{channels[target]} ts={posted.get('ts')}")
    print("\n✅ LIVE E2E OK — open the Canvas URL above; every citation deep-links to a source message.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
