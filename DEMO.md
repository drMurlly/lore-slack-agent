# Lore — demo

**Watch the demo:** <https://www.youtube.com/watch?v=F3_I2FzNH9I> — a short screen-recording of the **real Slack client**.

**What it shows:** a teammate asks Lore *"What did we decide about pricing, and did anything change
since?"* in Slack. Lore streams its **live research trace** (decompose the question → multi-hop
search across channels), then posts a cited **money-shot card**: the Pro tier was set at **$29**,
later changed to **$49** after a market review, and Lore resolves the current answer to **$49** — a
reversal a keyword search would miss. Every citation **deep-links to the exact source message**, and
a shared **Canvas** carries the full report.

## Reproduce it

**Offline (no Slack, no GPU)** — the identical pipeline over a seeded corpus:

```bash
.venv/bin/python -m pytest -q              # green, fully offline
.venv/bin/python scripts/run_demo.py       # prints the cited answer + decision timeline -> demo_output.json
```

**Live in Slack:**

1. Install the app from `manifest.yaml` (Socket Mode) and fill `.env` (`SLACK_*`, `OLLAMA_API_BASE`,
   `LORE_MODEL`, `LORE_MCP_GLOSSARY=1`).
2. Seed the story: `.venv/bin/python scripts/seed_corpus.py` — creates the demo channels and posts a
   real decision arc (including the **$29 → $49** pricing reversal); paste the printed
   `LORE_CHANNELS=…` into `.env`.
3. Start Lore: `python -m conduit.slack_app` — or `scripts/live_smoke.py` for a one-shot
   index → research → Canvas → post.
4. Ask in Slack: open the **Lore** assistant and click a suggested prompt, `@Lore` in a channel, or
   type `/lore <question>`.

The reversal is resolved **deterministically** — `contradiction.py` and the knowledge graph order the
evidence by time and pick the newest value — so it surfaces correctly regardless of how the local
model phrases its prose.
