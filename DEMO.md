# Lore — 3-minute demo script

**Goal of the video:** show Lore *doing deep research* over a team's Slack memory and landing
the money-shot — finding a decision that was **reversed** and stating the current, cited answer.

## Setup (once, before recording)
1. Reinstall the app from `manifest.yaml` (grants assistant/channels/canvas scopes).
2. Seed the story:  `SLACK_BOT_TOKEN=xoxb-… .venv/bin/python scripts/seed_corpus.py`
   → creates #pricing, #leadership, #engineering, #product, #general and posts a real arc:
   **Pro tier set to $29** in #pricing, later **reversed to $49** in #leadership, plus
   deploy-pipeline / roadmap / onboarding noise. Copy the printed `LORE_CHANNELS=…` into `.env`.
3. Start Lore:  `OLLAMA_API_BASE=… LORE_MODEL=qwen3.5:35b-a3b python -m conduit.slack_app`

## Recording (aim ~3:00)

**0:00–0:15 — The hook.** Open the **Lore** assistant (split-view). It greets you and shows
**suggested prompts**. Read the first one aloud — *"I'm new here — what's the story behind our
pricing?"* — and click it. (This is the For-Good framing on screen: a newcomer's question.)

**0:15–1:20 — Watch it research (the Design centerpiece).** The research trace **streams live**
in one message, editing in place as each phase completes:
`🔍 Decomposing → 2 sub-queries` · `🕸️ resolved "pricing tier" via MCP glossary` ·
`🔎 Searching "pricing tier decision" → 3 hits in #pricing, #leadership` ·
`✅ Cross-checking for gaps` · `🕸️ 6 entities · 2 decisions · 1 reversal` · `📄 Writing cited report`.
Narrate: *"It broke the question down, consulted our glossary over MCP, searched multiple
channels, and built a knowledge graph of the decisions."*

**1:20–2:30 — The money-shot.** A **Canvas** opens. Point at:
- **🕸️ Decision timeline** — **$29** (#pricing) → **$49** (#leadership) — **Current: $49**
- The answer: *"We set the Pro tier at **$29** [1], then **reversed it to $49** after a market
  review [2] — the current answer is **$49**."*
- Click a **[1] / [2] citation** → it deep-links straight to the exact source message in Slack.

Narrate: *"It didn't just find messages — it found the **reversal I'd forgotten**, resolved the
timeline, and told me the current answer, with every claim cited and clickable."*

**2:30–3:00 — Why it wins.** *"Perplexity-style deep research, native to Slack, over your own
history. It uses all three platform technologies — the AI assistant surface, a real MCP server,
and the search substrate — and it's an Agent for Good: every new hire gets the veteran's answer,
instantly and cited. Institutional memory shouldn't be a privilege of the tenured."*

## If live RTS/scopes are unavailable during recording
Run the offline demo — it exercises the identical pipeline and prints the same money-shot:
```bash
.venv/bin/python scripts/run_demo.py     # -> answer, decision timeline, demo_output.json
```
`scripts/live_smoke.py` runs the full real-Slack path (index → research → Canvas → post) in one
command once the app is reinstalled and the corpus seeded.
