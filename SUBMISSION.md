# Lore — Submission for the Slack Agent Builder Challenge

> **Deep research over your team's own Slack memory.** Ask a hard question; Lore decomposes
> it, runs a multi-hop search across your channels/threads/files, builds an ephemeral
> **knowledge graph** of decisions, resolves contradictions and timeline drift, and answers
> with **inline citations that deep-link to the exact source messages** — delivered as a
> **Canvas** report with a live streaming research trace in the assistant split-view.

## Track
**Slack Agent for Good** (primary) · *New Slack Agent* (fallback)

**Framing — institutional-knowledge equity.** Organizational memory shouldn't be a
privilege of the tenured. Every new hire, volunteer, or open-source contributor gets the
same instant, cited answer from the org's whole history as a five-year veteran — so mission
continuity survives churn. The assistant's first suggested prompt is literally *"I'm new
here — what's the story behind our pricing?"*, and the App Home leads with that promise.

## Required technologies — Lore uses all three (judges reward combining ≥2)
- [x] **Slack AI capabilities** — the app is an **AI Assistant** (Agents & AI Apps): assistant
      split-view container, `assistant_thread_started` greeting, **suggested prompts**, a
      **live streaming research trace** (one message edited in place as each phase completes),
      and status updates. `assistant_surface.py`, `slack_app.py`, `blocks.py`.
- [x] **MCP server integration** — a real **MCP client→server round-trip** over the official
      `mcp` SDK: `servers/glossary_server.py` is a FastMCP server exposing `lookup_terms`/
      `define` over an org glossary; `mcp_manager.py` is the stdio client; the research loop
      consults it (a genuine `initialize → tools/list → tools/call` handshake) to resolve
      domain terms/acronyms before searching. `research.py:_consult_glossary`.
- [x] **Real-Time Search API** — the multi-hop retrieval substrate. `rts_client.py` is the
      RTS-shaped `search(query) -> [SearchHit]` seam; because the RTS API is currently
      allowlisted, live retrieval runs through the **interchangeable** `SlackHistoryRTS`
      backend (`conversations.history` indexing + lexical/recency ranking) exposing the *same*
      seam — so the whole pipeline runs on real workspace data today and swaps to the official
      RTS API by replacing one class the day the scope is granted. `live_rts.py`, `fake_rts.py`.

## The 4 judging criteria (equal weight)
| Criterion | Lore's answer |
|---|---|
| **Quality of the Idea** | Perplexity-style **deep research** — the defining AI pattern of 2025-26 — brought first-to-Slack over your *conversational* data, with a genuine multi-hop loop and a knowledge graph. Not the saturated standup/Q&A/BI crowd; not a single-query search wrapper. |
| **Potential Impact** | Every knowledge worker's daily pain: "where was that decided / what changed / what do we actually know about X," buried across months of threads nobody remembers. For-Good: newcomers and underrepresented staff who *don't* "know who to ask" get the same instant, cited answer as a veteran. |
| **Technological Implementation** | Question **decomposition** → multi-hop **retrieval fan-out** with a follow-up hop on thin coverage → **ephemeral knowledge graph** (entities + typed `decided`/`changed`/`supersedes` edges) → **deterministic** contradiction / timeline-drift resolution → **citation-grounded synthesis** with deep-links → **Canvas** write-back. MCP consult in the loop. 154 tests, all offline-runnable. |
| **Design (+ Best UX)** | Assistant split-view with a **streaming research trace** ("🔍 Decomposing… 🔎 Searching #pricing → 4 hits… ✅ cross-checking… 🕸️ knowledge graph…"), suggested prompts, a **Lore-branded App Home**, friendly empty/error states, and a beautiful **Canvas** whose Decision-timeline and every citation deep-link back to the exact source message. Block Kit throughout — never a wall of text. |

## The demo money-shot (3-min video)
Open the **Lore** assistant and click *"…what's the story behind our pricing?"* (or type
*"What did we decide about pricing, and did anything change since?"*). Watch it **stream** its
plan and live searches, then a **Canvas** appears:

> **🕸️ Decision timeline** — **$29** (#pricing) → **$49** (#leadership) — **Current: $49**
> *"We set the Pro tier at **$29** ([#pricing]), then **reversed it to $49** after a market
> review ([#leadership]) — the current answer is **$49**,"* every claim click-through to the
> exact message.

That "it found the reversal I'd forgotten" moment is the point. **It is deterministic** —
`contradiction.py` + the knowledge graph resolve the reversal from timeline-ordered evidence,
so it surfaces correctly regardless of how the local model phrases its prose. *(Real-model run
with `qwen3.5` verified: clean cited answer, $10→$20 on the offline corpus.)*

## Deliverables checklist
- [x] Functional app surface — `/lore` slash command · `@Lore` mention · **assistant thread** — `slack_app.py`
- [x] Bootable in Socket Mode (`python -m conduit.slack_app`); manifest at `manifest.yaml`
- [x] Multi-hop retrieval, follow-up hop, ≥2 channels per query — `research.py`, `live_rts.py`
- [x] **Real MCP** client→server round-trip in the loop — `mcp_manager.py`, `servers/glossary_server.py`
- [x] Ephemeral knowledge graph (entities + typed edges + supersedes) — `knowledge_graph.py`
- [x] Deterministic contradiction / timeline-drift resolution — `contradiction.py`
- [x] Citation-grounded synthesis, deep-links to source messages — `citations.py`
- [x] **Canvas** report (Decision timeline + cited answer) — `canvas.py`; **live `canvases.create` contract verified**
- [x] Streaming assistant trace + suggested prompts + App Home — `assistant_surface.py`, `blocks.py`
- [x] Green test suite — `python -m pytest -q` → **154 passing**
- [x] Standalone offline demo — `scripts/run_demo.py` → `demo_output.json` (+ `DEMO.md`)
- [x] Architecture diagram — `README.md` (mermaid) + `docs/architecture.mmd` + `docs/architecture.png` (for Devpost)
- [x] **Live demo video (~2.5 min)** — `lore-demo.mp4`, built from a real live run ($29→$49, cited)
- [x] **Verified live** — full E2E ran in the "Simon" workspace (real Canvas + posted answer)
- [x] Public repo — <https://github.com/drMurlly/lore-slack-agent>
- [ ] **Sandbox access** granted to `slackhack@salesforce.com` + `testing@devpost.com` (invite via Slack → *Invite people*)
- [ ] Upload `lore-demo.mp4` to YouTube/Vimeo and paste the link into the Devpost form

## Reproduce it
```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q              # 154 green, fully offline
.venv/bin/python scripts/run_demo.py       # the money-shot over a seeded corpus -> demo_output.json
```
Live (in the sandbox): reinstall the app from `manifest.yaml`, `scripts/seed_corpus.py` to
seed the story, then `scripts/live_smoke.py` (or run `python -m conduit.slack_app` and ask in
Slack). See `DEMO.md`.
