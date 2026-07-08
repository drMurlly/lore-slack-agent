# Lore — Submission for the Slack Agent Builder Challenge

> **Deep research over your team's own Slack memory.** Ask a hard question; Lore decomposes
> it, runs a multi-hop search across your channels and threads, builds an ephemeral
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
      and status updates. The trace, the cited **Canvas**, and the Block Kit money-shot card
      (Decision-Graph badge → timeline → conflicting-signals) render on **every surface** —
      `/lore`, `@Lore`, DM, and the Assistant — not just the split-view. `assistant_surface.py`,
      `slack_app.py`, `blocks.py`.
- [x] **MCP server integration** — a real **MCP client→server round-trip** over the official
      `mcp` SDK: `servers/glossary_server.py` is a FastMCP server exposing `lookup_terms`/
      `define` over an org glossary; `mcp_manager.py` is the stdio client; the research loop
      consults it (a genuine `initialize → tools/list → tools/call` handshake) to resolve
      domain terms/acronyms and **feeds the definitions back into retrieval** (an `ARR` question
      also searches `Annual Recurring Revenue`) — so removing MCP measurably lowers recall.
      `research.py:_consult_glossary` + `_glossary_expansions`.
- [x] **Real-Time / Search API** — the multi-hop retrieval substrate, one `search(query) ->
      [SearchHit]` seam with three interchangeable backends: **`RTSClient` calls Slack's official
      `search.messages`** (functional with a user token + `search:read`; opt in via
      `LORE_USE_RTS_API`), **`SlackHistoryRTS`** indexes `conversations.history` with a
      lexical+recency ranker (the **default**, because the sandbox app holds a *bot* token and
      `search.messages` is user-token-only by Slack's rules), and **`FakeRTS`** for offline
      tests. The pipeline runs unchanged on any backend; switching is one line in `_build_rts`.
      `rts_client.py`, `live_rts.py`, `fake_rts.py`.

## The 4 judging criteria (equal weight)
| Criterion | Lore's answer |
|---|---|
| **Quality of the Idea** | Perplexity-style **deep research** — the defining AI pattern of 2025-26 — brought first-to-Slack over your *conversational* data, with a genuine multi-hop loop and a knowledge graph. Not the saturated standup/Q&A/BI crowd; not a single-query search wrapper. |
| **Potential Impact** | Every knowledge worker's daily pain: "where was that decided / what changed / what do we actually know about X," buried across months of threads nobody remembers. For-Good: newcomers and underrepresented staff who *don't* "know who to ask" get the same instant, cited answer as a veteran. |
| **Technological Implementation** | Question **decomposition** → multi-hop **retrieval fan-out** with a follow-up hop on thin coverage → **ephemeral knowledge graph** (entities + typed `decided`/`changed`/`supersedes` edges) → **deterministic** contradiction / timeline-drift resolution → **citation-grounded synthesis** with deep-links → **Canvas** write-back. MCP consult in the loop. 191 tests, all offline-runnable. |
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
- [x] Functional app surface — `/lore` · `@Lore` · DM · **assistant thread**, all delivering the streaming trace + cited Canvas + decision timeline — `slack_app.py`
- [x] Bootable in Socket Mode (`python -m conduit.slack_app`); manifest at `manifest.yaml`
- [x] Multi-hop retrieval, follow-up hop, ≥2 channels per query — `research.py`, `live_rts.py`
- [x] **Real MCP** client→server round-trip in the loop — `mcp_manager.py`, `servers/glossary_server.py`
- [x] Ephemeral knowledge graph (entities + typed edges + supersedes) — `knowledge_graph.py`
- [x] Deterministic contradiction / timeline-drift resolution — `contradiction.py`
- [x] Citation-grounded synthesis, deep-links to source messages — `citations.py`
- [x] **Canvas** report (Decision timeline + cited answer) — `canvas.py`; **live `canvases.create` contract verified**
- [x] Streaming assistant trace + suggested prompts + App Home — `assistant_surface.py`, `blocks.py`
- [x] Green test suite — `python -m pytest -q` → **191 passing**
- [x] **Hardened for live judging** — two adversarial robustness passes fixed 30 verified issues, each with a regression test. First pass: thread-safe event dedup (no duplicate answers / crashes under Bolt's worker pool), **topic-anchored drift** (no fabricated cross-topic reversals), an **off-topic dilution filter**, Block-Kit 3000-char clipping, **untrusted-quote sanitization**, a streaming-trace flood guard, per-sub-query error isolation, and Unicode-aware indexing. Second pass (multi-agent, adversarially verified): a **currency-suffix parse bug** (`$49 monthly` → `$49m`), **cue-aware current-value resolution** (`$20, up from $50` no longer inverts), **numeric-magnitude canonicalization** (`$49` vs `$49.00` isn't a false reversal), a **num/percent reversal gate** (a planned-vs-actual count isn't fabricated into a decision reversal), **answer-body neutralization** (an injected message can't `@channel`-ping or plant a live link under Lore's identity), **Canvas-autolink escaping**, a **View-Canvas access-grant gate** (the button never links a canvas the judge can't open), a **sentence-final-token recall fix**, a shorter index-cache TTL (live edits visible in seconds), and **corrected demo citation grounding** (every `[n]` deep-links to the message that asserts its value)
- [x] Standalone offline demo — `scripts/run_demo.py` → `demo_output.json` (+ `DEMO.md`)
- [x] Architecture diagram — `README.md` (mermaid) + `docs/architecture.mmd` + `docs/architecture.png` (for Devpost)
- [x] **Live demo video (~1 min)** — `lore-demo.mp4`, **real screen-capture of the live Slack client** (ask → streaming research trace → cited money-shot card $29→$49 → citation deep-link to the source message)
- [x] **Verified live** — full E2E ran in the "Simon.Ltd" workspace (real Canvas + posted answer)
- [x] Public repo — <https://github.com/drMurlly/lore-slack-agent>
- [ ] **Sandbox access** granted to `slackhack@salesforce.com` + `testing@devpost.com` (invite via Slack → *Invite people*)
- [x] Demo video on YouTube — <https://www.youtube.com/watch?v=F3_I2FzNH9I>

## Reproduce it
```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q              # 191 green, fully offline
.venv/bin/python scripts/run_demo.py       # the money-shot over a seeded corpus -> demo_output.json
```
Live (in the sandbox): reinstall the app from `manifest.yaml`, `scripts/seed_corpus.py` to
seed the story, then `scripts/live_smoke.py` (or run `python -m conduit.slack_app` and ask in
Slack). See `DEMO.md`.
