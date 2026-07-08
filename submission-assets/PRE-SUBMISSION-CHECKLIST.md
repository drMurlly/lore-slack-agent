# Lore — Pre-Submission Checklist (deadline **2026-07-13 17:00 PT**)

The code is done and live. What's left is human choreography. Work top-to-bottom.

## 🔴 Hard blockers (submission fails without these)
- [ ] **Grant judges MEMBER access** to the Simon.Ltd workspace — Slack → *Invite people* →
      `slackhack@salesforce.com` **and** `testing@devpost.com`. **Confirm the role is *Member*,
      not *Guest*** (guests can't see channels/canvases → judges see a broken app). Bot tokens
      can't invite; this is a manual click.
- [ ] **Upload the architecture diagram to the Devpost FILE-UPLOAD field** — use
      `docs/architecture.png`. It is a *required* item and must go in the file field, **not** the
      image carousel.
- [ ] **Confirm the video is public** — open <https://www.youtube.com/watch?v=F3_I2FzNH9I> in an
      **incognito window** (logged out). If it doesn't play, flip it to Public/Unlisted. Do this
      **≥24h before the deadline** (processing lag).
- [ ] **Click Submit on Devpost** — track: **Slack Agent for Good** (primary). Paste the sandbox
      URL + the YouTube link.

## 🟡 Should-do (cheap, raises the score)
- [ ] **Steer judges to any surface** — the money-shot now renders on `/lore`, `@Lore`, DM, **and**
      the Assistant (this push). Still, lead them to the **Assistant split-view** in the video: it's
      the richest (streaming trace + setStatus). Suggested-prompt one click = the whole demo.
- [ ] **Re-record or re-caption the demo?** Optional. The current video is from a real $29→$49 run
      and is strong. If you re-shoot, open on the *problem* (see description draft below), show it
      working in the first 30s, and demo `/lore` too (now that it shows the full Canvas + timeline).
- [ ] **Name** — keep **Lore**. It's specific, human, memorable — exactly what the guidance asks
      for. Do not rename.

## What changed in this push (mention in the write-up — these are differentiators)
- **Money-shot on every surface.** `/lore`, `@Lore`, and DM now stream the research trace and post
  the cited **Canvas + Decision-Graph badge + decision timeline + conflicting-signals** card — not
  just the Assistant. A judge sees the wow moment however they test.
- **The RTS/Search seam is a real, switchable backend.** `RTSClient` calls Slack's official
  `search.messages` (works with a user token + `search:read`); `SlackHistoryRTS` is the bot-token
  default. No dead code — three interchangeable backends behind one seam.
- **MCP earns its place.** The glossary consult now **feeds resolved definitions back into
  retrieval**, so a question about "ARR" also finds "Annual Recurring Revenue." Remove MCP and
  recall on jargon questions drops — it would be *meaningfully worse without it*.
- **Cleaner codebase.** Removed the leftover Conduit starter scaffold (non-MCP demo servers, an
  unused ReAct loop, an old home view). Every file now serves Lore. 149 tests, all offline.

---

## Devpost text description — DRAFT (rewrite in your own voice; judges spot AI boilerplate)

> **Answer the three questions in the first paragraph: What is it? Who is it for? Why does it matter?**

**What it does.** Lore is a deep-research agent that lives in Slack. Ask it a hard question —
*"what did we decide about pricing, and did anything change?"* — and it decomposes the question,
runs a multi-hop search across your channels and threads, builds a small knowledge graph of the
decisions it finds, and answers with inline citations that deep-link to the exact source messages —
delivered as a Slack Canvas with a live research trace you watch unfold.

**Who it's for.** The person who *doesn't* already know where the answer is buried: the new hire on
day 3, the volunteer who joined last week, the contributor who wasn't in the room. The veteran knows
which thread the decision is in; everyone else has to interrupt someone or give up. Lore gives every
newcomer the veteran's answer — instant, cited, and traceable.

**Why it matters (be specific).** Reversed decisions are the trap: the $29 price that quietly became
$49, the policy that changed in a channel you're not in. A keyword search hands you the *stale*
message with equal confidence. Lore resolves the timeline deterministically and tells you the
**current** answer with the receipts — so institutional memory survives churn instead of walking out
the door. That's the "Agent for Good" thesis: knowledge equity, not knowledge hoarding.

*(Then: 2–3 sentences on how it works — decomposition → multi-hop retrieval → knowledge graph →
deterministic timeline resolution → cited Canvas — and one line naming the three platform techs used
genuinely: Slack AI Assistant, a real MCP round-trip that feeds retrieval, and the interchangeable
RTS/Search seam.)*
