#!/usr/bin/env python3
"""Standalone demo for Lore — deep research over Slack memory. No live Slack, no GPU, no env.

Runs the REAL pipeline over a seeded FakeRTS corpus that contains a genuine pricing
reversal, and shows the money-shot: the answer surfaces BOTH values + the CURRENT one
(deterministically, via conduit.contradiction), with citations, delivered as a Canvas
report. Writes demo_output.json. Exits 0 on success, 1 on any exception.
"""
from __future__ import annotations

import json
import re
import sys
import traceback
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from conduit.agent import LLMClient
from conduit.fake_rts import CorpusMessage, FakeRTS
from conduit.research import run, synthesize
from conduit.canvas import build_report

QUESTION = "What did we decide about pricing, and did anything change since?"


def _ts(days_ago: int, hour: int = 12, minute: int = 0) -> str:
    """A Slack-style unix ts string for `days_ago` days back (older = smaller)."""
    dt = datetime.now(timezone.utc) - timedelta(days=days_ago)
    dt = dt.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return f"{int(dt.timestamp())}.000000"


def seed_corpus() -> FakeRTS:
    """~16 realistic messages across 5 channels, including a genuine pricing reversal."""
    corpus: dict[str, list[CorpusMessage]] = {
        "pricing": [
            CorpusMessage(text="We set the pricing tier to $10 per user for launch.",
                          channel="pricing", ts=_ts(70, 11, 0), author="alice"),
            CorpusMessage(text="Pricing tier discussion: are we underpriced vs competitors?",
                          channel="pricing", ts=_ts(45, 15, 30), author="bob"),
        ],
        "decisions": [
            CorpusMessage(text="After market review we changed the pricing tier to $20 per user.",
                          channel="decisions", ts=_ts(15, 9, 30), author="ceo"),
            CorpusMessage(text="Database choice: PostgreSQL for reliability.",
                          channel="decisions", ts=_ts(20, 16, 0), author="ceo"),
        ],
        "engineering": [
            CorpusMessage(text="We're going with Python for the backend, FastAPI framework.",
                          channel="engineering", ts=_ts(90, 14, 30), author="alice"),
            CorpusMessage(text="FastAPI it is. Let's use Pydantic for validation.",
                          channel="engineering", ts=_ts(89, 10, 15), author="bob"),
            CorpusMessage(text="Rate limit set to 100 req/min per key.",
                          channel="engineering", ts=_ts(40, 12, 0), author="dave"),
        ],
        "product": [
            CorpusMessage(text="Q3 roadmap: ship the assistant split-view first.",
                          channel="product", ts=_ts(30, 9, 0), author="carol"),
            CorpusMessage(text="Onboarding flow needs a cited-answers demo.",
                          channel="product", ts=_ts(25, 13, 0), author="carol"),
        ],
        "general": [
            CorpusMessage(text="Welcome to the team! Read the onboarding docs.",
                          channel="general", ts=_ts(100, 8, 0), author="bob"),
            CorpusMessage(text="Reminder: submit expense reports by month end.",
                          channel="general", ts=_ts(5, 17, 0), author="carol"),
        ],
    }
    return FakeRTS(corpus)


def _value_indices(synthesis_prompt: str) -> dict[str, int]:
    """Map each price value to the citation index of the evidence that actually asserts it.

    The synthesis prompt lists evidence as ``[n] Channel: …\\nText: …\\nPermalink: …`` blocks. The
    ranker reorders evidence, so a real model reads these indices and cites accordingly — this
    fake does the same, instead of hardcoding [1]/[2] (which would deep-link each claim to the
    WRONG source once the ranker moves the newest value to the top)."""
    out: dict[str, int] = {}
    for m in re.finditer(r"\[(\d+)\] Channel:.*?\nText: (.*?)\nPermalink:", synthesis_prompt, re.S):
        idx, text = int(m.group(1)), m.group(2)
        for val in ("$10", "$20"):
            if val in text and val not in out:
                out[val] = idx
    return out


class DemoLLMClient(LLMClient):
    """Deterministic LLM: decompose -> pricing sub-queries; synthesis -> cited reversal."""

    def chat(self, messages: list[dict[str, str]], tools: Optional[list[dict]] = None) -> dict[str, Any]:
        sysp = messages[0].get("content", "").lower()
        if "decompose" in sysp:
            return {"content": "What did we decide about the pricing tier?\n"
                               "Did the pricing tier change since launch?", "tool_calls": None}
        if "follow-up" in sysp or "fill gaps" in sysp:
            return {"content": "pricing tier change decision", "tool_calls": None}
        # synthesis — cite the ACTUAL evidence indices for each value so every [n] deep-links to
        # the message that asserts it (contradiction.py still appends the deterministic current
        # value). Falls back to 1/2 only if the prompt shape is unexpected.
        idx = _value_indices(messages[-1].get("content", ""))
        low, high = idx.get("$10", 1), idx.get("$20", 2)
        return {"content": f"The team set the pricing tier at $10 [{low}], then changed it to "
                           f"$20 [{high}] after a market review.", "tool_calls": None}


def _trace(phase: str, detail: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {phase}: {detail}")


def main() -> int:
    try:
        print("=" * 60)
        print("Lore Demo — Deep Research over Slack Memory")
        print("=" * 60)
        rts = seed_corpus()
        n_msgs = sum(len(v) for v in rts._corpus.values())
        _trace("seed", f"FakeRTS seeded with {n_msgs} messages across {len(rts._corpus)} channels")
        _trace("question", QUESTION)

        llm = DemoLLMClient()
        result = run(QUESTION, rts, llm)
        _trace("research", f"{len(result.evidence)} evidence items, {result.follow_up_hops} follow-up hop(s)")
        for ev in result.evidence[:6]:
            _trace("evidence", f"#{ev.channel} @ {ev.ts}: {ev.text[:60]}")

        answer = synthesize(result, llm)
        _trace("synthesis", f"{len(answer.citations)} citation(s)")
        print("\n--- ANSWER ---")
        print(answer.text)
        print("--- END ANSWER ---\n")

        # Grounding self-check: every "<value> [n]" claim must deep-link to a source message whose
        # quote actually contains that value — this is Lore's headline promise, so the demo fails
        # loudly rather than ever re-committing a mismatched artifact (the [1]->wrong-msg bug).
        cite_by_index = {c.index: c for c in answer.citations}
        for m in re.finditer(r"(\$\d[\d,]*)\s*\[(\d+)\]", answer.text):
            value, idx = m.group(1), int(m.group(2))
            cite = cite_by_index.get(idx)
            if cite is None or value not in cite.quote:
                print(f"GROUNDING ERROR: claim {value} [{idx}] does not link to a message "
                      f"containing {value!r} (linked quote: {cite.quote if cite else None!r})",
                      file=sys.stderr)
                return 1
        _trace("grounding", "every cited claim deep-links to a source that asserts its value ✓")

        graph = getattr(result, "graph", None)
        canvas = build_report(answer, QUESTION, graph=graph)
        n_blocks = len(canvas.get("document_content", {}).get("blocks", []))
        _trace("canvas", f"report built with {n_blocks} blocks")

        out = {
            "question": QUESTION,
            "answer": answer.text,
            "citations": [asdict(c) for c in answer.citations],
            "follow_up_hops": result.follow_up_hops,
            "evidence_count": len(result.evidence),
        }
        # Knowledge-graph summary + the resolved decision timeline — the visible proof of
        # deep research (reasoning over structure, not a hit list).
        if graph is not None:
            out["graph"] = graph.summary()
            topic = graph.primary_topic(QUESTION)
            if topic:
                timeline = [
                    {"value": graph.entities[e.obj].label, "channel": e.channel, "ts": e.ts}
                    for e in graph.timeline(topic) if e.obj in graph.entities
                ]
                out["timeline"] = timeline
                current, _edge = graph.resolve_current(topic)
                out["current_value"] = current
                print("\n--- DECISION TIMELINE (from knowledge graph) ---")
                for row in timeline:
                    print(f"  {row['value']}  (#{row['channel']} @ {row['ts']})")
                print(f"  => current: {current}")
                print("--- END TIMELINE ---\n")
        with open("demo_output.json", "w") as f:
            json.dump(out, f, indent=2)
        _trace("done", "wrote demo_output.json")
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
