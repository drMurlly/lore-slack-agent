"""Knowledge-graph core for Lore — turns evidence into a queryable decision graph.

Why this module exists
----------------------
Lore's differentiator is grounding reasoning in a **graph** of entities + typed edges,
not regex on prose. This module builds an ephemeral, per-query knowledge graph from
retrieved evidence: entities (topics, values, people) + typed edges (decided / changed /
mentioned / supersedes), each grounded to a source message.

The graph enables deterministic timeline queries, contradiction detection, and a
"decision graph" badge for the Canvas — all without any database or external service.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from conduit.research import Evidence

# Import from contradiction module for text analysis
from conduit.contradiction import (
    _NEGATION, _CLASS_PRIORITY, _keywords, _stem,
    extract_values, extract_typed_values, detect_drift, TimelineDrift,
)

# Word-boundary negation matcher — a substring test flags "now"/"know" (contain "no") and
# mislabels a plain decision as a reversal. Match whole words only.
_NEGATION_RE = re.compile(r"\b(?:" + "|".join(re.escape(w) for w in _NEGATION) + r")\b", re.I)


def _ts_float(ts: Any) -> float:
    """Parse a Slack ``ts`` string to float for correct chronological ordering.

    String comparison is wrong here: ``'999' > '1000'`` lexicographically, which would invert
    the timeline and the supersedes chain.
    """
    try:
        return float(ts or 0)
    except (TypeError, ValueError):
        return 0.0


@dataclass
class Entity:
    """A node in the knowledge graph."""
    id: str            # normalised key, e.g. "topic:pricing" / "value:$20" / "person:alice"
    type: str          # "topic" | "value" | "person"
    label: str
    first_ts: str      # earliest ts this entity was seen
    permalinks: list[str] = field(default_factory=list)

    def add_permalink(self, permalink: str) -> None:
        """Add a permalink if not already present."""
        if permalink and permalink not in self.permalinks:
            self.permalinks.append(permalink)


@dataclass
class Edge:
    """A typed edge between two entities."""
    subject: str       # Entity.id
    predicate: str     # "decided" | "changed" | "mentioned" | "supersedes"
    obj: str           # Entity.id
    ts: str
    permalink: str
    text: str          # the source message text (for citation quotes)
    channel: str = ""  # source channel (so a graph-grounded drift can name #where)
    value_class: str = ""  # money|pct|num — so timelines never compare a price to a stray count


class KnowledgeGraph:
    """In-memory knowledge graph built from evidence."""

    def __init__(self) -> None:
        self.entities: dict[str, Entity] = {}
        self.edges: list[Edge] = []

    def _get_or_create_entity(self, entity_id: str, entity_type: str, label: str,
                               ts: str, permalink: str) -> Entity:
        """Get existing entity or create new one with first_ts tracking."""
        if entity_id in self.entities:
            entity = self.entities[entity_id]
            entity.add_permalink(permalink)
            # Update first_ts if this is earlier
            try:
                if float(ts) < float(entity.first_ts):
                    entity.first_ts = ts
            except (TypeError, ValueError):
                pass
            return entity
        else:
            entity = Entity(
                id=entity_id,
                type=entity_type,
                label=label,
                first_ts=ts,
                permalinks=[permalink] if permalink else []
            )
            self.entities[entity_id] = entity
            return entity

    def _extract_mentions(self, text: str) -> list[str]:
        """Extract @mentions from text."""
        return re.findall(r"@(\w+)", text)

    def _has_negation(self, text: str) -> bool:
        """Check if text contains a negation/change word (whole-word match)."""
        return bool(_NEGATION_RE.search(text or ""))

    def add_evidence(self, ev: Any, question_kws: Optional[set[str]] = None) -> None:
        """Extract entities + edges from ONE Evidence and merge into the graph.

        Deterministic: topic = content keywords (contradiction._keywords over the text with
        @mentions stripped — a mention is a *person*, never a topic); values =
        contradiction.extract_values(text); @mentions -> person entities. Emit a
        (topic)-[decided|changed]->(value) edge per value, predicate 'changed' when a
        negation/change word (contradiction._NEGATION) is present else 'decided'.

        ``question_kws`` (from :func:`build_graph`) anchors every value in a query to the
        SAME topic so the timeline/supersedes chain forms — without it, differing per-message
        keyword sets would scatter values across topics and hide the reversal.
        """
        text = getattr(ev, "text", "") or ""
        ts = getattr(ev, "ts", "0") or "0"
        permalink = getattr(ev, "permalink", "") or ""
        channel = getattr(ev, "channel", "") or ""

        if not text or not ts:
            return

        # Topic keywords come from the CONTENT only — strip @mentions first so people
        # (alice/bob) never masquerade as topics and split a value chain nondeterministically.
        text_for_topics = re.sub(r"@\w+", " ", text)
        kws = _keywords(None, [text_for_topics])
        for kw in sorted(kws):
            topic_id = f"topic:{kw}"
            self._get_or_create_entity(topic_id, "topic", kw, ts, permalink)

        # Extract value entities — carry the value CLASS (money/pct/num) so a stray count
        # in a pricing message never joins the price timeline.
        typed_values = extract_typed_values(text)
        for value_class, val in typed_values:
            value_id = f"value:{val}"
            self._get_or_create_entity(value_id, "value", val, ts, permalink)

            # Determine predicate: changed if negation word present, else decided
            predicate = "changed" if self._has_negation(text) else "decided"

            # Choose the topic this value attaches to (deterministic):
            #   1. a question keyword whose STEM matches one in this message (so "price" and
            #      "pricing" land on ONE topic node and the value chain doesn't split), else
            #   2. the alphabetically-first content keyword, else
            #   3. a channel/unknown fallback.
            topic_id = None
            q_stems = {_stem(q): q for q in sorted(question_kws or set())}
            anchor = sorted(kw for kw in kws if _stem(kw) in q_stems)
            if anchor:
                topic_id = f"topic:{anchor[0]}"
                self._get_or_create_entity(topic_id, "topic", anchor[0], ts, permalink)
            else:
                for kw in sorted(kws):
                    tid = f"topic:{kw}"
                    if tid in self.entities:
                        topic_id = tid
                        break

            # If no topic found, create a generic one from channel or "unknown"
            if not topic_id:
                topic_id = f"topic:{channel}" if channel else "topic:unknown"
                self._get_or_create_entity(topic_id, "topic", channel or "unknown", ts, permalink)

            # Add edge from topic to value
            edge = Edge(
                subject=topic_id,
                predicate=predicate,
                obj=value_id,
                ts=ts,
                permalink=permalink,
                text=text,
                channel=channel,
                value_class=value_class,
            )
            self.edges.append(edge)

        # Extract person entities from @mentions
        mentions = self._extract_mentions(text)
        for mention in mentions:
            person_id = f"person:{mention}"
            person = self._get_or_create_entity(person_id, "person", mention, ts, permalink)

            # Connect person to any topic in the text
            for kw in kws:
                topic_id = f"topic:{kw}"
                if topic_id in self.entities:
                    edge = Edge(
                        subject=topic_id,
                        predicate="mentioned",
                        obj=person_id,
                        ts=ts,
                        permalink=permalink,
                        text=text
                    )
                    self.edges.append(edge)

    def _topic_class(self, topic_id: str) -> str:
        """The primary value class for a topic (money > pct > num), so a topic's timeline and
        supersedes chain only ever compare like with like."""
        present = {
            e.value_class for e in self.edges
            if e.subject == topic_id and e.predicate in ("decided", "changed") and e.value_class
        }
        return next((c for c in _CLASS_PRIORITY if c in present), "")

    def build_supersedes(self) -> None:
        """For each topic with ≥2 distinct decided/changed values across time, add a
        (newer_value)-[supersedes]->(older_value) edge ordered by ts. Idempotent.

        Only values of the topic's primary class participate — a stray count ("3 seats") in a
        pricing message never counts as a reversal of "$20"."""
        # Group edges by topic (primary-class values only)
        topic_values: dict[str, list[tuple[str, str, int]]] = {}  # topic_id -> [(ts, value_id, edge_idx)]

        topic_class: dict[str, str] = {}
        for i, edge in enumerate(self.edges):
            if edge.predicate in ("decided", "changed"):
                topic_id = edge.subject
                if topic_id not in topic_class:
                    topic_class[topic_id] = self._topic_class(topic_id)
                cls = topic_class[topic_id]
                if cls and edge.value_class and edge.value_class != cls:
                    continue  # off-class value — not part of this topic's decision chain
                topic_values.setdefault(topic_id, []).append((edge.ts, edge.obj, i))

        # For each topic, find value transitions and add supersedes edges
        for topic_id, value_list in topic_values.items():
            # Sort chronologically by ts (float — string sort inverts '999' vs '1000')
            value_list.sort(key=lambda x: _ts_float(x[0]))

            # Track unique values in order
            seen_values: list[str] = []
            for ts, value_id, edge_idx in value_list:
                if value_id not in seen_values:
                    seen_values.append(value_id)

            # Add supersedes edges: newer supersedes older
            for i in range(1, len(seen_values)):
                newer_value = seen_values[i]
                older_value = seen_values[i - 1]

                # Find the newer edge's ts and permalink
                newer_ts = "0"
                newer_permalink = ""
                for ts, value_id, edge_idx in value_list:
                    if value_id == newer_value:
                        newer_ts = ts
                        if edge_idx < len(self.edges):
                            newer_permalink = self.edges[edge_idx].permalink
                        break

                # Check if supersedes edge already exists
                existing = any(
                    e.subject == newer_value and e.obj == older_value and e.predicate == "supersedes"
                    for e in self.edges
                )

                if not existing:
                    edge = Edge(
                        subject=newer_value,
                        predicate="supersedes",
                        obj=older_value,
                        ts=newer_ts,
                        permalink=newer_permalink,
                        text=""
                    )
                    self.edges.append(edge)

    def neighbors(self, entity_id: str) -> list[Entity]:
        """All entities directly connected to entity_id (either edge direction)."""
        neighbor_ids: set[str] = set()

        for edge in self.edges:
            if edge.subject == entity_id:
                neighbor_ids.add(edge.obj)
            elif edge.obj == entity_id:
                neighbor_ids.add(edge.subject)

        return [self.entities[nid] for nid in neighbor_ids if nid in self.entities]

    def timeline(self, topic_id: str) -> list[Edge]:
        """decided/changed edges for a topic, sorted oldest→newest by ts."""
        topic_edges = [
            e for e in self.edges
            if e.subject == topic_id and e.predicate in ("decided", "changed")
        ]
        return sorted(topic_edges, key=lambda e: _ts_float(e.ts))

    def resolve_current(self, topic_id: str) -> tuple[Optional[str], Optional[Edge]]:
        """Return (current_value_label, Edge) = the newest decided/changed edge for the
        topic, or (None, None) if unknown."""
        timeline_edges = self.timeline(topic_id)
        if not timeline_edges:
            return None, None

        # Get the newest edge
        newest_edge = timeline_edges[-1]

        # Get the value entity label
        if newest_edge.obj in self.entities:
            value_entity = self.entities[newest_edge.obj]
            return value_entity.label, newest_edge

        return None, None

    def primary_topic(self, question: Optional[str] = None) -> Optional[str]:
        """The topic entity that best matches the question and has a value timeline.

        Matches a question keyword by stem, preferring the topic with the longest timeline
        (the most-decided entity). With no question, returns the topic with the most
        decided/changed edges."""
        from conduit.contradiction import _keywords, _stem
        kw_stems = {_stem(k) for k in _keywords(question, None)} if question else set()
        best: Optional[str] = None
        best_len = 0
        for eid, ent in self.entities.items():
            if ent.type != "topic":
                continue
            tl = self.timeline(eid)
            if not tl:
                continue
            if kw_stems and _stem(ent.label) not in kw_stems:
                continue
            if len(tl) > best_len:
                best, best_len = eid, len(tl)
        return best

    def decision_rows(self, question: Optional[str] = None) -> list[dict[str, str]]:
        """Class-anchored, consecutive-dedup'd decision rows for the question's primary topic,
        oldest→newest. Each row: ``{value, channel, ts, permalink}``.

        Only values of the topic's primary class (money > pct > num) are kept, and consecutive
        repeats of the same value are collapsed — so the rendered timeline and the resolved
        current value never include a stray number or a fake no-change step."""
        topic = self.primary_topic(question)
        if not topic:
            return []
        track = self._topic_class(topic)
        rows: list[dict[str, str]] = []
        last_val: Optional[str] = None
        for e in self.timeline(topic):  # decided/changed edges, oldest→newest (float ts)
            if track and e.value_class and e.value_class != track:
                continue
            ent = self.entities.get(e.obj)
            if ent is None or ent.label == last_val:
                continue
            rows.append({"value": ent.label, "channel": e.channel or "",
                         "ts": e.ts, "permalink": e.permalink})
            last_val = ent.label
        return rows

    def drift_for_question(self, question: Optional[str] = None) -> Optional[TimelineDrift]:
        """A :class:`TimelineDrift` grounded in the graph's decision chain for the question's
        primary topic — the graph as the single source of truth for the current value. Returns
        ``None`` when the topic has no genuine oldest→newest value change (same class only)."""
        from types import SimpleNamespace
        rows = self.decision_rows(question)
        if len({r["value"] for r in rows}) < 2:
            return None
        oldest, newest = rows[0], rows[-1]
        if oldest["value"] == newest["value"]:
            return None  # net no-change (e.g. reverted back to the original)
        older = SimpleNamespace(channel=oldest["channel"], permalink=oldest["permalink"])
        newer = SimpleNamespace(channel=newest["channel"], permalink=newest["permalink"])
        summary = (
            f"{oldest['value']} (#{oldest['channel'] or '?'}) → "
            f"{newest['value']} (#{newest['channel'] or '?'}) — current: {newest['value']}"
        )
        return TimelineDrift(
            old_value=oldest["value"], new_value=newest["value"], current_value=newest["value"],
            older=older, newer=newer, summary=summary,
        )

    def summary(self) -> dict:
        """{'entities': N, 'topics': n, 'values': n, 'people': n, 'decisions': n,
            'reversals': n(supersedes edges)} — for the Canvas 'decision graph' badge."""
        topics = sum(1 for e in self.entities.values() if e.type == "topic")
        values = sum(1 for e in self.entities.values() if e.type == "value")
        people = sum(1 for e in self.entities.values() if e.type == "person")
        decisions = sum(1 for e in self.edges if e.predicate in ("decided", "changed"))
        reversals = sum(1 for e in self.edges if e.predicate == "supersedes")

        return {
            "entities": len(self.entities),
            "topics": topics,
            "values": values,
            "people": people,
            "decisions": decisions,
            "reversals": reversals,
        }


def build_graph(evidence: list[Any], question: Optional[str] = None) -> KnowledgeGraph:
    """Construct a KnowledgeGraph from Evidence[]; passes `question` so topic extraction is
    focused. Calls add_evidence for each, then build_supersedes(). Returns the graph."""
    graph = KnowledgeGraph()

    question_kws = _keywords(question, None) if question else set()
    for ev in evidence:
        graph.add_evidence(ev, question_kws=question_kws)

    graph.build_supersedes()

    return graph


def graph_badge_from_summary(summary: dict) -> str:
    """A one-line 'Decision Graph' badge for the Canvas — visible proof of deep research.

    Renders e.g. '🕸️ Decision Graph: 6 entities · 2 people · 2 decisions · 1 reversal' from
    a ``graph.summary()`` dict. The reversal count is the money-shot: a search wrapper can't
    show it.
    """
    parts = [f"{summary.get('entities', 0)} entities", f"{summary.get('decisions', 0)} decisions"]
    if summary.get("people"):
        parts.insert(1, f"{summary['people']} people")
    if summary.get("reversals"):
        n = summary["reversals"]
        parts.append(f"{n} reversal" + ("s" if n != 1 else ""))
    return "🕸️ *Decision Graph:* " + " · ".join(parts)


def graph_badge(graph: "KnowledgeGraph") -> str:
    """Convenience: badge line from a live graph (delegates to :func:`graph_badge_from_summary`)."""
    return graph_badge_from_summary(graph.summary())
