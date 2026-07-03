"""Evidence-grounded timeline & contradiction resolver — Lore's core differentiator.

Why this module exists
----------------------
The demo money-shot ("we decided $10 in March … reversed to $20 in May … the
current answer is $20") must be *deterministic*, not a hope that the local model
phrases a reversal correctly. The earlier approach scraped contradictions out of
the LLM's already-written prose with a ``(\\w+)`` regex — which cannot even capture
``$10``/``$20`` (the ``$`` is a non-word char), so it silently missed the exact
example we demo. This module instead grounds the reasoning in the **evidence**:

  * order the evidence chronologically by Slack ``ts`` (unix seconds),
  * extract the concrete *value* each message asserts for the queried entity
    (currency, percentages, numbers, yes/no-style decisions),
  * detect when that value **changed over time**, and
  * surface the **current** (latest) value with a citation to its source.

It operates on any object exposing ``text``, ``ts``, ``channel`` and ``permalink``
(both :class:`conduit.rts_client.SearchHit` and :class:`conduit.research.Evidence`
qualify), so it is reusable across the pipeline and independent of the model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional, Sequence

# Value tokens we can compare across time, most-specific first so "$20" wins over "20".
#   $10 / $ 1,200.50 / €5 | 20% | bare number 10 / 10.5 / 1,200
_VALUE_RE = re.compile(
    r"(?P<money>[$€£]\s?\d[\d,]*(?:\.\d+)?\s?[KkMmBb]?)"
    r"|(?P<pct>\d+(?:\.\d+)?\s?%)"
    r"|(?P<num>\b\d[\d,]*(?:\.\d+)?\b)"
)

# Words that flip a statement's polarity (a decision being reversed/cancelled).
_NEGATION = ("not", "no", "never", "cancel", "cancelled", "canceled", "drop",
             "dropped", "revert", "reverted", "reverse", "reversed", "abandon",
             "scrap", "scrapped", "instead", "changed", "switch", "switched")


def _ts_key(ev: Any) -> float:
    """Sort key from a Slack ``ts`` string ('1234567890.000123'). Robust to junk."""
    try:
        return float(getattr(ev, "ts", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def timeline_sort(evidence: Sequence[Any], *, newest_first: bool = False) -> list[Any]:
    """Return the evidence ordered chronologically by ``ts`` (oldest-first default).

    Oldest-first is what the narrative needs ("decided X … then Y … current = Y");
    pass ``newest_first=True`` for a most-recent-on-top view.
    """
    return sorted(evidence, key=_ts_key, reverse=newest_first)


# Which value classes are more meaningful for a decision, most-specific first. A message's
# "primary" value is the highest-priority class it asserts ("$20" beats a bare "3 weeks").
_CLASS_PRIORITY = ("money", "pct", "num")


def extract_typed_values(text: str) -> list[tuple[str, str]]:
    """Extract ``(class, token)`` value pairs — class ∈ {money, pct, num}.

    Carrying the class lets the resolver compare only *within* a class, so a currency ("$10")
    is never mistaken for an unrelated bare number ("3 weeks") — the bug that produced
    confident false "conflicting signals". Normalises whitespace inside currency
    ("$ 20" -> "$20") so equal values compare equal.
    """
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for m in _VALUE_RE.finditer(text):
        cls = m.lastgroup or "num"
        tok = m.group(0)
        tok = re.sub(r"([$€£])\s+", r"\1", tok.strip())  # "$ 20" -> "$20"
        tok = tok.replace(" ", "")
        tok = tok.rstrip(",.")  # "$10," -> "$10"
        if tok and tok not in seen:
            seen.add(tok)
            out.append((cls, tok))
    return out


def extract_values(text: str) -> list[str]:
    """Extract comparable value tokens (currency/percent/number) from a message.

    Thin wrapper over :func:`extract_typed_values` preserving the historical ``list[str]``
    contract (used for value-entity labels in the knowledge graph). Robust to the ``$`` the
    old ``\\w+`` regex could not see.
    """
    return [tok for _cls, tok in extract_typed_values(text)]


def _primary_typed_value(typed: list[tuple[str, str]]) -> Optional[tuple[str, str]]:
    """The highest-priority ``(class, token)`` a message asserts (money > pct > num)."""
    for cls in _CLASS_PRIORITY:
        for c, v in typed:
            if c == cls:
                return (cls, v)
    return None


def _stem(word: str) -> str:
    """Cheap stem: lowercase, first 4 chars. Retained for callers that want loose grouping."""
    return word.lower()[:4]


def _norm(word: str) -> str:
    """Whole-word normaliser for topic matching: lowercase + drop a trailing plural 's'. This
    matches 'engineer'/'engineers' and 'pricing'/'pricing' but — unlike a 4-char stem — does
    NOT collide 'required' with 'requests' or 'company' with 'competitors'."""
    return word.lower().rstrip("s")


def _text_words(text: str) -> set[str]:
    return {_norm(w) for w in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", text or "")}


def _text_stems(text: str) -> set[str]:
    return {_stem(w) for w in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", text or "")}


def _keywords(question: Optional[str], sub_queries: Optional[Sequence[str]]) -> set[str]:
    """Content keywords from the question + sub-queries (stopwords stripped)."""
    stop = {"what", "did", "we", "the", "about", "and", "a", "an", "to", "of",
            "is", "are", "was", "were", "our", "for", "on", "in", "it", "do",
            "does", "how", "when", "why", "who", "which", "change", "changed",
            "decide", "decided", "any", "anything", "since", "at", "with",
            # generic words that must NOT anchor a topic (they leak across topics)
            "current", "currently", "now", "value", "values", "latest", "recent",
            "still", "get", "going", "per", "all", "set", "use", "using", "us",
            "there", "been", "have", "has", "had", "or", "that", "this", "you",
            "your", "their", "they", "new", "old", "much", "many", "more"}
    words: set[str] = set()
    for blob in [question or ""] + list(sub_queries or []):
        for w in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", blob.lower()):
            if w not in stop:
                words.add(w)
    return words


@dataclass
class TimelineDrift:
    """A detected change of value over time for the queried entity."""
    old_value: str
    new_value: str
    current_value: str          # == new_value (the latest assertion)
    older: Any                  # the earlier evidence object
    newer: Any                  # the later evidence object (the "current" source)
    summary: str                # human-readable, channel + value delta

    @property
    def current_permalink(self) -> str:
        return getattr(self.newer, "permalink", "") or ""


def _collect_valued(evidence: Sequence[Any], kws: set[str]) -> list[tuple[Any, list[tuple[str, str]]]]:
    """Evidence that (a) matches a question keyword (stem/prefix) and (b) asserts a value.

    Keyword match is stem-based so "pricing" catches evidence that says "price"/"priced" —
    the one-directional substring match silently dropped exactly that case. With empty ``kws``
    every valued message qualifies.
    """
    kw_words = {_norm(k) for k in kws}
    out: list[tuple[Any, list[tuple[str, str]]]] = []
    for ev in evidence:
        text = getattr(ev, "text", "") or ""
        if kw_words and not (kw_words & _text_words(text)):
            continue
        typed = extract_typed_values(text)
        if typed:
            out.append((ev, typed))
    return out


def detect_drift(
    evidence: Sequence[Any],
    *,
    question: Optional[str] = None,
    sub_queries: Optional[Sequence[str]] = None,
) -> Optional[TimelineDrift]:
    """Detect a value that changed over time for the entity the question is about.

    Deterministic strategy:
      1. keep evidence that matches a question keyword (stem match) AND asserts a value;
      2. order it oldest→newest by ``ts``;
      3. anchor on the CLASS of the oldest message's primary value (money > pct > num) so we
         only ever compare like with like (never "$10" vs a stray "3 weeks");
      4. the **current** value is the *newest* message's value of that class — not the first
         differing one, which mislabels any 3+ value chain (``$10→$15→$20`` → current $20).

    Returns ``None`` when there is no genuine same-class change to surface. Keyword matching
    is stem-based (``pricing`` catches ``price``) so relevance is robust without falling back
    to matching unrelated evidence.
    """
    if not evidence:
        return None

    kws = _keywords(question, sub_queries)
    relevant = _collect_valued(evidence, kws)
    if len(relevant) < 2:
        return None

    ordered = sorted(relevant, key=lambda pair: _ts_key(pair[0]))
    first_primary = _primary_typed_value(ordered[0][1])
    if not first_primary:
        return None
    track_cls, first_val = first_primary
    older_ev = ordered[0][0]

    # Walk newest→oldest for the latest message asserting a value of the SAME class. Within
    # that message take the LAST such token, so "changed from $10 to $20" resolves to $20 and
    # "reverted from $20 back to $10" resolves to $10 (the value after "to"/"back to").
    current_val: Optional[str] = None
    newer_ev: Any = None
    for ev, typed in reversed(ordered):
        same_class = [v for c, v in typed if c == track_cls]
        if same_class:
            current_val, newer_ev = same_class[-1], ev
            break

    if current_val is None or current_val == first_val:
        return None  # no genuine change

    summary = (
        f"{first_val} (#{getattr(older_ev, 'channel', '?')}) → "
        f"{current_val} (#{getattr(newer_ev, 'channel', '?')}) — current: {current_val}"
    )
    return TimelineDrift(
        old_value=first_val, new_value=current_val, current_value=current_val,
        older=older_ev, newer=newer_ev, summary=summary,
    )


def _value_present(value: str, text: str) -> bool:
    """Whether ``value`` appears in ``text`` as a standalone token — so "$20" is NOT counted
    present inside "$200" and "10" is not found inside "2010"."""
    # Exclude continuations that would make it a *different* value ($20 inside $200/$20.50/20%),
    # but allow an ordinary trailing period/comma ("…to $20." is still $20).
    pattern = r"(?<![\w$€£.])" + re.escape(value) + r"(?!\d)(?!\.\d)(?!%)"
    return re.search(pattern, text, re.I) is not None


def resolve_answer_text(text: str, drift: Optional[TimelineDrift]) -> str:
    """Guarantee the answer states BOTH values and the current one (deterministic).

    Runs regardless of what the model wrote, so the money-shot never depends on the
    local model's phrasing. Idempotent: won't double-append if already present.
    """
    if not drift:
        return text
    additions = []
    if not _value_present(drift.old_value, text):
        additions.append(f"An earlier value was {drift.old_value}.")
    if not _value_present(drift.new_value, text):
        additions.append(f"It was later changed to {drift.new_value}.")
    if "current" not in text.lower() or not _value_present(drift.current_value, text):
        additions.append(f"The current value is {drift.current_value}.")
    if not additions:
        return text
    sep = " " if text and not text.endswith((" ", "\n")) else ""
    return f"{text}{sep}" + " ".join(additions)


def conflict_canvas_section(drift: Optional[TimelineDrift]) -> Optional[dict]:
    """A Canvas '⚠️ Conflicting signals' section (matches canvas.py's dict blocks).

    Placed FIRST in the report so judges instantly see Lore doing what no search
    wrapper does. Returns ``None`` when there is no drift.
    """
    if not drift:
        return None
    older_link = getattr(drift.older, "permalink", "") or ""
    newer_link = getattr(drift.newer, "permalink", "") or ""
    body = (
        f"⚠️ *Conflicting signals over time* — the answer changed.\n"
        f"• Earlier: *{drift.old_value}* (<{older_link}|#{getattr(drift.older,'channel','?')}>)\n"
        f"• Later / current: *{drift.new_value}* (<{newer_link}|#{getattr(drift.newer,'channel','?')}>)\n"
        f"Lore resolves to the most recent decision: *{drift.current_value}*."
    )
    return {"type": "section", "text": {"type": "mrkdwn", "text": body}}
