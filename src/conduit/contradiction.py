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
# The money alternative ends with a trailing ``(?![A-Za-z])`` guard and NO ``\s?`` before the
# K/M/B suffix, so "$49 monthly" is NOT mis-parsed as "$49m" ($49 million) by swallowing the
# next word's first letter — while an adjacent suffix like "$2M"/"$49m" is still recognised.
_VALUE_RE = re.compile(
    r"(?P<money>[$€£]\s?\d[\d,]*(?:\.\d+)?[KkMmBb]?)(?![A-Za-z])"
    r"|(?P<pct>\d+(?:\.\d+)?\s?%)"
    r"|(?P<num>\b\d[\d,]*(?:\.\d+)?\b)"
)

# Words that flip a statement's polarity (a decision being reversed/cancelled).
_NEGATION = ("not", "no", "never", "cancel", "cancelled", "canceled", "drop",
             "dropped", "revert", "reverted", "reverse", "reversed", "abandon",
             "scrap", "scrapped", "instead", "changed", "switch", "switched",
             "raised", "raise", "lowered", "lower", "increased", "decreased",
             "bumped", "moved", "updated", "revised")

# Whole-word negation/change matcher — used to gate bare-count/percentage "reversals" so an
# ordinary "planned 3 / onboarded 2" pair isn't fabricated into a decision reversal.
_NEGATION_RE = re.compile(r"\b(?:" + "|".join(re.escape(w) for w in _NEGATION) + r")\b", re.I)

# A value that immediately follows one of these cues is a HISTORICAL reference ("up from $50",
# "changed from $29"), not the current value — used so "current is $20, up from $50" resolves to
# $20, not $50. Anchored at end-of-prefix so it only fires right before the value token.
_FROM_CUE_RE = re.compile(
    r"\b(?:up|down|back|previously|originally|was)?\s*from\s+"
    r"(?:the|a|an|our|their|about|around|roughly)?\s*$",
    re.I,
)


def _canon_value(tok: str) -> str:
    """Canonicalise a value token's magnitude for comparison so equal amounts written two ways
    compare equal: drop an all-zero / trailing-zero decimal fraction ("$49.00" -> "$49",
    "20.50%" -> "20.5%"). Thousands commas and the currency/percent glyphs are preserved for
    display. Without this, a confirmation that restates a price with cents ("$49" then "$49.00")
    is mis-read as a pricing reversal."""
    m = re.search(r"\.(\d+)", tok)
    if not m:
        return tok
    frac = m.group(1).rstrip("0")
    repl = ("." + frac) if frac else ""
    return tok[: m.start()] + repl + tok[m.end():]


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
        tok = _canon_value(tok)  # "$49.00" -> "$49" so restated-with-cents isn't a false reversal
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


def _same_class_values_in_order(text: str, track_cls: str) -> list[tuple[int, str]]:
    """``(start_offset, canonical_token)`` for every value of ``track_cls`` in ``text``, in order."""
    out: list[tuple[int, str]] = []
    for m in _VALUE_RE.finditer(text):
        if (m.lastgroup or "num") != track_cls:
            continue
        tok = re.sub(r"([$€£])\s+", r"\1", m.group(0).strip()).replace(" ", "").rstrip(",.")
        tok = _canon_value(tok)
        if tok:
            out.append((m.start(), tok))
    return out


def _pick_current_value(text: str, track_cls: str, fallback: str) -> str:
    """The value in ``text`` that states the CURRENT amount, skipping values introduced by a
    "from" / "up from" / "previously" cue (historical references). Prefers the last
    non-historical value; falls back to ``fallback`` (the last same-class token) when every value
    is cued or none is found — so "current is $20, up from $50" -> $20 while
    "changed from $29 to $49" -> $49 and "reverted from $20 back to $10" -> $10.
    """
    spans = _same_class_values_in_order(text, track_cls)
    if not spans:
        return fallback
    non_hist = [tok for start, tok in spans if not _FROM_CUE_RE.search(text[:start])]
    return non_hist[-1] if non_hist else spans[-1][1]


def _stem(word: str) -> str:
    """Cheap stem: lowercase, first 4 chars. Retained for callers that want loose grouping."""
    return word.lower()[:4]


def _norm(word: str) -> str:
    """Whole-word normaliser for topic matching: lowercase + drop a trailing plural 's'. This
    matches 'engineer'/'engineers' and 'pricing'/'pricing' but — unlike a 4-char stem — does
    NOT collide 'required' with 'requests' or 'company' with 'competitors'."""
    return word.lower().rstrip("s")


def _light_stem(word: str) -> str:
    """Light inflectional stemmer for topic matching — unifies a word's common forms so a
    question and the evidence match even when their wording differs slightly.

    Strips one inflectional suffix (``-ing`` / ``-ed`` / ``-es`` / plural ``-s``) then a
    resulting silent ``-e``, so ``price``/``pricing``/``priced``/``prices`` all collapse to
    ``pric`` and ``hire``/``hiring`` to ``hir``. Length guards + the trailing-``e`` step keep it
    from the crude 4-char-stem collisions ``_norm`` was written to avoid: ``required``→``requir``
    vs ``requests``→``request``, ``company``→``company`` vs ``competitors``→``competitor``,
    ``policy``→``policy`` vs ``police``→``polic`` all stay distinct.
    """
    w = word.lower()
    if w.endswith("ing") and len(w) > 5:
        w = w[:-3]
    elif w.endswith("ed") and len(w) > 4:
        w = w[:-2]
    elif w.endswith("es") and len(w) > 4:
        w = w[:-2]
    elif w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        w = w[:-1]
    if w.endswith("e") and len(w) > 3:
        w = w[:-1]
    return w


def _text_words(text: str) -> set[str]:
    return {_norm(w) for w in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", text or "")}


def _text_stem_set(text: str) -> set[str]:
    """Light-stemmed content words of ``text`` — the matching space for topic relevance."""
    return {_light_stem(w) for w in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", text or "")}


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
    kw_words = {_light_stem(k) for k in kws}
    out: list[tuple[Any, list[tuple[str, str]]]] = []
    for ev in evidence:
        text = getattr(ev, "text", "") or ""
        if kw_words and not (kw_words & _text_stem_set(text)):
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

    # Walk newest→oldest for the latest message asserting a value of the SAME class. Within that
    # message pick the CURRENT value cue-aware (see _pick_current_value): the last value that is
    # NOT introduced by a "from"/"up from"/"previously" cue — so "changed from $10 to $20" -> $20,
    # "reverted from $20 back to $10" -> $10, and "current is $20, up from the $50" -> $20 (not $50).
    current_val: Optional[str] = None
    newer_ev: Any = None
    for ev, typed in reversed(ordered):
        same_class = [v for c, v in typed if c == track_cls]
        if same_class:
            current_val = _pick_current_value(getattr(ev, "text", "") or "", track_cls, same_class[-1])
            newer_ev = ev
            break

    if current_val is None or current_val == first_val:
        return None  # no genuine change

    # Topic-anchor: the older and newer messages must share a QUESTION topic term (by light
    # stem), not merely a value class — otherwise "pricing $10" (ts1) and "marketing budget $99"
    # (ts2) would fabricate a false $10→$99 reversal across two unrelated decisions. Single-topic
    # questions pass automatically: _collect_valued already requires each kept message to contain
    # that one term, so both older and newer share it.
    kw_stems = {_light_stem(k) for k in kws}
    if kw_stems and not (
        _text_stem_set(getattr(older_ev, "text", ""))
        & _text_stem_set(getattr(newer_ev, "text", ""))
        & kw_stems
    ):
        return None

    # Bare counts and percentages change constantly for non-decision reasons (planned "3 engineers"
    # vs onboarded "2 engineers"; "99.9%" target vs "97%" measured), so only surface a num/pct change
    # as a genuine reversal when a message actually signals a change, or the older/newer messages
    # share ≥2 question topic stems. Money is left UNGATED — a price change is the money-shot.
    if track_cls in ("num", "pct"):
        older_text = getattr(older_ev, "text", "") or ""
        newer_text = getattr(newer_ev, "text", "") or ""
        has_change = bool(_NEGATION_RE.search(older_text) or _NEGATION_RE.search(newer_text))
        shared_topic = _text_stem_set(older_text) & _text_stem_set(newer_text) & kw_stems
        if not has_change and len(shared_topic) < 2:
            return None

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


def _strip_wrong_current_claim(text: str, drift: TimelineDrift) -> str:
    """Remove a sentence that asserts a *current* value of the tracked class different from
    ``drift.current_value`` — so the deterministic current-value statement we append never sits
    next to a contradictory claim the local model already wrote (e.g. model says "current price
    is $15" while the resolved current is $20)."""
    cur_typed = extract_typed_values(drift.current_value)
    cur_class = cur_typed[0][0] if cur_typed else None
    if not cur_class:
        return text
    kept: list[str] = []
    for s in re.split(r"(?<=[.!?])\s+", text):
        if "current" in s.lower():
            same_class_vals = [v for c, v in extract_typed_values(s) if c == cur_class]
            if same_class_vals and drift.current_value not in same_class_vals:
                continue  # a wrong 'current' claim → drop it rather than contradict it
        kept.append(s)
    return " ".join(kept).strip()


def resolve_answer_text(text: str, drift: Optional[TimelineDrift]) -> str:
    """Guarantee the answer states BOTH values and the current one (deterministic).

    Runs regardless of what the model wrote, so the money-shot never depends on the
    local model's phrasing. Idempotent: won't double-append if already present. First strips any
    contradictory 'current value' claim the model wrote, so the appended truth stands alone.
    """
    if not drift:
        return text
    text = _strip_wrong_current_claim(text, drift)
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
