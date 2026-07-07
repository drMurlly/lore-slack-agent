"""Multi-hop research loop for gathering evidence from RTS."""
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from conduit.rts_client import SearchHit
from conduit.agent import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class Evidence:
    """A piece of evidence gathered from RTS search.
    
    Attributes:
        text: The text content of the evidence.
        channel: The channel where this evidence was found.
        ts: The timestamp of the message.
        permalink: A stable URL to reference this evidence.
        score: The relevance score from the search.
        author: Optional author of the message.
        citation_index: Stable 1-based index for citation.
        source_hit: The original SearchHit this evidence came from.
    """
    text: str
    channel: str
    ts: str
    permalink: str
    score: float
    author: Optional[str]
    citation_index: int
    source_hit: SearchHit


@dataclass
class ResearchResult:
    """Result of a multi-hop research query.
    
    Attributes:
        question: The original question asked.
        evidence: Ordered list of Evidence items gathered.
        follow_up_hops: Number of follow-up hops performed.
        graph: Optional knowledge graph built over the evidence.
        glossary: Org-glossary definitions for domain terms found in the
            question, resolved via the MCP glossary server (empty when the
            consult is disabled or no terms matched). Each entry is a dict
            like ``{"term": "ARR", "definition": "Annual Recurring Revenue…"}``.
    """
    question: str
    evidence: list[Evidence]
    follow_up_hops: int = 0
    graph: Any = None
    glossary: list = field(default_factory=list)


def _evidence_key(hit: Any) -> str:
    """Stable identity for evidence dedup — permalink when present, else ``channel:ts``.

    Keying on permalink alone collapses DISTINCT messages whenever a backend omits the permalink
    (``RTSClient`` sets it to ``""`` when ``search.messages`` doesn't return one): two different
    decisions both keyed to ``""`` merge into one, silently dropping evidence — and potentially
    hiding the exact reversal Lore exists to surface."""
    permalink = getattr(hit, "permalink", "") or ""
    if permalink:
        return permalink
    return f"{getattr(hit, 'channel', '')}:{getattr(hit, 'ts', '')}"


def _decompose_question(question: str, llm: LLMClient) -> list[str]:
    """Decompose a question into sub-queries using the LLM.
    
    Args:
        question: The original question to decompose..
        llm: The LLM client to use for decomposition.
        
    Returns:
        A list of sub-query strings.
    """
    messages = [
        {
            "role": "system",
            "content": "You are a research assistant. Decompose the user's question into 2-3 specific sub-queries that would help answer it. Return each sub-query on a new line. Do not add any other text."
        },
        {
            "role": "user",
            "content": question
        }
    ]
    
    response = llm.chat(messages)
    content = response.get("content") or ""

    # Parse sub-queries from the response, sanitizing LLM formatting:
    # strip list markers ("1." / "2)" / "-" / "*"), drop preamble lines
    # ending with ":", drop empties, and cap at 4 sub-queries.
    sub_queries: list[str] = []
    for line in content.split("\n"):
        line = re.sub(r"^\s*(?:\d+[.)]|[-*])\s*", "", line).strip()
        if not line or line.endswith(":"):
            continue
        sub_queries.append(line)
    sub_queries = sub_queries[:4]

    # Default to the original question if decomposition fails
    if not sub_queries:
        sub_queries = [question]

    return sub_queries


def _gather_evidence(
    sub_queries: list[str],
    rts: Any,
    limit_per_query: int = 5,
    assistant: Optional[Any] = None,
) -> tuple[list[Evidence], int]:
    """Gather evidence from RTS for each sub-query.

    Args:
        sub_queries: List of sub-queries to search for.
        rts: The RTS client to search.
        limit_per_query: Maximum results per sub-query.
        assistant: Optional streaming surface; a per-search trace step is emitted so
            the user watches "Searching '<q>' → N hits" live.

    Returns:
        Tuple of (list of Evidence, number of search calls made).
    """
    evidence_by_key: dict[str, Evidence] = {}
    search_calls = 0

    for sub_query in sub_queries:
        try:
            hits = rts.search(sub_query, limit=limit_per_query)
        except Exception as exc:
            # Per-sub-query isolation: one transient search failure (rate_limit, network blip on
            # the official RTS backend) must not discard the evidence already gathered from the
            # other sub-queries. Log, trace, and move on.
            logger.warning("search failed for sub-query %r: %s", sub_query[:80], exc)
            if assistant is not None:
                assistant.emit_trace("search", f"“{sub_query}” → error (skipped)")
            search_calls += 1
            continue
        search_calls += 1

        if assistant is not None:
            channels = sorted({h.channel for h in hits if getattr(h, "channel", "")})
            where = " in #" + ", #".join(channels[:3]) if channels else ""
            assistant.set_status(f"Searching “{sub_query}”…")
            assistant.emit_trace("search", f"“{sub_query}” → {len(hits)} hits{where}")

        for hit in hits:
            key = _evidence_key(hit)
            existing = evidence_by_key.get(key)
            if existing is not None:
                # Keep one Evidence per message, but retain the best score seen.
                existing.score = max(existing.score, hit.score)
            else:
                evidence_by_key[key] = Evidence(
                    text=hit.text,
                    channel=hit.channel,
                    ts=hit.ts,
                    permalink=hit.permalink,
                    score=hit.score,
                    author=hit.author,
                    citation_index=0,  # Will be set later
                    source_hit=hit,
                )

    # Sort by score descending
    evidence_list = list(evidence_by_key.values())
    evidence_list.sort(key=lambda e: e.score, reverse=True)
    
    # Assign citation indices
    for i, evidence in enumerate(evidence_list, start=1):
        evidence.citation_index = i
    
    return evidence_list, search_calls


def _detect_thin_coverage(evidence: list[Evidence], threshold: int = 3) -> bool:
    """Detect if evidence coverage is thin.
    
    Args:
        evidence: List of gathered evidence.
        threshold: Minimum number of unique hits to consider coverage adequate.
        
    Returns:
        True if coverage is thin (below threshold).
    """
    return len(evidence) < threshold


def _generate_follow_up_query(
    original_question: str,
    evidence: list[Evidence],
    llm: LLMClient
) -> str:
    """Generate a follow-up query to address gaps in coverage.
    
    Args:
        original_question: The original question.
        evidence: Current evidence gathered.
        llm: The LLM client to use for generation.
        
    Returns:
        A follow-up query string.
    """
    evidence_texts = "\n".join([f"- {e.text}" for e in evidence[:5]])
    
    messages = [
        {
            "role": "system",
            "content": "You are a research assistant. Based on the original question and the evidence gathered so far, generate ONE follow-up search query to fill gaps in the coverage. Return only the query, no other text."
        },
        {
            "role": "user",
            "content": f"Original question: {original_question}\n\nEvidence gathered:\n{evidence_texts}\n\nGenerate a follow-up search query:"
        }
    ]
    
    response = llm.chat(messages)
    content = (response.get("content") or "").strip()

    return content if content else original_question


def _consult_glossary(
    question: str,
    glossary: Any,
    assistant: Optional[Any] = None,
) -> list:
    """Optionally resolve org/domain terms in the question via the MCP glossary server.

    ``glossary`` controls the consult:
      * ``None`` (default) — auto: consult only if the ``LORE_MCP_GLOSSARY``
        env var is truthy (off in tests, switched on in live deployments).
      * ``False`` — never consult.
      * ``True`` — consult via the default stdio manager (spawns
        ``servers/glossary_server.py`` through the official MCP SDK).
      * an object with ``call_tool`` — use it as the manager (injection).

    Defensive by design: any failure logs a warning and returns ``[]`` so the
    money-shot research path is never slowed down or broken by MCP issues.
    """
    if glossary is None:
        enabled = os.environ.get("LORE_MCP_GLOSSARY", "").strip().lower() in {"1", "true", "yes", "on"}
        if not enabled:
            return []
        glossary = True
    if not glossary:
        return []

    try:
        from conduit.mcp_manager import lookup_glossary_terms

        manager = glossary if hasattr(glossary, "call_tool") else None
        entries = lookup_glossary_terms(question, manager=manager)
    except Exception as exc:  # never let MCP break research
        logger.warning("glossary consult failed: %s", exc)
        return []

    if assistant is not None and entries:
        terms = ", ".join(str(e.get("term", "?")) for e in entries[:5])
        assistant.emit_trace("glossary", f"resolved {len(entries)} term(s) via MCP: {terms}")
    return entries


def _glossary_expansions(entries: list) -> list[str]:
    """Turn resolved glossary entries into extra search queries.

    This is what makes the MCP consult *matter* (not just decorate the result): the canonical
    long-form of each term becomes an additional sub-query, so a question that only mentions an
    acronym ("ARR") still retrieves messages that spell it out ("Annual Recurring Revenue") —
    evidence the raw keyword search would miss. Remove the MCP round-trip and recall on
    acronym/jargon questions measurably drops.
    """
    out: list[str] = []
    for e in entries or []:
        if not isinstance(e, dict):
            continue
        term = str(e.get("term", "")).strip()
        definition = str(e.get("definition", "")).strip()
        if not definition:
            continue
        # The canonical long-form is the phrase before the "— …" gloss (em/en dash or hyphen).
        head = re.split(r"\s[—–-]\s", definition, maxsplit=1)[0].strip()
        expansion = head if head and head.lower() != term.lower() else definition
        if expansion and expansion not in out:
            out.append(expansion)
    return out


def run(
    question: str,
    rts: Any,
    llm: LLMClient,
    follow_up_threshold: int = 3,
    max_follow_ups: int = 1,
    assistant: Optional[Any] = None,
    glossary: Any = None,
) -> ResearchResult:
    """Run a multi-hop research loop.

    Decomposes the question into sub-queries, searches for each, deduplicates
    results, and optionally fires a follow-up hop if coverage is thin.

    Args:
        question: The original question to research.
        rts: The RTS client to search.
        llm: The LLM client for decomposition and follow-up generation.
        follow_up_threshold: Minimum evidence count to avoid follow-up.
        max_follow_ups: Maximum number of follow-up hops allowed.
        assistant: Optional ResearchAssistant for streaming trace updates.
        glossary: Controls the MCP glossary consult — None (env-gated, default
            off), False (off), True (default MCP manager), or an injected
            manager with ``call_tool``. See ``_consult_glossary``.

    Returns:
        A ResearchResult with the question, evidence, follow-up count, and
        any glossary definitions resolved via MCP.
    """
    # Round 1: Decompose and search. Always search the ORIGINAL question too — the model's
    # sub-queries can drift away from the exact keywords (hurting recall on specific topics
    # like "hiring" or "SSO"), so keep the raw question in the mix.
    sub_queries = _decompose_question(question, llm)
    if question not in sub_queries:
        sub_queries = [question] + sub_queries

    # Notify assistant of decomposition
    if assistant is not None:
        assistant.set_status("Decomposing question…")
        assistant.emit_trace("decompose", f"{len(sub_queries)} sub-queries: " + "; ".join(sub_queries))

    # Consult the org glossary over MCP for domain terms in the question
    # (optional + defensive; no-op unless enabled or a manager is injected).
    glossary_entries = _consult_glossary(question, glossary, assistant=assistant)

    # Feed the resolved definitions back into retrieval: each term's canonical long-form
    # becomes an extra sub-query, so acronym/jargon questions ("ARR") also match spelled-out
    # evidence. This is the MCP consult *earning its place* — not just annotating the result.
    glossary_expansions: list[str] = []
    if glossary_entries:
        glossary_expansions = _glossary_expansions(glossary_entries)
        added = [e for e in glossary_expansions[:3] if e not in sub_queries]
        if added:
            sub_queries = sub_queries + added
            if assistant is not None:
                assistant.emit_trace("glossary", "expanded search via MCP: " + "; ".join(added))

    evidence, _ = _gather_evidence(sub_queries, rts, assistant=assistant)
    
    follow_up_count = 0

    # Fire follow-up hops while coverage stays thin, up to max_follow_ups
    while follow_up_count < max_follow_ups and _detect_thin_coverage(evidence, follow_up_threshold):
        # Notify assistant of follow-up
        if assistant is not None:
            assistant.set_status("Cross-checking for gaps…")
            assistant.emit_trace("cross-check", "follow-up hop")

        follow_up_query = _generate_follow_up_query(question, evidence, llm)
        follow_up_evidence, _ = _gather_evidence([follow_up_query], rts, assistant=assistant)

        # Dedup follow-up evidence against existing, keeping the best score per message
        evidence_by_key = {_evidence_key(e): e for e in evidence}
        for fe in follow_up_evidence:
            key = _evidence_key(fe)
            existing = evidence_by_key.get(key)
            if existing is not None:
                existing.score = max(existing.score, fe.score)
            else:
                evidence_by_key[key] = fe
                evidence.append(fe)

        # Re-sort and re-index
        evidence.sort(key=lambda e: e.score, reverse=True)
        for i, ev in enumerate(evidence, start=1):
            ev.citation_index = i

        follow_up_count += 1

    # Re-rank by relevance to the ORIGINAL question and keep the most relevant, so a broad
    # corpus doesn't dilute synthesis with cross-topic hits. Sub-query scores aren't comparable
    # across queries; overlap with the actual question is the honest final signal.
    evidence = _rank_by_question(question, evidence, top_k=8, expansions=glossary_expansions)
    for i, ev in enumerate(evidence, start=1):
        ev.citation_index = i

    # Notify assistant before synthesis
    if assistant is not None:
        assistant.set_status("Synthesizing answer…")
        assistant.emit_trace("synthesis", f"{len(evidence)} evidence items")

    # Build the knowledge graph over the gathered evidence
    from conduit.knowledge_graph import build_graph

    return ResearchResult(
        question=question,
        evidence=evidence,
        follow_up_hops=follow_up_count,
        graph=build_graph(evidence, question=question),
        glossary=glossary_entries,
    )


def _rank_by_question(
    question: str,
    evidence: list[Evidence],
    top_k: int = 8,
    expansions: Optional[list[str]] = None,
) -> list[Evidence]:
    """Order evidence by topic overlap with the question (tie-broken by recency) and keep the top
    ``top_k`` — but DROP zero-overlap off-topic prose so it can't dilute synthesis with unrelated
    hits (the observed "MFA note cited in a pricing answer" leak).

    Two protections keep this from starving the money-shot:
      * overlap is measured against the question keywords UNION the MCP glossary ``expansions`` (so
        an "ARR" question keeps evidence that only spells out "Annual Recurring Revenue"), and
      * value-bearing evidence (money/pct) is always kept — a terse "changed it to $20" that omits
        the topic word is exactly the reversal signal the resolver needs.
    Never returns empty: if nothing survives, fall back to the ranked list."""
    import re as _re
    from conduit.contradiction import _keywords, _light_stem, extract_typed_values

    qstems = {_light_stem(w) for w in _keywords(question, None)}
    for exp in (expansions or []):
        for w in _re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", exp):
            qstems.add(_light_stem(w))
    if not qstems:
        return evidence[:top_k]

    def _overlap(ev: Evidence) -> int:
        stems = {_light_stem(w) for w in _re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", ev.text or "")}
        return len(qstems & stems)

    def _recency(ev: Evidence) -> float:
        try:
            return float(ev.ts)
        except (TypeError, ValueError):
            return 0.0

    def _value_bearing(ev: Evidence) -> bool:
        return any(c in ("money", "pct") for c, _ in extract_typed_values(ev.text or ""))

    scored = [(ev, _overlap(ev), _recency(ev)) for ev in evidence]
    scored.sort(key=lambda t: (t[1], t[2]), reverse=True)
    kept = [ev for ev, ov, _ in scored if ov > 0 or _value_bearing(ev)]
    ranked = kept if kept else [ev for ev, _, _ in scored]
    return ranked[:top_k]


def synthesize(result: ResearchResult, llm: LLMClient) -> Any:
    """Synthesize an answer from research evidence with citations.
    
    This is a convenience wrapper that imports and calls the synthesize
    function from citations.py.
    
    Args:
        result: The ResearchResult containing evidence.
        llm: The LLM client to use for synthesis.
        
    Returns:
        An Answer with text containing [n] markers and corresponding citations.
    """
    from conduit.citations import synthesize as citation_synthesize
    return citation_synthesize(result, llm)
