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
    evidence_by_permalink: dict[str, Evidence] = {}
    search_calls = 0

    for sub_query in sub_queries:
        hits = rts.search(sub_query, limit=limit_per_query)
        search_calls += 1

        if assistant is not None:
            channels = sorted({h.channel for h in hits if getattr(h, "channel", "")})
            where = " in #" + ", #".join(channels[:3]) if channels else ""
            assistant.set_status(f"Searching “{sub_query}”…")
            assistant.emit_trace("search", f"“{sub_query}” → {len(hits)} hits{where}")

        for hit in hits:
            existing = evidence_by_permalink.get(hit.permalink)
            if existing is not None:
                # Keep one Evidence per permalink, but retain the best score seen.
                existing.score = max(existing.score, hit.score)
            else:
                evidence_by_permalink[hit.permalink] = Evidence(
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
    evidence_list = list(evidence_by_permalink.values())
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
    # Round 1: Decompose and search
    sub_queries = _decompose_question(question, llm)

    # Notify assistant of decomposition
    if assistant is not None:
        assistant.set_status("Decomposing question…")
        assistant.emit_trace("decompose", f"{len(sub_queries)} sub-queries: " + "; ".join(sub_queries))

    # Consult the org glossary over MCP for domain terms in the question
    # (optional + defensive; no-op unless enabled or a manager is injected).
    glossary_entries = _consult_glossary(question, glossary, assistant=assistant)

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

        # Dedup follow-up evidence against existing, keeping the best score per permalink
        evidence_by_permalink = {e.permalink: e for e in evidence}
        for fe in follow_up_evidence:
            existing = evidence_by_permalink.get(fe.permalink)
            if existing is not None:
                existing.score = max(existing.score, fe.score)
            else:
                evidence_by_permalink[fe.permalink] = fe
                evidence.append(fe)

        # Re-sort and re-index
        evidence.sort(key=lambda e: e.score, reverse=True)
        for i, ev in enumerate(evidence, start=1):
            ev.citation_index = i

        follow_up_count += 1

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
