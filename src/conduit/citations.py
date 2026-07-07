"""Citation handling and synthesis for research results."""
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from conduit.research import ResearchResult, Evidence
from conduit.agent import LLMClient
from conduit.contradiction import detect_drift, resolve_answer_text, TimelineDrift
from conduit.knowledge_graph import build_graph


@dataclass
class Citation:
    """A citation reference to a piece of evidence.
    
    Attributes:
        index: The 1-based citation index (matches [n] markers in text).
        permalink: Stable URL to the source message.
        channel: The channel where the evidence was found.
        quote: The exact text from the evidence.
    """
    index: int
    permalink: str
    channel: str
    quote: str


@dataclass
class Answer:
    """Synthesized answer with citation markers.

    Attributes:
        text: The answer text with inline [n] citation markers.
        citations: List of Citation objects mapping indices to sources.
        drift: Detected timeline drift (the reversal), or None. Grounds the
            "conflicting signals" Canvas section and the current-value claim.
        graph_summary: Knowledge-graph summary dict (entities/decisions/reversals),
            rendered as the Canvas "Decision Graph" badge — visible proof of deep research.
    """
    text: str
    citations: list[Citation] = field(default_factory=list)
    drift: Optional[TimelineDrift] = None
    graph_summary: Optional[dict[str, Any]] = None


def _validate_citation_markers(text: str, citations: list[Citation]) -> str:
    """Validate that all [n] markers in text resolve to citations.
    
    Args:
        text: The answer text with citation markers.
        citations: List of Citation objects.
        
    Returns:
        Text with any dangling markers removed.
    """
    # Find all citation markers [n]
    marker_pattern = r'\[(\d+)\]'
    markers = re.findall(marker_pattern, text)
    
    # Get valid citation indices
    valid_indices = {str(c.index) for c in citations}
    
    # Remove markers that don't have corresponding citations
    def replace_invalid(match):
        marker_num = match.group(1)
        if marker_num in valid_indices:
            return match.group(0)
        return ''  # Remove dangling marker
    
    return re.sub(marker_pattern, replace_invalid, text)


def _extract_citations_from_response(content: str, evidence: list[Evidence]) -> list[Citation]:
    """Extract citations from LLM response and map to evidence.
    
    IMPORTANT: All citation data (permalink, channel, quote) MUST come from
    the actual Evidence objects, not from what the LLM writes. The LLM may
    only choose which [n] indices to cite, never what the link/channel/quote is.
    
    Args:
        content: The LLM response text.
        evidence: The original evidence list to map citations to.
        
    Returns:
        List of Citation objects with data grounded from evidence.
    """
    citations = []
    
    # Try to parse explicit CITATION: lines first to get indices
    citation_pattern = r'CITATION:\s*\[(\d+)\]\s*\|\s*(\S+)\s*\|\s*(\S+)\s*\|\s*"([^"]*)"'
    citation_matches = re.findall(citation_pattern, content)
    
    if citation_matches:
        for match in citation_matches:
            index_str, _channel_from_llm, _permalink_from_llm, _quote_from_llm = match
            try:
                idx = int(index_str)
                # Validate index is within evidence bounds
                if 1 <= idx <= len(evidence):
                    ev = evidence[idx - 1]
                    # ALWAYS use data from evidence, not from LLM
                    citations.append(Citation(
                        index=idx,
                        permalink=ev.permalink,
                        channel=ev.channel,
                        quote=ev.text[:200],  # Truncate long quotes
                    ))
            except (ValueError, IndexError):
                continue
    else:
        # Fallback: map [n] markers to evidence by index
        marker_pattern = r'\[(\d+)\]'
        markers = re.findall(marker_pattern, content)
        
        for marker in markers:
            try:
                idx = int(marker)
                if 1 <= idx <= len(evidence):
                    ev = evidence[idx - 1]
                    citations.append(Citation(
                        index=idx,
                        permalink=ev.permalink,
                        channel=ev.channel,
                        quote=ev.text[:200],  # Truncate long quotes
                    ))
            except (ValueError, IndexError):
                continue
    
    # Deduplicate by index
    seen_indices = set()
    unique_citations = []
    for c in citations:
        if c.index not in seen_indices:
            seen_indices.add(c.index)
            unique_citations.append(c)
    
    return unique_citations


def synthesize(result: ResearchResult, llm: LLMClient) -> Answer:
    """Synthesize an answer from research evidence with citations.
    
    Feeds the Evidence to the LLM and produces an answer whose claims carry
    inline [n] citation markers mapping to evidence indices. Detects
    contradiction/timeline drift and makes the answer state the current value
    explicitly.
    
    Args:
        result: The ResearchResult containing evidence.
        llm: The LLM client to use for synthesis.
        
    Returns:
        An Answer with text containing [n] markers and corresponding citations.
    """
    if not result.evidence:
        return Answer(text="No evidence found to answer this question.", citations=[])
    
    # Build evidence context for the LLM
    evidence_context = "\n\n".join([
        f"[{e.citation_index}] Channel: {e.channel}, Author: {e.author or 'Unknown'}\n"
        f"Text: {e.text}\n"
        f"Permalink: {e.permalink}"
        for e in result.evidence
    ])
    
    # Build prompt for LLM. Keep the required output SHORT — the citations are grounded from
    # the evidence by their [n] index (see _extract_citations_from_response), so the model does
    # NOT need to write verbose CITATION lines. A concise answer keeps generation fast.
    messages = [
        {
            "role": "system",
            "content": """You are a research synthesis assistant. Using ONLY the evidence below \
(each tagged with an index like [1], [2]), write a SHORT, direct answer (2-4 sentences) to the \
question. You MUST place an inline [n] marker immediately after EVERY claim, citing the exact \
evidence index you used — an answer with no [n] markers is wrong. If a value or decision changed \
over time, state the CURRENT value explicitly. Never invent facts or citations. Output ONLY the \
answer sentences with their [n] markers — no preamble, no separate source list.

Example: The team set pricing at $10 [1], then raised it to $20 [2]; the current price is $20 [2]."""
        },
        {
            "role": "user",
            "content": f"Question: {result.question}\n\nEvidence:\n{evidence_context}\n\nAnswer:"
        }
    ]
    
    response = llm.chat(messages)
    content = response.get("content") or ""  # guard: a model may return content=None

    # Normalise grouped markers "[1, 3]" / "[1,3]" -> "[1][3]" so each index is cited.
    content = re.sub(
        r"\[(\d+(?:\s*,\s*\d+)+)\]",
        lambda m: "".join(f"[{i.strip()}]" for i in m.group(1).split(",")),
        content,
    )

    # Extract citations from response
    citations = _extract_citations_from_response(content, result.evidence)
    
    # Remove CITATION lines from the text
    text = re.sub(r'\nCITATION:.*', '', content, flags=re.MULTILINE)
    text = text.strip()
    
    # Validate and clean up citation markers
    text = _validate_citation_markers(text, citations)

    # Fallback: if the model wrote no [n] markers (so nothing was cited), ground the answer in
    # the top evidence anyway — an answer should always show its sources.
    if not citations and result.evidence:
        top = sorted(result.evidence, key=lambda e: e.citation_index)[:3]
        citations = [Citation(index=e.citation_index, permalink=e.permalink,
                              channel=e.channel, quote=e.text[:200]) for e in top]

    # Build (or reuse) the ephemeral knowledge graph — the reasoning substrate (entities +
    # typed edges) whose summary becomes the Canvas "Decision Graph" badge. research.run may
    # have already attached it to the result; reuse it so the badge, the drift, and the answer
    # all read from ONE graph.
    graph = getattr(result, "graph", None)
    if graph is None:
        graph = build_graph(result.evidence, question=result.question)

    # Contradiction / timeline-drift resolution — DETERMINISTIC, so the money-shot never
    # depends on the local model's phrasing. Evidence-grounded detector first; if it finds
    # nothing (e.g. an over-eager keyword gate), fall back to the graph's supersedes chain so
    # a reversal the graph captured is still surfaced. Both order by float ts and agree.
    drift = detect_drift(result.evidence, question=result.question)
    if drift is None:
        drift = graph.drift_for_question(result.question)
    if drift:
        text = resolve_answer_text(text, drift)

    # Defuse any Slack control sequences (broadcast pings / links) an injected message could have
    # steered the model to emit — the answer body is rendered raw on every surface, unlike the
    # citation quotes which are already escaped. Keeps [n] markers so deep-links still render.
    from conduit.textsafe import neutralize_answer_body
    text = neutralize_answer_body(text)

    return Answer(
        text=text,
        citations=citations,
        drift=drift,
        graph_summary=graph.summary(),
    )
