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
    
    # Build prompt for LLM
    messages = [
        {
            "role": "system",
            "content": """You are a research synthesis assistant. Your task is to:

1. Read the evidence provided with their citation indices [n].
2. Synthesize a coherent answer to the research question.
3. Use inline [n] citation markers to reference specific evidence.
4. Detect any contradictions or timeline drift (e.g., "decided X ... later changed to Y") and explicitly state the CURRENT value.
5. Only use citation markers that correspond to actual evidence.

Format your answer as plain text with [n] markers where claims are made.
After your answer, list each citation in this format:
CITATION: [n] | channel | permalink | "quote"

Example:
The team decided to use Python [1], but later switched to Go [2]. The current stack is Go.

CITATION: [1] | #engineering | https://slack.com/archives/... | "We'll use Python"
CITATION: [2] | #engineering | https://slack.com/archives/... | "Switching to Go"

Make sure every [n] in your text has a corresponding CITATION line."""
        },
        {
            "role": "user",
            "content": f"Research question: {result.question}\n\nEvidence:\n{evidence_context}"
        }
    ]
    
    response = llm.chat(messages)
    content = response.get("content") or ""  # guard: a model may return content=None

    # Extract citations from response
    citations = _extract_citations_from_response(content, result.evidence)
    
    # Remove CITATION lines from the text
    text = re.sub(r'\nCITATION:.*', '', content, flags=re.MULTILINE)
    text = text.strip()
    
    # Validate and clean up citation markers
    text = _validate_citation_markers(text, citations)
    
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

    return Answer(
        text=text,
        citations=citations,
        drift=drift,
        graph_summary=graph.summary(),
    )
