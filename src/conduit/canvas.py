"""Canvas report building for Slack canvases.create API."""
import re
from typing import Any, Optional

from conduit.citations import Answer, Citation
from conduit.contradiction import conflict_canvas_section
from conduit.knowledge_graph import graph_badge_from_summary


def _timeline_rows(graph: Any, question: str) -> list[dict[str, str]]:
    """Oldest→newest decision rows for the question's primary topic, from the knowledge graph.

    Delegates to ``graph.decision_rows`` which is class-anchored (money/pct/num) and collapses
    consecutive duplicate values — so the timeline never mixes a price with a stray count and
    the last row is the true current value. Empty when the graph has no decision chain — this
    is the visible proof that Lore reasoned over a decision *graph*, not a hit list.
    """
    if graph is None:
        return []
    try:
        return graph.decision_rows(question)
    except Exception:
        return []


def build_report(answer: Answer, question: str, graph: Any = None) -> dict[str, Any]:
    """Build a canvas document content payload for Slack's canvases.create API.

    Produces a document_content dict with:
    - H1 title with the research question
    - "Decision Graph" badge (entities/decisions/reversals) — proof of deep research
    - "🕸️ Decision timeline" section (oldest→newest, from the graph) when ``graph`` is given
    - "⚠️ Conflicting signals over time" section when a reversal was detected (money-shot)
    - Answer body with [n] citation markers rendered as deep-links
    - Sources section with clickable links to each source message

    Args:
        answer: The synthesized Answer with text, citations, drift, and graph_summary.
        question: The original research question to use as title.
        graph: Optional KnowledgeGraph — when present, its decision timeline is rendered.
            The 2-arg call (no graph) still works and omits the timeline section.

    Returns:
        Document content dict suitable for Slack canvases.create API.
    """
    # Build the document blocks
    blocks: list[dict[str, Any]] = []

    # H1 title with the question
    blocks.append({
        "type": "header",
        "text": {
            "type": "plain_text",
            "text": f"Research: {question}",
        }
    })

    # Decision Graph badge — visible proof Lore reasoned over a graph, not a search list.
    if answer.graph_summary:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": graph_badge_from_summary(answer.graph_summary),
            },
        })

    # Decision timeline (oldest→newest) straight from the graph — the structure behind
    # the answer. Rendered when a graph with ≥2 timeline rows is supplied.
    rows = _timeline_rows(graph, question)
    if len(rows) >= 2:
        lines = ["🕸️ *Decision timeline*"]
        for r in rows:
            where = f"<{r['permalink']}|#{r['channel']}>" if r["permalink"] else f"#{r['channel']}"
            lines.append(f"• *{r['value']}* — {where}")
        lines.append(f"*Current: {rows[-1]['value']}*")
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "\n".join(lines)},
        })

    # Conflicting-signals section FIRST when there's a reversal — the thing a search
    # wrapper can never surface. Grounded in the two source messages.
    conflict = conflict_canvas_section(answer.drift)
    if conflict:
        blocks.append(conflict)

    # Answer body - convert [n] markers to markdown links
    answer_text = _format_answer_with_links(answer.text, answer.citations)
    blocks.append({
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": answer_text,
        }
    })

    # Sources section
    if answer.citations:
        sources_text = _build_sources_section(answer.citations)
        blocks.append({
            "type": "divider",
        })
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": sources_text,
            }
        })

    return {
        "document_content": {
            "type": "document",
            "blocks": blocks,
        }
    }


def build_report_markdown(answer: Answer, question: str, graph: Any = None) -> str:
    """Build a Canvas document as a **markdown string** for Slack's ``canvases.create`` API.

    The live ``canvases.create`` accepts ``document_content={"type": "markdown", "markdown": …}``
    (not Block Kit), so this renders the same report — title, Decision-Graph badge, the
    decision timeline, the conflicting-signals section, the cited answer, and a Sources list —
    as GitHub-flavored markdown with ``[text](url)`` links.

    Args:
        answer: The synthesized Answer (text, citations, drift, graph_summary).
        question: The original research question (becomes the H1).
        graph: Optional KnowledgeGraph — when present, its decision timeline is rendered.

    Returns:
        A markdown string ready for ``canvases.create``.
    """
    from conduit.textsafe import markdown_safe, oneline
    # The question and quotes are untrusted (user input / raw indexed messages) — neutralize
    # markup so they can't inject a link/heading into the trusted Canvas, and keep the H1 one line.
    lines: list[str] = [f"# Research: {markdown_safe(oneline(question, 200))}", ""]

    if answer.graph_summary:
        badge = graph_badge_from_summary(answer.graph_summary)
        lines += [_mrkdwn_to_markdown(badge), ""]

    rows = _timeline_rows(graph, question)
    if len(rows) >= 2:
        lines += ["## 🕸️ Decision timeline", ""]
        for r in rows:
            where = f"[#{r['channel']}]({r['permalink']})" if r["permalink"] else f"#{r['channel']}"
            lines.append(f"- **{r['value']}** — {where}")
        lines += [f"- **Current: {rows[-1]['value']}**", ""]

    if answer.drift:
        d = answer.drift
        older = getattr(d.older, "permalink", "") or ""
        newer = getattr(d.newer, "permalink", "") or ""
        ochan = getattr(d.older, "channel", "?")
        nchan = getattr(d.newer, "channel", "?")
        lines += [
            "## ⚠️ Conflicting signals over time — the answer changed",
            "",
            f"- Earlier: **{d.old_value}** ([#{ochan}]({older}))" if older
            else f"- Earlier: **{d.old_value}** (#{ochan})",
            f"- Later / current: **{d.new_value}** ([#{nchan}]({newer}))" if newer
            else f"- Later / current: **{d.new_value}** (#{nchan})",
            f"- Lore resolves to the most recent decision: **{d.current_value}**",
            "",
        ]

    lines += ["## Answer", "", _format_answer_with_links(answer.text, answer.citations), ""]

    if answer.citations:
        lines += ["---", "", "## Sources", ""]
        for c in sorted(answer.citations, key=lambda x: x.index):
            quote = markdown_safe(oneline(c.quote))  # untrusted message text → no active markup
            where = f"[#{c.channel}]({c.permalink})" if c.permalink else f"#{c.channel}"
            lines.append(f"{c.index}. {where} — \"{quote}\"")

    return "\n".join(lines)


def _mrkdwn_to_markdown(text: str) -> str:
    """Convert Slack mrkdwn to standard markdown: ``<url|label>`` → ``[label](url)``."""
    text = re.sub(r"<([^|>]+)\|([^>]+)>", r"[\2](\1)", text)
    text = re.sub(r"<(https?://[^>]+)>", r"\1", text)
    return text


def _format_answer_with_links(text: str, citations: list[Citation]) -> str:
    """Format answer text, converting [n] markers to clickable links.
    
    Args:
        text: The answer text with [n] citation markers.
        citations: List of Citation objects to map markers to.
        
    Returns:
        Text with [n] markers converted to markdown links.
    """
    if not citations:
        return text
    
    # Build a mapping from index to permalink
    citation_map = {c.index: c.permalink for c in citations}
    
    # Replace [n] with markdown links
    import re
    
    def replace_marker(match):
        index = int(match.group(1))
        if index in citation_map:
            permalink = citation_map[index]
            return f"[{index}]({permalink})"
        return match.group(0)  # Keep original if no link
    
    return re.sub(r'\[(\d+)\]', replace_marker, text)


def _build_sources_section(citations: list[Citation]) -> str:
    """Build the Sources section text.
    
    Args:
        citations: List of Citation objects.
        
    Returns:
        Formatted sources section text in markdown.
    """
    lines = ["*Sources:*"]
    
    for citation in sorted(citations, key=lambda c: c.index):
        # Format: [n] Channel: #channel - "quote"
        quote = citation.quote.replace('"', "'")  # Escape quotes
        lines.append(f"[{citation.index}] <{citation.permalink}|#{citation.channel}>: \"{quote}\"")
    
    return "\n".join(lines)


def create_canvas(
    client: Any,
    title: str,
    answer: Answer,
    question: str,
    graph: Any = None,
) -> dict[str, Any]:
    """Create a Slack canvas with the research report.

    This is a seam for the live API call that can be mocked in tests. The live
    ``canvases.create`` accepts ``document_content={"type": "markdown", "markdown": …}``,
    so we send markdown (not Block Kit) — the Block Kit ``build_report`` is for message views.

    Args:
        client: Slack API client instance.
        title: The canvas title.
        answer: The synthesized Answer with text and citations.
        question: The original research question.
        graph: Optional KnowledgeGraph for the decision-timeline section.

    Returns:
        The API response from canvases.create.
    """
    markdown = build_report_markdown(answer, question, graph=graph)

    return client.canvases_create(
        title=title,
        document_content={"type": "markdown", "markdown": markdown},
    )
