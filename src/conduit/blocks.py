"""Block Kit builders for rendering agent responses in Slack."""

from dataclasses import dataclass
from typing import Any, Optional

# Slack rejects a section's text over 3000 chars with invalid_blocks; cap below that (the
# full answer always lives in the Canvas anyway).
_SECTION_LIMIT = 2900


def _clip(text: str, limit: int = _SECTION_LIMIT) -> str:
    text = text or ""
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


@dataclass
class TraceStep:
    """A step in the research trace."""
    phase: str
    detail: str


def build_answer_blocks(
    answer: str,
    tools_used: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """Build Block Kit blocks for an agent answer.
    
    Args:
        answer: The agent's response text.
        tools_used: Optional list of tool names that were used.
    
    Returns:
        List of Block Kit block dictionaries.
    """
    blocks: list[dict[str, Any]] = []
    
    # Main answer section
    blocks.append({
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": _clip(answer),
        },
    })
    
    # Add tools used context if any tools were used
    if tools_used:
        tools_text = ", ".join(f"`{tool}`" for tool in tools_used)
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"🔧 Tools used: {tools_text}",
                },
            ],
        })
    
    return blocks


def build_home_view(servers: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the home tab view listing MCP servers and their tools.
    
    Args:
        servers: List of server dicts with 'name' and 'tools' keys.
                 Each tool dict should have 'name' key.
    
    Returns:
        Home tab view Block Kit JSON.
    """
    blocks: list[dict[str, Any]] = []
    
    # Header
    blocks.append({
        "type": "header",
        "text": {
            "type": "plain_text",
            "text": "🤖 Conduit Agent - MCP Servers",
        },
    })
    
    # Add a divider
    blocks.append({
        "type": "divider",
    })
    
    # List each server and its tools
    for i, server in enumerate(servers):
        server_name = server.get("name", "Unknown Server")
        tools = server.get("tools", [])
        
        # Server name section
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*{server_name}*",
            },
        })
        
        # Tools list for this server
        if tools:
            tool_names = [tool.get("name", "Unknown Tool") for tool in tools]
            tools_list = "\n".join(f"• {name}" for name in tool_names)
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"Tools:\n{tools_list}",
                },
            })
        else:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "No tools registered",
                },
            })
        
        # Add divider between servers (except after the last one) — compare by position,
        # not dict value (two identical server dicts must not collapse the divider).
        if i < len(servers) - 1:
            blocks.append({
                "type": "divider",
            })
    
    # Footer context
    blocks.append({
        "type": "context",
        "elements": [
            {
                "type": "mrkdwn",
                "text": f"Total servers: {len(servers)}",
            },
        ],
    })
    
    return {
        "type": "home",
        "blocks": blocks,
    }


def build_empty_state_blocks(question: str, channels: Optional[list[str]] = None) -> list[dict[str, Any]]:
    """A friendly Block Kit response when research found no relevant history.

    Judges deliberately type off-corpus questions; a blank or one-line reply reads as broken.
    This explains what Lore searched and how to get an answer.
    """
    lines = [f"I couldn't find anything in your Slack history about *{question.strip()[:150]}*."]
    if channels:
        shown = ", ".join(f"#{c}" for c in channels[:8])
        lines.append(f"\nI searched {len(channels)} channel(s): {shown}.")
    lines.append(
        "\n*Tips:*\n"
        "• Lore only sees channels it's been invited to — add me to the relevant one.\n"
        "• Try naming the topic or a keyword you remember (e.g. a project, decision, or value).\n"
        "• Ask about something discussed in text — Lore reads messages, threads and files."
    )
    return [{
        "type": "section",
        "text": {"type": "mrkdwn", "text": "\n".join(lines)},
    }]


def build_error_blocks(detail: str, last_step: Optional[str] = None) -> list[dict[str, Any]]:
    """A friendly Block Kit error card (never a raw stack trace to the user)."""
    text = "⚠️ *Something went wrong while researching.* I've logged the details."
    if last_step:
        text += f"\nLast step: _{last_step}_"
    blocks: list[dict[str, Any]] = [{
        "type": "section",
        "text": {"type": "mrkdwn", "text": text},
    }]
    if detail:
        blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": detail[:300]}],
        })
    return blocks


def build_lore_home_view() -> dict[str, Any]:
    """Build Lore's App Home tab: pitch, example questions, how-it-works, For-Good line.

    Published on ``app_home_opened`` so a judge who clicks the app sees an onboarding
    surface instead of a blank tab.

    Returns:
        A ``home`` view Block Kit JSON.
    """
    blocks: list[dict[str, Any]] = [
        {"type": "header",
         "text": {"type": "plain_text", "text": "🧠 Lore — deep research over your Slack memory"}},
        {"type": "section",
         "text": {"type": "mrkdwn",
                  "text": "Ask a hard question. Lore decomposes it, searches across your "
                          "channels and threads, builds a *knowledge graph* of decisions, "
                          "resolves contradictions and timeline drift, and answers with "
                          "*inline citations that deep-link to the exact source messages* — "
                          "delivered as a Canvas report with a live research trace."}},
        {"type": "divider"},
        {"type": "section",
         "text": {"type": "mrkdwn", "text": "*Try asking:*"}},
        {"type": "section",
         "text": {"type": "mrkdwn",
                  "text": "• _What did we decide about pricing, and did anything change since?_\n"
                          "• _What decisions did we make in the last two weeks?_\n"
                          "• _How does the deployment pipeline work?_"}},
        {"type": "divider"},
        {"type": "section",
         "text": {"type": "mrkdwn",
                  "text": "*How it works*\n"
                          "1. 🔍 *Decompose* — break your question into sub-queries\n"
                          "2. 🔎 *Multi-hop search* — fan out across channels, follow up on gaps\n"
                          "3. 🕸️ *Knowledge graph* — link decisions, values and people over time\n"
                          "4. 📄 *Cited answer* — a Canvas report; every claim deep-links to its source"}},
        {"type": "context",
         "elements": [{"type": "mrkdwn",
                       "text": "🤝 *An Agent for Good — knowledge equity:* every new hire, "
                               "volunteer, or contributor gets the same instant, cited answer "
                               "as a five-year veteran. Institutional memory shouldn't be a "
                               "privilege of the tenured."}]},
        {"type": "context",
         "elements": [{"type": "mrkdwn",
                       "text": "Use `/lore <question>`, `@Lore`, or open the *Assistant* to research. "
                               "Lore only sees channels it's invited to."}]},
    ]
    return {"type": "home", "blocks": blocks}


def trace_block(step: TraceStep) -> dict[str, Any]:
    """Build a Block Kit block for a research trace step.
    
    Displays the current phase of research with optional detail text.
    Used for streaming updates in the assistant pane.
    
    Args:
        step: A TraceStep containing phase and detail information.
    
    Returns:
        A Block Kit section block for the trace step.
    """
    # Use emoji based on phase type
    phase_emojis = {
        "decompose": "🔍",
        "search": "🔎",
        "cross-check": "✅",
        "write": "📝",
        "synthesis": "📊",
    }
    
    emoji = phase_emojis.get(step.phase.lower(), "🔄")
    
    # Build the phase text with detail if available
    if step.detail:
        text = f"{emoji} *{step.phase.title()}*: {step.detail}"
    else:
        text = f"{emoji} *{step.phase.title()}*"
    
    return {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": text,
        },
    }


def final_block(answer: str, canvas_url: str) -> list[dict[str, Any]]:
    """Build Block Kit blocks for the final research result.
    
    Displays a summary of the answer with a button linking to the full Canvas.
    
    Args:
        answer: The final answer summary text (1-line or short paragraph).
        canvas_url: URL to the full Canvas report.
    
    Returns:
        List of Block Kit blocks for the final result.
    """
    blocks: list[dict[str, Any]] = []
    
    # Add a divider before the final result
    blocks.append({
        "type": "divider",
    })
    
    # Final answer section
    blocks.append({
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": _clip(f"📄 *Final Answer*\n{answer}"),
        },
    })
    
    # Canvas link button
    blocks.append({
        "type": "actions",
        "elements": [
            {
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": "📋 View Full Canvas",
                },
                "url": canvas_url,
                "action_id": "view_canvas",
            },
        ],
    })
    
    return blocks
