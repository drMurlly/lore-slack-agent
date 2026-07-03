#!/usr/bin/env python3
"""Lore org-glossary MCP server (official MCP Python SDK, stdio transport).

Exposes canonical definitions for org/domain terms and acronyms so the Lore
research pipeline can ground answers in shared vocabulary. The research loop
connects to this server as an MCP client (see ``conduit.mcp_manager``),
calls ``lookup_terms`` on the user's question, and attaches the resolved
definitions to the research result.

Run standalone:  python servers/glossary_server.py   (speaks MCP over stdio)
"""

import re

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("lore-glossary")

# Canonical org/domain glossary. Keys are matched case-insensitively on word
# boundaries, so "ARR", "arr" and "What's our ARR?" all resolve, while
# "carrying" does not.
GLOSSARY: dict[str, str] = {
    "ARR": "Annual Recurring Revenue — the yearly run-rate of subscription revenue.",
    "MRR": "Monthly Recurring Revenue — subscription revenue normalized to one month.",
    "MAU": "Monthly Active Users — unique users active in the product over the last 30 days.",
    "DAU": "Daily Active Users — unique users active in the product on a given day.",
    "SSO": "Single Sign-On — authentication through the org identity provider (Okta) instead of per-app passwords.",
    "SLA": "Service Level Agreement — the contractual uptime/response commitment (99.9% for Enterprise).",
    "NPS": "Net Promoter Score — the customer-loyalty metric from the quarterly survey.",
    "RTS": "Real-Time Search — Slack's search API surface that Lore mines for evidence.",
    "churn": "The percentage of customers who cancel in a period; reported monthly by Customer Success.",
    "pricing tier": "One of the three sellable plans (Starter, Growth, Enterprise); tier changes need RevOps sign-off.",
    "canvas": "Slack Canvas — the collaborative doc surface where Lore publishes its cited research reports.",
    "runbook": "The on-call operational doc in #incidents describing how to mitigate a class of failure.",
}


@mcp.tool()
def lookup_terms(text: str) -> list[dict[str, str]]:
    """Find org glossary terms/acronyms mentioned in the text and return their
    canonical definitions as a list of {"term", "definition"} objects."""
    found: list[dict[str, str]] = []
    seen: set[str] = set()
    for term, definition in GLOSSARY.items():
        pattern = r"\b" + re.escape(term) + r"\b"
        if re.search(pattern, text, flags=re.IGNORECASE) and term.lower() not in seen:
            seen.add(term.lower())
            found.append({"term": term, "definition": definition})
    return found


@mcp.tool()
def define(term: str) -> str:
    """Return the canonical org definition for a single glossary term, or a
    not-found message if the term is unknown."""
    wanted = term.strip().lower()
    for known, definition in GLOSSARY.items():
        if known.lower() == wanted:
            return definition
    return f"No glossary entry for {term!r}."


if __name__ == "__main__":
    mcp.run(transport="stdio")
