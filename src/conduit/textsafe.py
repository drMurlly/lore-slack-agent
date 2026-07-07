"""Neutralize UNTRUSTED text before rendering it into Slack mrkdwn or Canvas markdown.

Lore quotes raw indexed message text verbatim into its cited Canvas report and posted answer.
Without escaping, any message Lore indexes can inject markup into Lore's *trusted*,
channel-shared report — most dangerously a clickable link: a message like
``Payroll <https://phish.example|Payroll Portal>`` (Slack mrkdwn) or
``[Payroll](https://phish.example)`` (Canvas markdown) would render as a live, attacker-controlled
link under Lore's authority. These helpers defuse that.
"""
import re


def mrkdwn_safe(text: str) -> str:
    """Escape Slack **mrkdwn** control chars so quoted text can't form a link (``<url|label>``)
    or start a code span. Slack renders ``&lt; &gt; &amp;`` as literal angle brackets/ampersand."""
    text = (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return text.replace("`", "ʼ")  # a backtick would open a code span — use a lookalike


def markdown_safe(text: str) -> str:
    """Neutralize Markdown link/image/emphasis/code syntax in quoted text for the Canvas renderer.

    Escapes the structural chars of links/images/code (`[ ] ( ) \\``) and drops the image bang, so
    ``[x](url)``, ``![](url)`` and `` `code` `` render as literal text, not active markup. Angle
    brackets are escaped too, so a CommonMark/GFM autolink (``<https://phish>``) or mailto autolink
    (``<a@phish>``) in quoted text can't render as a LIVE link in the channel-shared Canvas."""
    text = text or ""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")  # kill autolinks
    text = re.sub(r"[\[\]()`]", lambda m: "\\" + m.group(0), text)
    return text.replace("!", "")  # image syntax ![](...) — drop the bang


# Slack control sequences a model could be steered (by injected evidence) to emit into the answer
# body: broadcast pings, user/channel/usergroup refs, and <url|label> links. All start with '<'.
_SLACK_CONTROL_RE = re.compile(r"<[^<>]*>")


def neutralize_answer_body(text: str) -> str:
    """Defuse Slack control sequences in the LLM-written answer body before it is posted.

    The answer is synthesized over UNTRUSTED indexed evidence, so a prompt-injected message could
    steer the model to emit ``<!channel>`` (a real workspace-wide ping under Lore's identity) or
    ``<https://evil|click here>`` (a live link). This strips those angle-bracket forms — keeping the
    human label of a ``<url|label>`` — then escapes any stray ``<``/``>``. It deliberately leaves
    ``[n]`` citation markers and their ``()`` untouched so ``canvas.py`` can still deep-link them.
    """
    text = text or ""

    def _repl(m: "re.Match[str]") -> str:
        inner = m.group(0)[1:-1]
        return inner.split("|", 1)[1] if "|" in inner else ""  # keep label; drop pings/refs/urls

    text = _SLACK_CONTROL_RE.sub(_repl, text)
    return text.replace("<", "&lt;").replace(">", "&gt;")


def oneline(text: str, limit: int = 300) -> str:
    """Collapse all whitespace (incl. newlines) to single spaces — for titles / H1 headings where
    an embedded newline would break the heading."""
    return re.sub(r"\s+", " ", (text or "").strip())[:limit]
