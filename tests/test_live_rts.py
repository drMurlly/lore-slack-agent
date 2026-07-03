"""Tests for SlackHistoryRTS — live-history search driven by a fake Slack client."""
from conduit.live_rts import SlackHistoryRTS
from conduit.rts_client import SearchHit


class FakeSlack:
    """Minimal Slack client returning canned conversations_history per channel."""

    def __init__(self, history: dict[str, list[dict]], users: dict | None = None):
        self._history = history
        self._users = users or {}

    def conversations_history(self, channel, limit=200, cursor=None):
        # single page; ignore cursor for the test corpus
        return {"messages": self._history.get(channel, []), "response_metadata": {}}

    def users_info(self, user):
        return {"user": {"real_name": self._users.get(user, user)}}


def _slack():
    return FakeSlack(
        history={
            "C_PRICING": [
                {"text": "We decided on $10 for the pricing tier.", "ts": "100.0001", "user": "U1"},
                {"text": "Update: we changed pricing to $20 after review.", "ts": "200.0002", "user": "U2"},
                {"text": "channel joined", "ts": "090.0", "user": "U3", "subtype": "channel_join"},
            ],
            "C_RANDOM": [
                {"text": "anyone up for lunch?", "ts": "150.0003", "user": "U4"},
            ],
        },
        users={"U1": "Alice", "U2": "Bob"},
    )


def test_indexes_and_returns_search_hits():
    rts = SlackHistoryRTS(_slack(), channels={"C_PRICING": "pricing", "C_RANDOM": "random"},
                          team_url="https://simon.slack.com").refresh()
    hits = rts.search("pricing", limit=5)
    assert hits and all(isinstance(h, SearchHit) for h in hits)
    assert any("pricing" in h.text.lower() for h in hits)
    # channel display name resolved from the supplied map
    assert hits[0].channel == "pricing"


def test_skips_join_noise():
    rts = SlackHistoryRTS(_slack(), channels=["C_PRICING"], team_url="https://simon.slack.com").refresh()
    assert all("joined" not in h.text for h in rts.search("channel", limit=10))


def test_permalink_is_archive_deeplink():
    rts = SlackHistoryRTS(_slack(), channels={"C_PRICING": "pricing"},
                          team_url="https://simon.slack.com").refresh()
    hit = rts.search("$10", limit=1)[0]
    assert hit.permalink == "https://simon.slack.com/archives/C_PRICING/p1000001"


def test_resolves_author_display_name():
    rts = SlackHistoryRTS(_slack(), channels={"C_PRICING": "pricing"},
                          team_url="https://simon.slack.com").refresh()
    hit = [h for h in rts.search("$10", limit=5) if "$10" in h.text][0]
    assert hit.author == "Alice"


def test_ranks_reversal_messages_for_pricing_question():
    """The two pricing messages both surface for a pricing query (drives the money-shot)."""
    rts = SlackHistoryRTS(_slack(), channels={"C_PRICING": "pricing"},
                          team_url="https://simon.slack.com").refresh()
    texts = " ".join(h.text for h in rts.search("what did we decide about pricing", limit=10))
    assert "$10" in texts and "$20" in texts


def test_lazy_refresh_on_first_search():
    rts = SlackHistoryRTS(_slack(), channels=["C_PRICING"], team_url="https://simon.slack.com")
    assert rts.search("pricing", limit=1)  # refresh happens implicitly
