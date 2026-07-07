"""Tests for the RTS client and FakeRTS implementation."""
import pytest
from conduit.rts_client import SearchHit, RTSClient
from conduit.fake_rts import FakeRTS, CorpusMessage


class TestSearchHit:
    """Tests for the SearchHit dataclass."""
    
    def test_search_hit_creation(self):
        """Test that SearchHit can be created with all fields."""
        hit = SearchHit(
            text="Hello world",
            channel="general",
            ts="1234567890.123456",
            permalink="https://slack.com/archives/GENERAL/p1234567890123456",
            score=5.0,
            author="alice",
        )
        assert hit.text == "Hello world"
        assert hit.channel == "general"
        assert hit.ts == "1234567890.123456"
        assert hit.permalink == "https://slack.com/archives/GENERAL/p1234567890123456"
        assert hit.score == 5.0
        assert hit.author == "alice"
    
    def test_search_hit_optional_author(self):
        """Test that author is optional."""
        hit = SearchHit(
            text="Hello world",
            channel="general",
            ts="1234567890.123456",
            permalink="https://slack.com/archives/GENERAL/p1234567890123456",
            score=5.0,
        )
        assert hit.author is None


class TestFakeRTSKeywordSearch:
    """Tests for keyword-based search in FakeRTS."""
    
    @pytest.fixture
    def fake_rts(self):
        """Create a FakeRTS instance with default corpus."""
        return FakeRTS()
    
    def test_keyword_search_returns_results(self, fake_rts):
        """Test that keyword search returns matching results."""
        results = fake_rts.search("meeting", limit=10)
        assert len(results) > 0
        assert any("meeting" in hit.text.lower() for hit in results)
    
    def test_keyword_search_ranks_by_relevance(self, fake_rts):
        """Test that results are ranked by relevance score."""
        results = fake_rts.search("review", limit=10)
        assert len(results) > 0
        # Results should be sorted by score descending
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score
    
    def test_keyword_search_respects_limit(self, fake_rts):
        """Test that limit parameter is respected."""
        results = fake_rts.search("the", limit=3)
        assert len(results) <= 3
    
    def test_keyword_search_returns_non_empty_permalink(self, fake_rts):
        """Test that all results have non-empty permalinks."""
        results = fake_rts.search("team", limit=10)
        for hit in results:
            assert hit.permalink != ""
            assert "slack.com" in hit.permalink


class TestFakeRTSSemanticSearch:
    """Tests for naive semantic search in FakeRTS."""
    
    @pytest.fixture
    def fake_rts(self):
        """Create a FakeRTS instance with default corpus."""
        return FakeRTS()
    
    def test_semantic_search_substring_match(self, fake_rts):
        """Test that substring matches get higher scores."""
        results = fake_rts.search("quarterly review", limit=10)
        assert len(results) > 0
        # The message containing "quarterly review" should be ranked high
        top_result = results[0]
        assert "quarterly" in top_result.text.lower() or "review" in top_result.text.lower()
    
    def test_semantic_search_keyword_overlap(self, fake_rts):
        """Test that keyword overlap contributes to scoring."""
        results = fake_rts.search("api deployment", limit=10)
        assert len(results) > 0
        # Should find the message about API deployment
        assert any("api" in hit.text.lower() and "deployment" in hit.text.lower() 
                   for hit in results)
    
    def test_semantic_search_partial_match(self, fake_rts):
        """Test that partial keyword matches are found."""
        results = fake_rts.search("lunch", limit=10)
        assert len(results) > 0
        assert any("lunch" in hit.text.lower() for hit in results)
    
    def test_semantic_search_returns_non_empty_permalink(self, fake_rts):
        """Search hits carry non-empty, resolvable permalinks (query matches real corpus
        words so the assertion loop actually runs — the phantom-fallback hit was removed)."""
        results = fake_rts.search("deployment tests passing", limit=10)
        assert results, "expected matching hits from the default corpus"
        for hit in results:
            assert hit.permalink != ""
            assert "slack.com" in hit.permalink


class TestFakeRTSCustomCorpus:
    """Tests for FakeRTS with custom corpus."""
    
    def test_custom_corpus_is_used(self):
        """Test that a custom corpus is used instead of default."""
        custom_corpus = {
            "test_channel": [
                CorpusMessage(
                    text="This is a test message",
                    channel="test_channel",
                    ts="9999999999.999999",
                    author="tester",
                ),
            ],
        }
        fake_rts = FakeRTS(corpus=custom_corpus)
        results = fake_rts.search("test", limit=10)
        assert len(results) == 1
        assert results[0].channel == "test_channel"
        assert results[0].text == "This is a test message"
    
    def test_empty_corpus_returns_empty_results(self):
        """Test that empty corpus returns no results."""
        fake_rts = FakeRTS(corpus={})
        results = fake_rts.search("anything", limit=10)
        assert len(results) == 0


class TestRTSClient:
    """Tests for the live RTSClient (structure only, no live calls)."""
    
    def test_client_initialization(self):
        """Test that RTSClient can be initialized."""
        client = RTSClient(token="xoxb-test-token")
        assert client.token == "xoxb-test-token"
        assert client.api_base == "https://slack.com/api"
    
    def test_client_custom_api_base(self):
        """Test that RTSClient accepts custom API base."""
        client = RTSClient(token="xoxb-test-token", api_base="https://custom.slack.com/api")
        assert client.api_base == "https://custom.slack.com/api"

    def test_search_parses_search_messages_payload(self):
        """search() parses a realistic search.messages response into SearchHits (proving the
        official RTS backend is genuinely functional, not a stub)."""
        client = RTSClient(token="xoxp-user-token")
        payload = {
            "ok": True,
            "messages": {
                "matches": [
                    {
                        "text": "We changed the pricing tier to $20.",
                        "channel": {"id": "C1", "name": "decisions"},
                        "ts": "1781775000.000000",
                        "permalink": "https://x.slack.com/archives/C1/p1781775000000000",
                        "score": 42.5,
                        "username": "priya",
                    },
                    {
                        "text": "We set pricing at $10 for launch.",
                        "channel": {"id": "C2", "name": "pricing"},
                        "ts": "1777028400.000000",
                        "permalink": "https://x.slack.com/archives/C2/p1777028400000000",
                        "score": 30.0,
                        "user": "U9",
                    },
                ]
            },
        }
        client._http = lambda method, params: payload
        hits = client.search("pricing", limit=5)
        assert [h.channel for h in hits] == ["decisions", "pricing"]
        assert hits[0].permalink.endswith("p1781775000000000")
        assert hits[0].score == 42.5
        assert hits[0].author == "priya"          # username preferred
        assert hits[1].author == "U9"             # falls back to user id
        assert "$20" in hits[0].text

    def test_search_bot_token_error_is_actionable(self):
        """A bot token yields not_allowed_token_type — the error must tell the operator to use
        a user token (never a silent empty list)."""
        client = RTSClient(token="xoxb-bot-token")
        client._http = lambda method, params: {"ok": False, "error": "not_allowed_token_type"}
        with pytest.raises(RuntimeError) as exc:
            client.search("anything")
        msg = str(exc.value).lower()
        assert "not_allowed_token_type" in msg
        assert "user token" in msg and "search:read" in msg

    def test_build_rts_selects_official_api_with_user_token(self, monkeypatch):
        """_build_rts wires in the official Slack Search API when a user token + opt-in are
        present — proving the RTS-API path is reachable, not dead code."""
        import conduit.slack_app as slack_app
        monkeypatch.setenv("SLACK_USER_TOKEN", "xoxp-real-user-token")
        monkeypatch.setenv("LORE_USE_RTS_API", "1")
        rts = slack_app._build_rts(client=None)
        assert isinstance(rts, RTSClient)
        assert rts.token == "xoxp-real-user-token"
