"""Tests for the knowledge graph module."""
import pytest
from conduit.knowledge_graph import KnowledgeGraph, Entity, Edge, build_graph
from conduit.research import Evidence


def _make_evidence(text: str, ts: str, channel: str = "test",
                   permalink: str = "https://slack.com/test", author: str = None) -> Evidence:
    """Helper to create Evidence objects for testing."""
    return Evidence(
        text=text,
        channel=channel,
        ts=ts,
        permalink=permalink,
        score=0.9,
        author=author,
        citation_index=1,
        source_hit=None  # type: ignore
    )


class TestBuildGraph:
    """Tests for build_graph function."""

    def test_build_graph_detects_value_change(self):
        """Test that build_graph detects a value change over time."""
        evidence = [
            _make_evidence("decided $10 for pricing", "100"),
            _make_evidence("changed pricing to $20", "200"),
        ]
        graph = build_graph(evidence)

        summary = graph.summary()
        assert summary["topics"] >= 1
        assert summary["values"] == 2
        assert summary["reversals"] == 1

    def test_build_graph_creates_person_entity_from_mention(self):
        """Test that @mentions create person entities."""
        evidence = [
            _make_evidence("@alice decided $10 for pricing", "100", author="alice"),
        ]
        graph = build_graph(evidence)

        assert "person:alice" in graph.entities
        person = graph.entities["person:alice"]
        assert person.type == "person"
        assert person.label == "alice"


class TestResolveCurrent:
    """Tests for resolve_current method."""

    def test_resolve_current_returns_newest_value(self):
        """Test that resolve_current returns the newest value and edge."""
        evidence = [
            _make_evidence("decided $10 for pricing", "100"),
            _make_evidence("changed pricing to $20", "200"),
        ]
        graph = build_graph(evidence)

        current_value, edge = graph.resolve_current("topic:pricing")

        assert current_value == "$20"
        assert edge is not None
        assert edge.ts == "200"
        assert "$20" in edge.text or "changed" in edge.text.lower()

    def test_resolve_current_returns_none_when_no_topic(self):
        """Test that resolve_current returns (None, None) for unknown topic."""
        evidence = [
            _make_evidence("decided $10 for pricing", "100"),
        ]
        graph = build_graph(evidence)

        current_value, edge = graph.resolve_current("topic:unknown_topic")

        assert current_value is None
        assert edge is None


class TestTimeline:
    """Tests for timeline method."""

    def test_timeline_returns_edges_oldest_to_newest(self):
        """Test that timeline returns edges sorted oldest→newest."""
        evidence = [
            _make_evidence("decided $10 for pricing", "100"),
            _make_evidence("changed pricing to $20", "200"),
            _make_evidence("changed pricing to $30", "300"),
        ]
        graph = build_graph(evidence)

        timeline_edges = graph.timeline("topic:pricing")

        assert len(timeline_edges) >= 2
        # First edge should be the oldest ($10)
        assert "$10" in timeline_edges[0].text or timeline_edges[0].obj == "value:$10"
        # Last edge should be the newest ($30)
        assert "$30" in timeline_edges[-1].text or timeline_edges[-1].obj == "value:$30"


class TestEntityGrounding:
    """Tests that entities have permalinks."""

    def test_value_entities_have_permalinks(self):
        """Test that every value entity has at least one permalink."""
        evidence = [
            _make_evidence("decided $10 for pricing", "100", permalink="https://slack.com/msg1"),
            _make_evidence("changed pricing to $20", "200", permalink="https://slack.com/msg2"),
        ]
        graph = build_graph(evidence)

        for entity in graph.entities.values():
            if entity.type == "value":
                assert len(entity.permalinks) >= 1, f"Value entity {entity.id} has no permalinks"

    def test_topic_entities_have_permalinks(self):
        """Test that every topic entity has at least one permalink."""
        evidence = [
            _make_evidence("decided $10 for pricing", "100", permalink="https://slack.com/msg1"),
        ]
        graph = build_graph(evidence)

        for entity in graph.entities.values():
            if entity.type == "topic":
                assert len(entity.permalinks) >= 1, f"Topic entity {entity.id} has no permalinks"


class TestNeighbors:
    """Tests for neighbors method."""

    def test_neighbors_includes_value_entities(self):
        """Test that neighbors of a topic includes connected value entities."""
        evidence = [
            _make_evidence("decided $10 for pricing", "100"),
            _make_evidence("changed pricing to $20", "200"),
        ]
        graph = build_graph(evidence)

        neighbors = graph.neighbors("topic:pricing")
        neighbor_ids = [n.id for n in neighbors]

        assert "value:$10" in neighbor_ids
        assert "value:$20" in neighbor_ids

    def test_neighbors_includes_person_entities(self):
        """Test that neighbors of a topic includes mentioned person entities."""
        evidence = [
            _make_evidence("@alice decided $10 for pricing", "100"),
        ]
        graph = build_graph(evidence)

        neighbors = graph.neighbors("topic:pricing")
        neighbor_ids = [n.id for n in neighbors]

        assert "person:alice" in neighbor_ids


class TestSupersedesEdges:
    """Tests for supersedes edge creation."""

    def test_supersedes_edge_direction_is_newer_to_older(self):
        """Test that supersedes edges go from newer value to older value."""
        evidence = [
            _make_evidence("decided $10 for pricing", "100"),
            _make_evidence("changed pricing to $20", "200"),
        ]
        graph = build_graph(evidence)

        supersedes_edges = [e for e in graph.edges if e.predicate == "supersedes"]
        assert len(supersedes_edges) >= 1

        # The newer value ($20) should supersede the older value ($10)
        supersedes_edge = supersedes_edges[0]
        assert supersedes_edge.subject == "value:$20"
        assert supersedes_edge.obj == "value:$10"

    def test_supersedes_is_idempotent(self):
        """Test that calling build_supersedes multiple times doesn't duplicate edges."""
        evidence = [
            _make_evidence("decided $10 for pricing", "100"),
            _make_evidence("changed pricing to $20", "200"),
        ]
        graph = build_graph(evidence)

        # Call build_supersedes again
        graph.build_supersedes()

        supersedes_edges = [e for e in graph.edges if e.predicate == "supersedes"]
        assert len(supersedes_edges) == 1


class TestSummary:
    """Tests for summary method."""

    def test_summary_counts_are_correct(self):
        """Test that summary returns correct counts."""
        evidence = [
            _make_evidence("@alice and @bob decided $10 for pricing", "100"),
            _make_evidence("changed pricing to $20", "200"),
        ]
        graph = build_graph(evidence)

        summary = graph.summary()

        assert summary["entities"] >= 4  # topic:pricing, value:$10, value:$20, person:alice, person:bob
        assert summary["topics"] >= 1
        assert summary["values"] == 2
        assert summary["people"] >= 2
        assert summary["decisions"] >= 2
        assert summary["reversals"] == 1
