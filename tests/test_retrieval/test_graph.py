"""Tests for knowledge graph builder and retriever."""

from src.config import GraphConfig
from src.graph.builder import KnowledgeGraph
from src.schemas import GraphEntity, GraphRelationship


def _make_graph_config() -> GraphConfig:
    """Create a test graph config.

    Returns:
        GraphConfig with test values.
    """
    return GraphConfig(entity_similarity_threshold=0.85, traversal_hops=1)


def test_add_entities() -> None:
    """Adding entities should create nodes in the graph."""
    graph = KnowledgeGraph(_make_graph_config())
    entities = [
        GraphEntity(name="attention", entity_type="concept", chunk_ids=["c1"]),
        GraphEntity(name="transformer", entity_type="model", chunk_ids=["c2"]),
    ]
    graph.add_entities(entities)

    assert graph.num_nodes == 2
    assert "attention" in graph.graph.nodes()
    assert "transformer" in graph.graph.nodes()


def test_add_relationships() -> None:
    """Adding relationships should create edges in the graph."""
    graph = KnowledgeGraph(_make_graph_config())
    entities = [
        GraphEntity(name="attention", entity_type="concept", chunk_ids=["c1"]),
        GraphEntity(name="transformer", entity_type="model", chunk_ids=["c2"]),
    ]
    graph.add_entities(entities)

    rels = [
        GraphRelationship(
            source="attention",
            target="transformer",
            relation="used_in",
            chunk_ids=["c1"],
        )
    ]
    graph.add_relationships(rels)

    assert graph.num_edges == 1


def test_entity_normalization() -> None:
    """Similar entity names should be merged into one node."""
    config = GraphConfig(entity_similarity_threshold=0.8)
    graph = KnowledgeGraph(config)

    entities = [
        GraphEntity(name="self-attention", entity_type="concept", chunk_ids=["c1"]),
        GraphEntity(name="self attention", entity_type="concept", chunk_ids=["c2"]),
    ]
    graph.add_entities(entities)

    # "self-attention" and "self attention" should be merged
    assert graph.num_nodes == 1


def test_chunk_id_tracking() -> None:
    """Entity-to-chunk mapping should track all associated chunks."""
    graph = KnowledgeGraph(_make_graph_config())
    entities = [
        GraphEntity(name="attention", entity_type="concept", chunk_ids=["c1", "c2"]),
    ]
    graph.add_entities(entities)

    assert graph.entity_to_chunks["attention"] == {"c1", "c2"}


def test_graph_traversal_finds_related_chunks() -> None:
    """Graph traversal should find chunks connected to matching entities."""
    graph = KnowledgeGraph(_make_graph_config())

    entities = [
        GraphEntity(name="attention", entity_type="concept", chunk_ids=["c1"]),
        GraphEntity(name="transformer", entity_type="model", chunk_ids=["c2"]),
        GraphEntity(name="bert", entity_type="model", chunk_ids=["c3"]),
    ]
    graph.add_entities(entities)

    rels = [
        GraphRelationship(
            source="attention", target="transformer", relation="used_in", chunk_ids=["c1"]
        ),
        GraphRelationship(source="attention", target="bert", relation="used_in", chunk_ids=["c1"]),
    ]
    graph.add_relationships(rels)

    chunk_ids = graph.get_related_chunk_ids(["attention"], hops=1)

    # Should find chunks from attention + its neighbors (transformer, bert)
    assert "c1" in chunk_ids
    assert "c2" in chunk_ids
    assert "c3" in chunk_ids


def test_graph_traversal_no_match() -> None:
    """Traversal with unmatched terms should return empty list."""
    graph = KnowledgeGraph(_make_graph_config())
    entities = [GraphEntity(name="attention", entity_type="concept", chunk_ids=["c1"])]
    graph.add_entities(entities)

    chunk_ids = graph.get_related_chunk_ids(["unrelated_term"], hops=1)
    assert chunk_ids == []


def test_save_and_load(tmp_path: str) -> None:
    """Graph should survive serialization and deserialization."""
    config = _make_graph_config()
    graph = KnowledgeGraph(config)

    entities = [
        GraphEntity(name="attention", entity_type="concept", chunk_ids=["c1"]),
        GraphEntity(name="transformer", entity_type="model", chunk_ids=["c2"]),
    ]
    graph.add_entities(entities)

    rels = [
        GraphRelationship(
            source="attention", target="transformer", relation="used_in", chunk_ids=["c1"]
        )
    ]
    graph.add_relationships(rels)

    save_path = f"{tmp_path}/graph.json"
    graph.save(save_path)

    # Load into new graph
    loaded = KnowledgeGraph(config)
    loaded.load(save_path)

    assert loaded.num_nodes == 2
    assert loaded.num_edges == 1
    assert "attention" in loaded.graph.nodes()


def test_bidirectional_traversal() -> None:
    """Traversal should follow edges in both directions."""
    graph = KnowledgeGraph(_make_graph_config())

    entities = [
        GraphEntity(name="a", entity_type="concept", chunk_ids=["c1"]),
        GraphEntity(name="b", entity_type="concept", chunk_ids=["c2"]),
    ]
    graph.add_entities(entities)

    # Edge from a -> b
    rels = [GraphRelationship(source="a", target="b", relation="related", chunk_ids=["c1"])]
    graph.add_relationships(rels)

    # Searching for "b" should find "a" via reverse traversal
    chunk_ids = graph.get_related_chunk_ids(["b"], hops=1)
    assert "c1" in chunk_ids
    assert "c2" in chunk_ids
