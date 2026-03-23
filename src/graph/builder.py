"""NetworkX graph construction with entity normalization and JSON persistence."""

import json
from difflib import SequenceMatcher
from pathlib import Path

import networkx as nx

from src.config import GraphConfig
from src.logging import get_logger
from src.schemas import GraphEntity, GraphRelationship

logger = get_logger(__name__)


class KnowledgeGraph:
    """In-memory knowledge graph backed by NetworkX.

    Stores entities as nodes and relationships as edges. Supports
    entity normalization (merging similar names), JSON persistence,
    and chunk-aware traversal for retrieval.

    Attributes:
        graph: The underlying NetworkX directed graph.
        config: Graph configuration.
        entity_to_chunks: Mapping from entity name to associated chunk IDs.
    """

    def __init__(self, config: GraphConfig) -> None:
        """Initialize the knowledge graph.

        Args:
            config: Graph configuration with normalization settings.
        """
        self.graph: nx.DiGraph = nx.DiGraph()
        self.config = config
        self.entity_to_chunks: dict[str, set[str]] = {}

    def _normalize_entity(self, name: str) -> str:
        """Normalize an entity name by finding existing similar entities.

        Uses SequenceMatcher to find entities above the similarity threshold.
        Returns the existing entity name if a match is found.

        Args:
            name: Raw entity name to normalize.

        Returns:
            Normalized entity name (may be an existing entity).
        """
        for existing in self.graph.nodes():
            ratio = SequenceMatcher(None, name, existing).ratio()
            if ratio >= self.config.entity_similarity_threshold:
                return str(existing)
        return name

    def add_entities(self, entities: list[GraphEntity]) -> None:
        """Add entities to the graph as nodes.

        Normalizes entity names and merges chunk IDs for duplicates.

        Args:
            entities: List of entities to add.
        """
        for entity in entities:
            normalized = self._normalize_entity(entity.name)
            if not self.graph.has_node(normalized):
                self.graph.add_node(normalized, entity_type=entity.entity_type)
                self.entity_to_chunks[normalized] = set()
            self.entity_to_chunks[normalized].update(entity.chunk_ids)

    def add_relationships(self, relationships: list[GraphRelationship]) -> None:
        """Add relationships to the graph as directed edges.

        Normalizes entity names in source/target. Creates nodes for
        entities not yet in the graph.

        Args:
            relationships: List of relationships to add.
        """
        for rel in relationships:
            source = self._normalize_entity(rel.source)
            target = self._normalize_entity(rel.target)

            # Ensure nodes exist
            if not self.graph.has_node(source):
                self.graph.add_node(source, entity_type="unknown")
                self.entity_to_chunks[source] = set()
            if not self.graph.has_node(target):
                self.graph.add_node(target, entity_type="unknown")
                self.entity_to_chunks[target] = set()

            self.entity_to_chunks[source].update(rel.chunk_ids)
            self.entity_to_chunks[target].update(rel.chunk_ids)

            self.graph.add_edge(source, target, relation=rel.relation)

    def get_related_chunk_ids(self, query_terms: list[str], hops: int = 1) -> list[str]:
        """Find chunk IDs related to query terms via graph traversal.

        Matches query terms against entity names, then traverses
        the graph to find related entities and their chunks.

        Args:
            query_terms: Lowercased query terms to match against entities.
            hops: Number of graph traversal hops from matched entities.

        Returns:
            List of chunk IDs ranked by connection count (most connected first).
        """
        # Match query terms to graph entities
        matched_entities: set[str] = set()
        for term in query_terms:
            for node in self.graph.nodes():
                if term in str(node) or str(node) in term:
                    matched_entities.add(str(node))

        if not matched_entities:
            return []

        # Traverse graph to find related entities
        related_entities: set[str] = set(matched_entities)
        frontier = set(matched_entities)

        for _ in range(hops):
            next_frontier: set[str] = set()
            for entity in frontier:
                # Follow edges in both directions
                successors = set(self.graph.successors(entity))
                predecessors = set(self.graph.predecessors(entity))
                neighbors = successors | predecessors
                next_frontier.update(neighbors - related_entities)
            related_entities.update(next_frontier)
            frontier = next_frontier

        # Collect chunk IDs with connection counts
        chunk_counts: dict[str, int] = {}
        for entity in related_entities:
            for chunk_id in self.entity_to_chunks.get(entity, set()):
                chunk_counts[chunk_id] = chunk_counts.get(chunk_id, 0) + 1

        # Sort by count descending
        sorted_chunks = sorted(chunk_counts.keys(), key=lambda c: chunk_counts[c], reverse=True)

        logger.info(
            "Graph traversal",
            matched_entities=len(matched_entities),
            related_entities=len(related_entities),
            chunk_ids=len(sorted_chunks),
        )
        return sorted_chunks

    def save(self, path: str) -> None:
        """Serialize the graph to a JSON file.

        Args:
            path: File path to save the graph data.
        """
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "nodes": [
                {
                    "name": str(node),
                    "entity_type": str(self.graph.nodes[node].get("entity_type", "unknown")),
                    "chunk_ids": list(self.entity_to_chunks.get(str(node), set())),
                }
                for node in self.graph.nodes()
            ],
            "edges": [
                {
                    "source": str(u),
                    "target": str(v),
                    "relation": str(self.graph.edges[u, v].get("relation", "related_to")),
                }
                for u, v in self.graph.edges()
            ],
        }

        file_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info(
            "Saved knowledge graph",
            nodes=len(data["nodes"]),
            edges=len(data["edges"]),
            path=path,
        )

    def load(self, path: str) -> None:
        """Load the graph from a JSON file.

        Args:
            path: File path to load the graph data from.
        """
        file_path = Path(path)
        if not file_path.exists():
            logger.warning("Graph file not found", path=path)
            return

        raw = json.loads(file_path.read_text(encoding="utf-8"))

        self.graph = nx.DiGraph()
        self.entity_to_chunks = {}

        for node_data in raw.get("nodes", []):
            name = node_data["name"]
            self.graph.add_node(name, entity_type=node_data.get("entity_type", "unknown"))
            self.entity_to_chunks[name] = set(node_data.get("chunk_ids", []))

        for edge_data in raw.get("edges", []):
            self.graph.add_edge(
                edge_data["source"],
                edge_data["target"],
                relation=edge_data.get("relation", "related_to"),
            )

        logger.info(
            "Loaded knowledge graph",
            nodes=self.graph.number_of_nodes(),
            edges=self.graph.number_of_edges(),
        )

    @property
    def num_nodes(self) -> int:
        """Return the number of nodes in the graph."""
        return self.graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        """Return the number of edges in the graph."""
        return self.graph.number_of_edges()
