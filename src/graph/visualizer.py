"""Graph visualization for presentation slides and UI."""

from pathlib import Path

import networkx as nx
from pyvis.network import Network

from src.graph.builder import KnowledgeGraph
from src.logging import get_logger

logger = get_logger(__name__)

# Color mapping for entity types
ENTITY_COLORS = {
    "concept": "#4A90D9",
    "model": "#E74C3C",
    "technique": "#27AE60",
    "metric": "#F39C12",
    "dataset": "#8E44AD",
    "person": "#E67E22",
    "organization": "#1ABC9C",
    "unknown": "#95A5A6",
}


def visualize_graph(
    graph: KnowledgeGraph,
    output_path: str = "graph_data/graph.html",
    height: str = "700px",
    width: str = "100%",
) -> str:
    """Generate an interactive HTML visualization of the knowledge graph.

    Uses pyvis to create a force-directed graph layout with color-coded
    entity types and labeled edges.

    Args:
        graph: The knowledge graph to visualize.
        output_path: Path for the output HTML file.
        height: Height of the visualization.
        width: Width of the visualization.

    Returns:
        Path to the generated HTML file.
    """
    net = Network(
        height=height,
        width=width,
        directed=True,
        notebook=False,
        cdn_resources="remote",
    )

    # Add nodes
    for node in graph.graph.nodes():
        entity_type = str(graph.graph.nodes[node].get("entity_type", "unknown"))
        color = ENTITY_COLORS.get(entity_type, ENTITY_COLORS["unknown"])
        chunk_count = len(graph.entity_to_chunks.get(str(node), set()))
        # Scale node size by chunk count
        size = max(15, min(50, 10 + chunk_count * 5))
        net.add_node(
            str(node),
            label=str(node),
            color=color,
            size=size,
            title=f"{node}\nType: {entity_type}\nChunks: {chunk_count}",
        )

    # Add edges
    for source, target in graph.graph.edges():
        relation = str(graph.graph.edges[source, target].get("relation", "related_to"))
        net.add_edge(str(source), str(target), label=relation, title=relation)

    # Physics settings for better layout
    net.set_options("""
    {
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 200,
                "springConstant": 0.08
            },
            "solver": "forceAtlas2Based"
        },
        "edges": {
            "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}},
            "font": {"size": 10, "align": "middle"}
        }
    }
    """)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    net.save_graph(str(output))

    logger.info(
        "Generated graph visualization",
        nodes=graph.num_nodes,
        edges=graph.num_edges,
        output=output_path,
    )
    return str(output)


def get_graph_stats(graph: KnowledgeGraph) -> dict[str, int | list[tuple[str, int]]]:
    """Get summary statistics about the knowledge graph.

    Args:
        graph: The knowledge graph to analyze.

    Returns:
        Dictionary with graph statistics.
    """
    g = graph.graph

    # Most connected entities
    degree_centrality = nx.degree_centrality(g)
    top_entities = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]

    # Entity type counts
    type_counts: dict[str, int] = {}
    for node in g.nodes():
        etype = str(g.nodes[node].get("entity_type", "unknown"))
        type_counts[etype] = type_counts.get(etype, 0) + 1

    return {
        "total_nodes": g.number_of_nodes(),
        "total_edges": g.number_of_edges(),
        "connected_components": nx.number_weakly_connected_components(g),
        "top_entities": [(str(name), round(score, 4)) for name, score in top_entities],
        "entity_type_counts": type_counts,
    }
