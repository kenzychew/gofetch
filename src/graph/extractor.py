"""LLM-based entity and relationship extraction for knowledge graph."""

import json

from google import genai
from google.genai import types

from src.config import GraphConfig
from src.exceptions import ExtractionError
from src.logging import get_logger
from src.schemas import Chunk, GraphEntity, GraphRelationship

logger = get_logger(__name__)

EXTRACTION_PROMPT = (
    "Extract entities and relationships from the following text chunks.\n"
    "Return a JSON object with two arrays:\n\n"
    '"entities": [\n'
    '  {{"name": "entity name", "type": "concept|model|technique|'
    'metric|dataset|person|organization"}}\n'
    "]\n"
    '"relationships": [\n'
    '  {{"source": "entity1", "target": "entity2", '
    '"relation": "relationship_type"}}\n'
    "]\n\n"
    "Rules:\n"
    "- Entity names should be lowercase and normalized\n"
    "- Relationship types should be descriptive verbs or phrases "
    "(eg introduced_in, outperforms, based_on, used_by)\n"
    "- Extract at most {max_entities} entities and "
    "{max_relationships} relationships\n"
    "- Focus on technical concepts, methods, and their connections\n"
    "- Only extract entities and relationships explicitly stated "
    "in the text\n\n"
    "Text chunks:\n{chunks_text}\n\n"
    "Return ONLY the JSON object, no other text."
)


async def extract_entities_and_relationships(
    chunks: list[Chunk],
    client: genai.Client,
    config: GraphConfig,
) -> tuple[list[GraphEntity], list[GraphRelationship]]:
    """Extract entities and relationships from a batch of chunks.

    Uses the LLM to identify technical entities (concepts, models,
    techniques) and their relationships from chunk text. Returns
    structured data for knowledge graph construction.

    Args:
        chunks: Batch of chunks to extract from.
        client: Google GenAI client instance.
        config: Graph configuration with extraction settings.

    Returns:
        Tuple of (entities list, relationships list).

    Raises:
        ExtractionError: If extraction fails or returns invalid data.
    """
    chunk_ids = [c.chunk_id for c in chunks]
    chunks_text = "\n\n---\n\n".join(f"[Chunk {c.chunk_id}]: {c.text}" for c in chunks)

    prompt = EXTRACTION_PROMPT.format(
        max_entities=config.max_entities_per_chunk * len(chunks),
        max_relationships=config.max_relationships_per_chunk * len(chunks),
        chunks_text=chunks_text,
    )

    try:
        response = client.models.generate_content(
            model=config.extraction_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=4096,
                response_mime_type="application/json",
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
    except Exception as exc:
        raise ExtractionError(f"LLM extraction call failed: {exc}") from exc

    content = response.text or ""
    if not content.strip() or content.strip() in ("{}", "[]"):
        logger.warning("Empty extraction response", chunks=len(chunks))
        return [], []

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        logger.warning("Failed to parse extraction JSON, skipping batch", preview=content[:200])
        return [], []

    # Parse entities
    entities: list[GraphEntity] = []
    for raw_entity in data.get("entities", []):
        name = str(raw_entity.get("name", "")).strip().lower()
        entity_type = str(raw_entity.get("type", "concept")).strip().lower()
        if name:
            entities.append(
                GraphEntity(
                    name=name,
                    entity_type=entity_type,
                    chunk_ids=list(chunk_ids),
                )
            )

    # Parse relationships
    relationships: list[GraphRelationship] = []
    for raw_rel in data.get("relationships", []):
        source = str(raw_rel.get("source", "")).strip().lower()
        target = str(raw_rel.get("target", "")).strip().lower()
        relation = str(raw_rel.get("relation", "related_to")).strip().lower()
        if source and target:
            relationships.append(
                GraphRelationship(
                    source=source,
                    target=target,
                    relation=relation,
                    chunk_ids=list(chunk_ids),
                )
            )

    logger.info(
        "Extracted entities and relationships",
        chunks=len(chunks),
        entities=len(entities),
        relationships=len(relationships),
    )
    return entities, relationships
