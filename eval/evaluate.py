"""Evaluation framework for retrieval and generation metrics."""

import asyncio
import json
import os
import time
from pathlib import Path

import asyncpg
from pgvector.asyncpg import register_vector

from src.config import AppConfig, GenerationConfig, IngestionConfig, RetrievalConfig
from src.ingestion.embedder import Embedder
from src.ingestion.indexer import BM25Indexer
from src.logging import get_logger, setup_logging
from src.retrieval.dense import DenseRetriever
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.sparse import SparseRetriever
from src.schemas import RetrievalResult

logger = get_logger(__name__)


def load_questions(path: str = "eval/questions.json") -> list[dict[str, str | list[str]]]:
    """Load evaluation questions from JSON file.

    Args:
        path: Path to the questions JSON file.

    Returns:
        List of question dictionaries.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compute_hit_rate(
    results: list[RetrievalResult],
    expected_source: str,
    k: int,
) -> float:
    """Compute Hit@K -- whether the expected source appears in top K results.

    Args:
        results: Ranked retrieval results.
        expected_source: Expected source document name.
        k: Number of top results to check.

    Returns:
        1.0 if hit, 0.0 if miss.
    """
    if expected_source in ("none", "multiple"):
        return -1.0  # Skip metric for unanswerable/multi-source

    for result in results[:k]:
        if expected_source in result.chunk.source:
            return 1.0
    return 0.0


def compute_mrr(
    results: list[RetrievalResult],
    expected_source: str,
) -> float:
    """Compute Mean Reciprocal Rank for the expected source.

    Args:
        results: Ranked retrieval results.
        expected_source: Expected source document name.

    Returns:
        Reciprocal rank (1/rank) or 0.0 if not found. -1.0 if skipped.
    """
    if expected_source in ("none", "multiple"):
        return -1.0

    for i, result in enumerate(results, start=1):
        if expected_source in result.chunk.source:
            return 1.0 / i
    return 0.0


def compute_keyword_recall(
    results: list[RetrievalResult],
    expected_keywords: list[str],
    k: int = 5,
) -> float:
    """Compute what fraction of expected keywords appear in the top K chunks.

    Args:
        results: Ranked retrieval results.
        expected_keywords: Keywords expected to appear in relevant chunks.
        k: Number of top results to check.

    Returns:
        Fraction of keywords found (0.0 to 1.0).
    """
    if not expected_keywords:
        return -1.0

    combined_text = " ".join(r.chunk.text.lower() for r in results[:k])
    found = sum(1 for kw in expected_keywords if kw.lower() in combined_text)
    return found / len(expected_keywords)


async def evaluate_retrieval(
    config: AppConfig,
    questions: list[dict[str, str | list[str]]],
) -> dict[str, list[dict[str, float | str]]]:
    """Run retrieval evaluation across all questions.

    Tests multiple retrieval configurations for ablation:
    dense only, sparse only, hybrid (RRF), hybrid + rerank.

    Args:
        config: Application configuration.
        questions: List of evaluation questions.

    Returns:
        Dictionary mapping configuration name to list of per-question metrics.
    """
    embedder = Embedder(config.ingestion)

    # Load BM25 index
    bm25_indexer = BM25Indexer(config.bm25_index_path)
    bm25, bm25_chunks = bm25_indexer.load_index()
    sparse_retriever = SparseRetriever(bm25, bm25_chunks)

    # Initialize asyncpg pool and dense retriever
    database_url = os.environ.get("DATABASE_URL", config.database_url)

    async def _register_pgvector(conn: asyncpg.Connection) -> None:
        """Register pgvector codec on new pool connections."""
        await register_vector(conn)

    pool = await asyncpg.create_pool(
        dsn=database_url, min_size=1, max_size=5, init=_register_pgvector
    )
    dense_retriever = DenseRetriever(pool, config)

    # Initialize reranker
    reranker = CrossEncoderReranker(config.retrieval)

    configurations = {
        "Dense only": {"dense": True, "sparse": False, "rerank": False},
        "BM25 only": {"dense": False, "sparse": True, "rerank": False},
        "Hybrid (RRF)": {"dense": True, "sparse": True, "rerank": False},
        "Hybrid + Rerank": {"dense": True, "sparse": True, "rerank": True},
    }

    all_results: dict[str, list[dict[str, float | str]]] = {}

    for config_name, flags in configurations.items():
        logger.info("Evaluating configuration", config=config_name)
        config_results: list[dict[str, float | str]] = []

        for q in questions:
            question = str(q["question"])
            expected_source = str(q.get("expected_source", ""))
            expected_keywords = q.get("expected_keywords", [])
            if not isinstance(expected_keywords, list):
                expected_keywords = []

            # Embed query
            query_embedding = embedder.embed_query(question)

            ranked_lists: list[list[RetrievalResult]] = []

            if flags["dense"]:
                dense_retriever.set_query_embedding(query_embedding)
                dense_results = await dense_retriever.retrieve(question, 20)
                ranked_lists.append(dense_results)

            if flags["sparse"]:
                sparse_results = await sparse_retriever.retrieve(question, 20)
                ranked_lists.append(sparse_results)

            # Fuse
            if len(ranked_lists) > 1:
                fused = reciprocal_rank_fusion(ranked_lists, k=60, top_k=10)
            elif ranked_lists:
                fused = ranked_lists[0][:10]
            else:
                fused = []

            # Optionally rerank
            if flags["rerank"] and fused:
                final = await reranker.rerank(question, fused, 5)
            else:
                final = fused[:5]

            # Compute metrics
            hit1 = compute_hit_rate(final, expected_source, 1)
            hit3 = compute_hit_rate(final, expected_source, 3)
            hit5 = compute_hit_rate(final, expected_source, 5)
            mrr = compute_mrr(final, expected_source)
            kw_recall = compute_keyword_recall(final, expected_keywords)

            config_results.append(
                {
                    "question": question[:80],
                    "hit@1": hit1,
                    "hit@3": hit3,
                    "hit@5": hit5,
                    "mrr": mrr,
                    "keyword_recall": kw_recall,
                }
            )

        all_results[config_name] = config_results

    await pool.close()
    return all_results


def _avg_metric(key: str, rows: list[dict[str, float | str]]) -> float:
    """Compute the average of a metric column, skipping non-float and negative values.

    Args:
        key: Metric column name.
        rows: List of per-question metric dictionaries.

    Returns:
        Average value, or 0.0 if no valid values exist.
    """
    vals: list[float] = []
    for r in rows:
        v = r[key]
        if isinstance(v, float) and v >= 0:
            vals.append(v)
    return sum(vals) / len(vals) if vals else 0.0


def print_ablation_table(results: dict[str, list[dict[str, float | str]]]) -> None:
    """Print a formatted ablation table of retrieval metrics.

    Args:
        results: Dictionary mapping configuration name to per-question metrics.
    """
    print("\n" + "=" * 80)
    print("RETRIEVAL ABLATION TABLE")
    print("=" * 80)

    cols = ["Configuration", "Hit@1", "Hit@3", "Hit@5", "MRR", "KW Rec"]
    header = f"{cols[0]:<25} " + " ".join(f"{c:>7}" for c in cols[1:])
    print(header)
    print("-" * 80)

    for config_name, config_results in results.items():
        valid = 0
        for r in config_results:
            v = r["hit@1"]
            if isinstance(v, float) and v >= 0:
                valid += 1
        total = len(config_results)

        print(
            f"{config_name:<25} "
            f"{_avg_metric('hit@1', config_results):>7.3f} "
            f"{_avg_metric('hit@3', config_results):>7.3f} "
            f"{_avg_metric('hit@5', config_results):>7.3f} "
            f"{_avg_metric('mrr', config_results):>7.3f} "
            f"{_avg_metric('keyword_recall', config_results):>7.3f}"
        )
        print(f"  (evaluated {valid}/{total} questions, skipped unanswerable/multi-source)")

    print("=" * 80)


async def main() -> None:
    """Run the full evaluation pipeline."""
    setup_logging(level="INFO")

    config = AppConfig(
        ingestion=IngestionConfig(),
        retrieval=RetrievalConfig(),
        generation=GenerationConfig(),
    )

    questions = load_questions()
    logger.info("Loaded evaluation questions", count=len(questions))

    t0 = time.perf_counter()
    results = await evaluate_retrieval(config, questions)
    elapsed = time.perf_counter() - t0

    print_ablation_table(results)
    logger.info("Evaluation complete", elapsed_s=round(elapsed, 1))

    # Save raw results
    output_path = Path("eval/results.json")
    output_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    logger.info("Saved results", path=str(output_path))


if __name__ == "__main__":
    asyncio.run(main())
