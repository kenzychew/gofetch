"""LLM-as-judge evaluation for generation quality (faithfulness and relevancy)."""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import asyncpg
from google import genai
from google.genai import types
from omegaconf import OmegaConf
from pgvector.asyncpg import register_vector
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import (
    AppConfig,
    GenerationConfig,
    IngestionConfig,
    PromptConfig,
    RetrievalConfig,
)
from src.generation.prompt import PromptBuilder
from src.ingestion.embedder import Embedder
from src.ingestion.indexer import BM25Indexer
from src.logging import get_logger, setup_logging
from src.retrieval.dense import DenseRetriever
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.sparse import SparseRetriever
from src.schemas import RetrievalResult

logger = get_logger(__name__)

FAITHFULNESS_PROMPT = """\
You are an impartial judge evaluating a RAG system's answer.

Score how well the ANSWER is supported by the CONTEXT on a 1-5 scale:
5 = Every claim is directly supported by the context
4 = Almost all claims supported, minor unsupported detail
3 = Core answer supported but some claims lack evidence
2 = Significant claims not grounded in context
1 = Answer is fabricated or contradicts context

CONTEXT:
{context}

ANSWER:
{answer}

Return JSON: {{"score": <int 1-5>, "reasoning": "<one sentence>"}}"""

RELEVANCY_PROMPT = """\
You are an impartial judge evaluating a RAG system's answer.

Score how well the ANSWER addresses the QUESTION on a 1-5 scale:
5 = Fully answers the question with specific detail
4 = Mostly answers with minor gaps
3 = Partially answers, missing key aspects
2 = Tangentially related but does not answer
1 = Completely irrelevant or off-topic

QUESTION:
{question}

ANSWER:
{answer}

Return JSON: {{"score": <int 1-5>, "reasoning": "<one sentence>"}}"""


def load_prompt_config() -> PromptConfig:
    """Load prompt configuration from the citation YAML file.

    Returns:
        Populated PromptConfig instance.
    """
    path = Path("configs/prompts/citation.yaml")
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict):
        return PromptConfig()

    few_shot = raw.get("few_shot_examples", [])
    return PromptConfig(
        system_prompt=str(raw.get("system_prompt", "")),
        context_template=str(raw.get("context_template", "")),
        chunk_template=str(raw.get("chunk_template", "")),
        few_shot_examples=few_shot if isinstance(few_shot, list) else [],
        low_confidence_warning=str(raw.get("low_confidence_warning", "")),
    )


def load_questions(path: str = "eval/questions.json") -> list[dict[str, str | list[str]]]:
    """Load evaluation questions from JSON file.

    Args:
        path: Path to the questions JSON file.

    Returns:
        List of question dictionaries.
    """
    data: list[dict[str, str | list[str]]] = json.loads(
        Path(path).read_text(encoding="utf-8")
    )
    return data


RETRY_DECORATOR = retry(
    wait=wait_exponential(multiplier=3, min=10, max=120),
    stop=stop_after_attempt(6),
    reraise=True,
)


@RETRY_DECORATOR
async def generate_answer(
    question: str,
    results: list[RetrievalResult],
    prompt_builder: PromptBuilder,
    client: genai.Client,
    config: GenerationConfig,
) -> tuple[str, str]:
    """Generate an answer from retrieval results using the full RAG pipeline.

    Args:
        question: The user's question.
        results: Retrieved and re-ranked chunks.
        prompt_builder: Citation-aware prompt builder.
        client: Google GenAI client.
        config: Generation configuration.

    Returns:
        Tuple of (generated answer text, formatted context string).
    """
    messages = prompt_builder.build_messages(question, results)
    system_msg = ""
    user_msg = ""
    for msg in messages:
        if msg["role"] == "system":
            system_msg = msg["content"]
        elif msg["role"] == "user":
            user_msg = msg["content"]
    response = client.models.generate_content(
        model=config.model,
        contents=user_msg,
        config=types.GenerateContentConfig(
            system_instruction=system_msg if system_msg else None,
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    answer = response.text or ""
    context = prompt_builder.format_chunks(results)
    return answer, context


@RETRY_DECORATOR
async def judge_faithfulness(
    context: str,
    answer: str,
    client: genai.Client,
    judge_model: str,
) -> tuple[int, str]:
    """Score how well the answer is grounded in the provided context.

    Args:
        context: The formatted context chunks.
        answer: The generated answer.
        client: Google GenAI client.
        judge_model: Model name for the judge.

    Returns:
        Tuple of (score 1-5, reasoning string).
    """
    prompt = FAITHFULNESS_PROMPT.format(context=context, answer=answer)
    response = client.models.generate_content(
        model=judge_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    raw = response.text or "{}"
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Failed to parse faithfulness JSON", raw=raw[:200])
        return 0, "parse error"
    return int(result.get("score", 0)), str(result.get("reasoning", ""))


@RETRY_DECORATOR
async def judge_relevancy(
    question: str,
    answer: str,
    client: genai.Client,
    judge_model: str,
) -> tuple[int, str]:
    """Score how well the answer addresses the question.

    Args:
        question: The original question.
        answer: The generated answer.
        client: Google GenAI client.
        judge_model: Model name for the judge.

    Returns:
        Tuple of (score 1-5, reasoning string).
    """
    prompt = RELEVANCY_PROMPT.format(question=question, answer=answer)
    response = client.models.generate_content(
        model=judge_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    raw = response.text or "{}"
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Failed to parse relevancy JSON", raw=raw[:200])
        return 0, "parse error"
    return int(result.get("score", 0)), str(result.get("reasoning", ""))


async def evaluate_generation(
    config: AppConfig,
    prompt_config: PromptConfig,
    questions: list[dict[str, str | list[str]]],
    judge_model: str,
) -> dict[str, list[dict[str, str | int]] | dict[str, float | int]]:
    """Run end-to-end generation evaluation with LLM-as-judge scoring.

    For each question, runs Hybrid + Rerank retrieval, generates an answer,
    then scores faithfulness and relevancy using a judge LLM.

    Args:
        config: Application configuration.
        prompt_config: Prompt template configuration.
        questions: List of evaluation question dicts.
        judge_model: Gemini model name for the judge.

    Returns:
        Dict with "per_question" results list and "summary" aggregates.
    """
    embedder = Embedder(config.ingestion)
    prompt_builder = PromptBuilder(prompt_config, config.generation)

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
    reranker = CrossEncoderReranker(config.retrieval)

    # Initialize Gemini client
    gcp_project = os.environ.get("GCP_PROJECT", config.gcp_project)
    gcp_region = os.environ.get("GCP_REGION", config.gcp_region)
    client = genai.Client(vertexai=True, project=gcp_project, location=gcp_region)

    per_question: list[dict[str, str | int]] = []
    total_faithfulness = 0
    total_relevancy = 0
    scored_count = 0

    for i, q in enumerate(questions):
        question = str(q["question"])
        expected_source = str(q.get("expected_source", ""))

        # Skip unanswerable/multi-source questions
        if expected_source in ("none", "multiple"):
            logger.info("Skipping unanswerable question", index=i, question=question[:60])
            continue

        logger.info("Evaluating question", index=i, question=question[:60])

        # Retrieve: Hybrid + Rerank (production config)
        query_embedding = embedder.embed_query(question)
        dense_results = await dense_retriever.retrieve(question, 20, query_embedding)
        sparse_results = await sparse_retriever.retrieve(question, 20)
        fused = reciprocal_rank_fusion([dense_results, sparse_results], k=60, top_k=10)
        final = await reranker.rerank(question, fused, 5)

        # Generate answer
        answer, context = await generate_answer(
            question, final, prompt_builder, client, config.generation
        )

        # Judge
        faith_score, faith_reason = await judge_faithfulness(
            context, answer, client, judge_model
        )
        rel_score, rel_reason = await judge_relevancy(
            question, answer, client, judge_model
        )

        per_question.append(
            {
                "question": question,
                "answer": answer[:500],
                "faithfulness_score": faith_score,
                "faithfulness_reasoning": faith_reason,
                "relevancy_score": rel_score,
                "relevancy_reasoning": rel_reason,
            }
        )

        total_faithfulness += faith_score
        total_relevancy += rel_score
        scored_count += 1

        logger.info(
            "Scored question",
            index=i,
            faithfulness=faith_score,
            relevancy=rel_score,
        )

        # Rate-limit to avoid hitting Gemini quotas (3 API calls per question)
        await asyncio.sleep(10)

    await pool.close()

    summary: dict[str, float | int] = {
        "avg_faithfulness": round(total_faithfulness / scored_count, 2) if scored_count else 0.0,
        "avg_relevancy": round(total_relevancy / scored_count, 2) if scored_count else 0.0,
        "total_questions": len(questions),
        "scored_questions": scored_count,
    }

    return {"per_question": per_question, "summary": summary}


def print_generation_table(
    results: dict[str, list[dict[str, str | int]] | dict[str, float | int]],
) -> None:
    """Print a formatted summary of generation evaluation results.

    Args:
        results: Dictionary with per_question and summary keys.
    """
    summary = results["summary"]
    per_question = results["per_question"]
    if not isinstance(per_question, list):
        return

    print("\n" + "=" * 80)
    print("GENERATION QUALITY (LLM-AS-JUDGE)")
    print("=" * 80)

    cols = ["#", "Faithfulness", "Relevancy", "Question"]
    print(f"{cols[0]:<4} {cols[1]:<14} {cols[2]:<11} {cols[3]}")
    print("-" * 80)

    for i, row in enumerate(per_question, start=1):
        print(
            f"{i:<4} {row['faithfulness_score']:<14} {row['relevancy_score']:<11} "
            f"{str(row['question'])[:50]}"
        )

    print("-" * 80)

    if isinstance(summary, dict):
        print(
            f"{'AVG':<4} {summary.get('avg_faithfulness', 0):<14} "
            f"{summary.get('avg_relevancy', 0):<11} "
            f"({summary.get('scored_questions', 0)}/{summary.get('total_questions', 0)} scored)"
        )

    print("=" * 80)


async def main() -> None:
    """Run the full generation evaluation pipeline."""
    parser = argparse.ArgumentParser(description="Run generation quality evaluation.")
    parser.add_argument(
        "--output",
        type=str,
        default="eval/results_generation.json",
        help="Path for the output JSON results file.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gemini-2.5-flash",
        help="Gemini model name for the judge.",
    )
    args = parser.parse_args()

    setup_logging(level="INFO")

    config = AppConfig(
        ingestion=IngestionConfig(),
        retrieval=RetrievalConfig(),
        generation=GenerationConfig(),
    )
    prompt_config = load_prompt_config()

    questions = load_questions()
    logger.info("Loaded evaluation questions", count=len(questions))

    t0 = time.perf_counter()
    results = await evaluate_generation(config, prompt_config, questions, args.judge_model)
    elapsed = time.perf_counter() - t0

    print_generation_table(results)
    logger.info("Generation evaluation complete", elapsed_s=round(elapsed, 1))

    output_path = Path(args.output)
    output_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    logger.info("Saved results", path=str(output_path))


if __name__ == "__main__":
    asyncio.run(main())
