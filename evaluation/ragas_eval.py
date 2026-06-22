"""RAGAS evaluation script comparing CRAG pipeline against naive RAG baseline.

Run: ``python -m evaluation.ragas_eval``
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from datasets import Dataset
from langchain_core.documents import Document
from ragas import evaluate as ragas_evaluate
from ragas.metrics import answer_relevancy, context_precision, faithfulness

from crag.generator import AnswerGenerator
from crag.graph.graph import build_graph
from crag.graph.state import make_initial_state
from crag.retriever import HybridRetriever
from evaluation.dataset import EVAL_DATASET

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

RESULTS_PATH = Path(__file__).parent / "results.json"


def _run_crag_pipeline(questions: list[str]) -> list[dict[str, Any]]:
    """Execute each question through the full CRAG graph."""
    graph = build_graph()
    results: list[dict[str, Any]] = []

    for question in questions:
        logger.info("CRAG: processing '%s'", question)
        state = graph.invoke(make_initial_state(question))
        contexts = [
            g["doc"].page_content
            for g in state.get("graded_docs", [])
            if g["score"] in ("RELEVANT", "AMBIGUOUS")
        ]
        results.append(
            {
                "question": question,
                "answer": state.get("answer", ""),
                "contexts": contexts if contexts else ["No relevant context found."],
            }
        )
    return results


def _run_naive_rag(questions: list[str]) -> list[dict[str, Any]]:
    """Baseline: retrieve → generate with no grading or correction."""
    retriever = HybridRetriever()
    generator = AnswerGenerator()
    results: list[dict[str, Any]] = []

    for question in questions:
        logger.info("Naive RAG: processing '%s'", question)
        docs: list[Document] = retriever.retrieve(question)
        context = "\n\n".join(d.page_content for d in docs)
        answer = generator.generate(question, context) if context.strip() else "No context available."
        results.append(
            {
                "question": question,
                "answer": answer,
                "contexts": [d.page_content for d in docs] if docs else ["No context found."],
            }
        )
    return results


def _build_ragas_dataset(
    run_results: list[dict[str, Any]],
    ground_truths: list[str],
) -> Dataset:
    """Convert run results into a HuggingFace Dataset for RAGAS."""
    return Dataset.from_dict(
        {
            "question": [r["question"] for r in run_results],
            "answer": [r["answer"] for r in run_results],
            "contexts": [r["contexts"] for r in run_results],
            "ground_truth": ground_truths,
        }
    )


def _print_comparison(
    crag_scores: dict[str, float],
    naive_scores: dict[str, float],
) -> None:
    """Pretty-print a side-by-side metrics table."""
    header = f"{'Metric':<25} {'CRAG':>10} {'Naive RAG':>10} {'Delta':>10}"
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for metric in ["faithfulness", "answer_relevancy", "context_precision"]:
        c = crag_scores.get(metric, 0.0)
        n = naive_scores.get(metric, 0.0)
        delta = c - n
        sign = "+" if delta >= 0 else ""
        print(f"{metric:<25} {c:>10.4f} {n:>10.4f} {sign}{delta:>9.4f}")
    print(sep)


def main() -> None:
    """Run the full evaluation pipeline."""
    questions = [item["question"] for item in EVAL_DATASET]
    ground_truths = [item["ground_truth"] for item in EVAL_DATASET]

    metrics = [faithfulness, answer_relevancy, context_precision]

    logger.info("Running CRAG pipeline on %d questions", len(questions))
    crag_results = _run_crag_pipeline(questions)
    crag_dataset = _build_ragas_dataset(crag_results, ground_truths)
    logger.info("Evaluating CRAG with RAGAS")
    crag_eval = ragas_evaluate(crag_dataset, metrics=metrics)
    crag_scores = {k: float(v) for k, v in crag_eval.items() if isinstance(v, (int, float))}

    logger.info("Running naive RAG baseline on %d questions", len(questions))
    naive_results = _run_naive_rag(questions)
    naive_dataset = _build_ragas_dataset(naive_results, ground_truths)
    logger.info("Evaluating naive RAG with RAGAS")
    naive_eval = ragas_evaluate(naive_dataset, metrics=metrics)
    naive_scores = {k: float(v) for k, v in naive_eval.items() if isinstance(v, (int, float))}

    _print_comparison(crag_scores, naive_scores)

    output = {
        "crag": {"scores": crag_scores, "results": crag_results},
        "naive_rag": {"scores": naive_scores, "results": naive_results},
    }
    RESULTS_PATH.write_text(json.dumps(output, indent=2, default=str))
    logger.info("Results saved to %s", RESULTS_PATH)


if __name__ == "__main__":
    main()
