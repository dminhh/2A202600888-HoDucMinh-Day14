"""
CI/CD Quality Gate — Day 14 AI Evaluation Pipeline
Runs benchmark on golden dataset and blocks deploy if any metric falls below threshold.

Exit codes:
    0 — all metrics pass, safe to deploy
    1 — one or more metrics below threshold, block deploy
"""

import json
import sys
from pathlib import Path

# Make solution importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "solution"))
from solution import (
    QAPair, RAGASEvaluator, BenchmarkRunner, FailureAnalyzer,
)

# ---------------------------------------------------------------------------
# Thresholds (block deploy if average metric falls below these)
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "faithfulness": 0.40,   # hallucination risk — strict
    "relevance":    0.35,   # off-topic risk — moderate
    "completeness": 0.50,   # missing info risk — moderate
}

# ---------------------------------------------------------------------------
# Golden dataset (mini — use full 20 pairs in production)
# ---------------------------------------------------------------------------
GOLDEN_DATASET = [
    QAPair(
        "What is RAG?",
        "RAG stands for Retrieval-Augmented Generation, combining retrieval of documents with LLM text generation.",
        "RAG (Retrieval-Augmented Generation) is a technique that retrieves relevant documents and uses them to ground LLM outputs.",
        {"difficulty": "easy", "id": "E01"},
    ),
    QAPair(
        "What does LLM stand for?",
        "LLM stands for Large Language Model, a deep learning model trained on large text corpora.",
        "LLMs (Large Language Models) are transformer-based models trained on massive text datasets to generate language.",
        {"difficulty": "easy", "id": "E02"},
    ),
    QAPair(
        "What is a vector database?",
        "A vector database stores embeddings (numerical representations of text) and enables fast similarity search.",
        "Vector databases like Pinecone and Weaviate store high-dimensional embeddings and support similarity search.",
        {"difficulty": "easy", "id": "E03"},
    ),
    QAPair(
        "What is a prompt?",
        "A prompt is the input text given to an LLM to instruct it on what to generate.",
        "A prompt is the instruction or question sent to a language model. Prompt engineering optimizes this input.",
        {"difficulty": "easy", "id": "E04"},
    ),
    QAPair(
        "What is hallucination in LLMs?",
        "Hallucination is when an LLM generates factually incorrect or fabricated information not grounded in the context.",
        "LLM hallucination refers to confident generation of false or unsupported facts. RAG helps reduce hallucination.",
        {"difficulty": "easy", "id": "E05"},
    ),
    QAPair(
        "How does RAG reduce hallucination?",
        "RAG reduces hallucination by grounding the LLM response in retrieved documents, limiting generation to supported facts.",
        "RAG retrieves relevant documents and injects them as context. The LLM generates answers based on these documents, reducing fabrication.",
        {"difficulty": "medium", "id": "M01"},
    ),
    QAPair(
        "What is the difference between semantic search and keyword search?",
        "Semantic search uses embeddings to find conceptually similar content, while keyword search matches exact words using BM25 or TF-IDF.",
        "Keyword search (BM25) matches literal terms. Semantic search encodes text as vectors and finds similar meanings using cosine similarity.",
        {"difficulty": "medium", "id": "M02"},
    ),
    QAPair(
        "How does reranking improve RAG performance?",
        "Reranking re-scores retrieved chunks using a cross-encoder model, placing the most relevant chunks first to improve context precision.",
        "After initial retrieval, a reranker (cross-encoder) scores each chunk against the query. This improves precision without changing the retrieved set.",
        {"difficulty": "medium", "id": "M06"},
    ),
    QAPair(
        "Should I use RAG or fine-tuning for my customer support chatbot?",
        "RAG is better for frequently updated knowledge bases; fine-tuning suits consistent tone/behavior.",
        "RAG retrieves live knowledge; fine-tuning bakes knowledge into weights. Cost, data freshness, and latency determine the choice.",
        {"difficulty": "hard", "id": "H01"},
    ),
    QAPair(
        "Is RAG always better than fine-tuning?",
        "Not always. RAG excels for dynamic knowledge but adds retrieval latency. Fine-tuning is better for stable domain knowledge and low-latency needs.",
        "RAG vs fine-tuning depends on use case. Neither is universally superior.",
        {"difficulty": "adversarial", "id": "A03"},
    ),
]


def mock_agent(question: str) -> str:
    """Replace with real agent in production: agent_fn = your_agent.run"""
    answers = {
        "What is RAG?": "RAG stands for Retrieval-Augmented Generation. It combines document retrieval with large language model generation.",
        "What does LLM stand for?": "LLM stands for Large Language Model, which is a deep neural network trained on massive text data.",
        "What is a vector database?": "A vector database stores embeddings and supports similarity search using cosine distance.",
        "What is a prompt?": "A prompt is the input text sent to a language model. It instructs the model on what to generate.",
        "What is hallucination in LLMs?": "Hallucination is when an AI model generates false information that sounds confident but is not grounded in evidence.",
        "How does RAG reduce hallucination?": "RAG reduces hallucination by providing retrieved documents as grounding context.",
        "What is the difference between semantic search and keyword search?": "Semantic search uses vector embeddings. Keyword search matches exact terms using BM25.",
        "How does reranking improve RAG performance?": "Reranking re-scores retrieved chunks to put the most relevant ones first, improving context precision.",
        "Should I use RAG or fine-tuning for my customer support chatbot?": "It depends. RAG is better for dynamic knowledge bases. Fine-tuning works for consistent tone.",
        "Is RAG always better than fine-tuning?": "No. RAG suits dynamic knowledge; fine-tuning suits stable knowledge with low-latency needs.",
    }
    return answers.get(question, "I don't have enough information to answer that question.")


def run_quality_gate() -> int:
    """
    Run benchmark and check metrics against thresholds.
    Returns 0 (pass) or 1 (fail).
    """
    evaluator = RAGASEvaluator()
    runner = BenchmarkRunner()
    analyzer = FailureAnalyzer()

    print("=" * 60)
    print("AI EVALUATION QUALITY GATE")
    print("=" * 60)
    print(f"Dataset: {len(GOLDEN_DATASET)} QA pairs")
    print(f"Thresholds: {THRESHOLDS}")
    print()

    # Run benchmark
    results = runner.run(GOLDEN_DATASET, mock_agent, evaluator)
    report = runner.generate_report(results)
    failures = runner.identify_failures(results, threshold=0.5)

    # Print per-pair results
    print("PER-PAIR RESULTS:")
    print(f"{'ID':<4} {'Question':<42} {'Faith':>6} {'Rel':>6} {'Comp':>6} {'Pass':>5}")
    print("-" * 70)
    for qa, r in zip(GOLDEN_DATASET, results):
        qid = qa.metadata.get("id", "?")
        short_q = qa.question[:40]
        status = "PASS" if r.passed else "FAIL"
        print(f"{qid:<4} {short_q:<42} {r.faithfulness:>6.2f} {r.relevance:>6.2f} {r.completeness:>6.2f} {status:>5}")

    print()
    print("AGGREGATE REPORT:")
    print(f"  Total:            {report['total']}")
    print(f"  Passed:           {report['passed']}")
    print(f"  Pass rate:        {report['pass_rate']:.0%}")
    print(f"  Avg faithfulness: {report['avg_faithfulness']:.3f}")
    print(f"  Avg relevance:    {report['avg_relevance']:.3f}")
    print(f"  Avg completeness: {report['avg_completeness']:.3f}")
    print(f"  Failure types:    {report['failure_types']}")

    # Check thresholds
    print()
    print("QUALITY GATE CHECK:")
    gate_passed = True
    gate_results = {}
    for metric, threshold in THRESHOLDS.items():
        avg_key = f"avg_{metric}"
        actual = report[avg_key]
        passed = actual >= threshold
        gate_results[metric] = {"actual": actual, "threshold": threshold, "passed": passed}
        status = "PASS" if passed else "FAIL"
        marker = "" if passed else " <-- BLOCKED"
        print(f"  {metric:<15} {actual:.3f} >= {threshold:.2f} [{status}]{marker}")
        if not passed:
            gate_passed = False

    # Failure analysis
    if failures:
        print()
        print("FAILURE ANALYSIS:")
        categories = analyzer.categorize_failures(failures)
        print(f"  Failure categories: {categories}")
        suggestions = analyzer.generate_improvement_suggestions(failures)
        print("  Top suggestions:")
        for s in suggestions[:3]:
            print(f"    - {s}")

    # Save JSON report for artifact upload
    report_data = {
        "summary": report,
        "gate_results": gate_results,
        "gate_passed": gate_passed,
        "failure_categories": analyzer.categorize_failures(failures),
        "improvement_suggestions": analyzer.generate_improvement_suggestions(failures),
    }
    with open("benchmark_report.json", "w") as f:
        json.dump(report_data, f, indent=2)
    print()
    print("Report saved to: benchmark_report.json")

    # Final verdict
    print()
    print("=" * 60)
    if gate_passed:
        print("QUALITY GATE: PASSED — safe to deploy")
        print("=" * 60)
        return 0
    else:
        print("QUALITY GATE: FAILED — deployment blocked")
        print("Fix the metrics above threshold before merging.")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(run_quality_gate())
