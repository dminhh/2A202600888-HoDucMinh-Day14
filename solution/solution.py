"""
Day 14 — AI Evaluation & Benchmarking Pipeline
AICB-P1: AI Practical Competency Program, Phase 1

Key concepts from lecture:
    - Evaluation = Scientific Method for AI (Hypothesis → Experiment → Measure → Conclude → Iterate)
    - 4 nhóm metrics: Task Completion, Answer Quality, RAG-Specific, Business
    - RAG pipeline metrics: Context Recall → Context Precision → Faithfulness → Answer Relevancy
    - LLM-as-Judge: rubric scoring 1-5, detect bias (positional, verbosity, self-preference)
    - Golden dataset: stratified sampling (5 Easy + 7 Medium + 5 Hard + 3 Adversarial)
    - Failure taxonomy: hallucination, irrelevant, incomplete, off_topic, refusal
    - 5 Whys method for root cause analysis
    - CI/CD integration: eval as quality gate (score < threshold = block deploy)
    - Continuous Improvement Loop: Evaluate → Analyze → Improve → Augment → Repeat
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Task 1 — Data Models (Golden Dataset + Evaluation Results)
# ---------------------------------------------------------------------------

@dataclass
class QAPair:
    """
    A question-answer pair for evaluation (part of the Golden Dataset).

    Fields:
        question:           The question to answer.
        expected_answer:    The reference/ground-truth answer (expert-written).
        context:            Source context (may be empty string if not applicable).
        metadata:           Optional metadata dict (difficulty, category, etc.).
        retrieved_contexts: List of retrieved chunks (ORDER = retriever rank).
                            Used by the retrieval-side metrics (Task 2b).
    """
    question: str
    expected_answer: str
    context: str = ""
    metadata: dict = field(default_factory=dict)
    retrieved_contexts: list = field(default_factory=list)


@dataclass
class EvalResult:
    """
    Evaluation result for a single Q&A pair.

    Fields:
        qa_pair:           The original QAPair.
        actual_answer:     What the agent actually returned.
        faithfulness:      Float 0-1, how grounded the answer is in context.
        relevance:         Float 0-1, how relevant the answer is to the question.
        completeness:      Float 0-1, how complete the answer is vs expected.
        passed:            True if all three scores >= 0.5.
        failure_type:      None if passed, otherwise one of:
                           "hallucination", "irrelevant", "incomplete", "off_topic".
        context_precision: Float 0-1 or None — quality of retrieval ranking.
        context_recall:    Float 0-1 or None — coverage of expected by context.
    """
    qa_pair: QAPair
    actual_answer: str
    faithfulness: float
    relevance: float
    completeness: float
    passed: bool
    failure_type: str | None = None
    context_precision: float | None = None
    context_recall: float | None = None

    def overall_score(self) -> float:
        """Return mean of faithfulness, relevance, and completeness."""
        return (self.faithfulness + self.relevance + self.completeness) / 3.0


# ---------------------------------------------------------------------------
# Task 2 — RAGAS Evaluator (Simplified word-overlap heuristic)
# ---------------------------------------------------------------------------

STOPWORDS: set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "as", "by", "and", "or",
    "it", "its", "this", "that", "these", "those", "from", "into", "than",
}


def _tokenize(text: str) -> set[str]:
    """Lowercase word tokenization, ignoring punctuation and stopwords."""
    if not text:
        return set()
    tokens = re.findall(r"\b\w+\b", text.lower())
    return {t for t in tokens if t not in STOPWORDS}


class RAGASEvaluator:
    """
    Evaluates RAG pipeline outputs using RAGAS-inspired heuristics.

    All metrics use word overlap rather than LLM calls for simplicity.
    """

    def evaluate_faithfulness(self, answer: str, context: str) -> float:
        """
        Measure how grounded the answer is in the context.

        faithfulness = |answer_tokens ∩ context_tokens| / |answer_tokens|
        """
        answer_tokens = _tokenize(answer)
        if not answer_tokens:
            return 1.0
        context_tokens = _tokenize(context)
        score = len(answer_tokens & context_tokens) / len(answer_tokens)
        return max(0.0, min(1.0, score))

    def evaluate_relevance(self, answer: str, question: str) -> float:
        """
        Measure how relevant the answer is to the question.

        relevance = |answer_tokens ∩ question_tokens| / |question_tokens|
        """
        question_tokens = _tokenize(question)
        if not question_tokens:
            return 1.0
        answer_tokens = _tokenize(answer)
        score = len(answer_tokens & question_tokens) / len(question_tokens)
        return max(0.0, min(1.0, score))

    def evaluate_completeness(self, answer: str, expected: str) -> float:
        """
        Measure how well the answer covers the expected answer.

        completeness = |answer_tokens ∩ expected_tokens| / |expected_tokens|
        """
        expected_tokens = _tokenize(expected)
        if not expected_tokens:
            return 1.0
        answer_tokens = _tokenize(answer)
        score = len(answer_tokens & expected_tokens) / len(expected_tokens)
        return max(0.0, min(1.0, score))

    # -----------------------------------------------------------------------
    # Task 2b — Retrieval-side metrics
    # -----------------------------------------------------------------------

    def evaluate_context_recall(self, contexts: list[str], expected: str) -> float:
        """
        Context Recall — how much of the expected answer is covered by the
        UNION of retrieved chunks.

        recall = |expected_tokens ∩ (⋃ chunk_tokens)| / |expected_tokens|
        """
        expected_tokens = _tokenize(expected)
        if not expected_tokens:
            return 1.0
        union_tokens: set[str] = set()
        for chunk in contexts:
            union_tokens |= _tokenize(chunk)
        score = len(expected_tokens & union_tokens) / len(expected_tokens)
        return max(0.0, min(1.0, score))

    def evaluate_context_precision(
        self,
        contexts: list[str],
        expected: str,
        relevance_threshold: float = 0.1,
    ) -> float:
        """
        Context Precision — RANK-AWARE Average Precision (AP@K).
        Rewards retrievers that place RELEVANT chunks BEFORE noise.

        A chunk is relevant if |chunk ∩ expected| / |expected| >= threshold.
        AP@K = (1 / #relevant) * Σ_k [ Precision@k · relevant_k ]
        """
        expected_tokens = _tokenize(expected)
        if not expected_tokens:
            return 1.0
        if not contexts:
            return 0.0

        relevance_flags = []
        for chunk in contexts:
            chunk_tokens = _tokenize(chunk)
            overlap = len(chunk_tokens & expected_tokens) / len(expected_tokens)
            relevance_flags.append(overlap >= relevance_threshold)

        num_relevant = sum(relevance_flags)
        if num_relevant == 0:
            return 0.0

        ap = 0.0
        running_relevant = 0
        for k, is_relevant in enumerate(relevance_flags, start=1):
            if is_relevant:
                running_relevant += 1
                ap += running_relevant / k
        ap /= num_relevant
        return max(0.0, min(1.0, ap))

    def run_full_eval(
        self,
        answer: str,
        question: str,
        context: str,
        expected: str,
        contexts: list[str] | None = None,
    ) -> EvalResult:
        """
        Run all three answer-side evaluations and combine into an EvalResult.

        passed = True if all three scores >= 0.5.

        failure_type (first match wins):
            faithfulness < 0.3  → "hallucination"
            relevance < 0.3     → "irrelevant"
            completeness < 0.3  → "incomplete"
            otherwise if failed → "off_topic"
        """
        faithfulness = self.evaluate_faithfulness(answer, context)
        relevance = self.evaluate_relevance(answer, question)
        completeness = self.evaluate_completeness(answer, expected)

        passed = faithfulness >= 0.5 and relevance >= 0.5 and completeness >= 0.5

        failure_type = None
        if not passed:
            if faithfulness < 0.3:
                failure_type = "hallucination"
            elif relevance < 0.3:
                failure_type = "irrelevant"
            elif completeness < 0.3:
                failure_type = "incomplete"
            else:
                failure_type = "off_topic"

        context_recall = None
        context_precision = None
        if contexts is not None:
            context_recall = self.evaluate_context_recall(contexts, expected)
            context_precision = self.evaluate_context_precision(contexts, expected)

        qa_pair = QAPair(
            question=question,
            expected_answer=expected,
            context=context or "",
        )

        return EvalResult(
            qa_pair=qa_pair,
            actual_answer=answer,
            faithfulness=faithfulness,
            relevance=relevance,
            completeness=completeness,
            passed=passed,
            failure_type=failure_type,
            context_recall=context_recall,
            context_precision=context_precision,
        )


# ---------------------------------------------------------------------------
# Reranking helper (Exercise 3.5)
# ---------------------------------------------------------------------------

def rerank_by_overlap(contexts: list[str], query: str) -> list[str]:
    """Sort chunks by word overlap with query, most-overlapping first."""
    return sorted(
        contexts,
        key=lambda c: len(_tokenize(c) & _tokenize(query)),
        reverse=True,
    )


# ---------------------------------------------------------------------------
# Task 3 — LLM Judge
# ---------------------------------------------------------------------------

class LLMJudge:
    """Uses an LLM to score AI responses according to a rubric."""

    def __init__(self, judge_llm_fn: Callable[[str], str]) -> None:
        self.judge_llm_fn = judge_llm_fn

    def score_response(
        self,
        question: str,
        answer: str,
        rubric: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Score an AI response using the judge LLM.

        Returns:
            {"scores": dict[str, float], "reasoning": str}
        """
        rubric_text = "\n".join(f"- {k}: {v}" for k, v in rubric.items())
        prompt = (
            f"You are an expert judge. Score the following answer.\n\n"
            f"Question: {question}\n"
            f"Answer: {answer}\n\n"
            f"Rubric:\n{rubric_text}\n\n"
            f"Return a JSON object with scores (0.0-1.0) for each criterion.\n"
            f'Example: {{"accuracy": 0.8, "clarity": 0.7}}'
        )

        response = self.judge_llm_fn(prompt)

        try:
            parsed = json.loads(response)
            if not isinstance(parsed, dict):
                raise ValueError("Expected dict")
            scores = {k: float(v) for k, v in parsed.items()}
        except Exception:
            scores = {k: 0.5 for k in rubric.keys()}

        return {"scores": scores, "reasoning": response}

    def detect_bias(self, scores_batch: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Detect potential bias patterns in a batch of judge scores.

        Checks:
            positional_bias: first item scores consistently higher than the rest
            leniency_bias:   global average score > 0.8
            severity_bias:   global average score < 0.3
        """
        if not scores_batch:
            return {"positional_bias": False, "leniency_bias": False, "severity_bias": False}

        def item_avg(item: dict) -> float | None:
            scores = item.get("scores", {})
            if not scores:
                return None
            return sum(scores.values()) / len(scores)

        avgs = [item_avg(item) for item in scores_batch]
        avgs = [a for a in avgs if a is not None]

        if not avgs:
            return {"positional_bias": False, "leniency_bias": False, "severity_bias": False}

        global_avg = sum(avgs) / len(avgs)

        # Positional bias: first item scores noticeably higher than the rest
        positional_bias = False
        if len(avgs) > 1:
            first_avg = avgs[0]
            rest_avg = sum(avgs[1:]) / len(avgs[1:])
            positional_bias = (first_avg - rest_avg) > 0.1

        return {
            "positional_bias": positional_bias,
            "leniency_bias": global_avg > 0.8,
            "severity_bias": global_avg < 0.3,
        }


# ---------------------------------------------------------------------------
# Task 4 — Benchmark Runner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """Runs a full evaluation benchmark."""

    def run(
        self,
        qa_pairs: list[QAPair],
        agent_fn: Callable[[str], str],
        evaluator: RAGASEvaluator,
    ) -> list[EvalResult]:
        """Run all QA pairs through the agent and evaluate each result."""
        results = []
        for pair in qa_pairs:
            answer = agent_fn(pair.question)
            contexts = pair.retrieved_contexts if pair.retrieved_contexts else None
            result = evaluator.run_full_eval(
                answer=answer,
                question=pair.question,
                context=pair.context or "",
                expected=pair.expected_answer,
                contexts=contexts,
            )
            # Preserve the original qa_pair (with metadata, retrieved_contexts)
            result.qa_pair = pair
            results.append(result)
        return results

    def generate_report(self, results: list[EvalResult]) -> dict[str, Any]:
        """Generate an aggregate report from evaluation results."""
        total = len(results)
        if total == 0:
            return {
                "total": 0,
                "passed": 0,
                "pass_rate": 0.0,
                "avg_faithfulness": 0.0,
                "avg_relevance": 0.0,
                "avg_completeness": 0.0,
                "failure_types": {},
            }

        passed = sum(1 for r in results if r.passed)
        failure_types: dict[str, int] = {}
        for r in results:
            if r.failure_type:
                failure_types[r.failure_type] = failure_types.get(r.failure_type, 0) + 1

        return {
            "total": total,
            "passed": passed,
            "pass_rate": passed / total,
            "avg_faithfulness": sum(r.faithfulness for r in results) / total,
            "avg_relevance": sum(r.relevance for r in results) / total,
            "avg_completeness": sum(r.completeness for r in results) / total,
            "failure_types": failure_types,
        }

    def run_regression(self, new_results: list, baseline_results: list) -> dict:
        """
        Compare new evaluation results against a baseline.

        A regression is when a metric's average drops by more than 0.05.
        """
        def avg(results: list, attr: str) -> float:
            if not results:
                return 0.0
            return sum(getattr(r, attr) for r in results) / len(results)

        new_f = avg(new_results, "faithfulness")
        new_r = avg(new_results, "relevance")
        new_c = avg(new_results, "completeness")
        base_f = avg(baseline_results, "faithfulness")
        base_r = avg(baseline_results, "relevance")
        base_c = avg(baseline_results, "completeness")

        regressions = []
        if base_f - new_f > 0.05:
            regressions.append("faithfulness")
        if base_r - new_r > 0.05:
            regressions.append("relevance")
        if base_c - new_c > 0.05:
            regressions.append("completeness")

        return {
            "new_avg_faithfulness": new_f,
            "new_avg_relevance": new_r,
            "new_avg_completeness": new_c,
            "baseline_avg_faithfulness": base_f,
            "baseline_avg_relevance": base_r,
            "baseline_avg_completeness": base_c,
            "regressions": regressions,
            "passed": len(regressions) == 0,
        }

    def identify_failures(
        self,
        results: list[EvalResult],
        threshold: float = 0.5,
    ) -> list[EvalResult]:
        """Return EvalResults where any score is below threshold."""
        return [
            r for r in results
            if r.faithfulness < threshold
            or r.relevance < threshold
            or r.completeness < threshold
        ]


# ---------------------------------------------------------------------------
# Task 5 — Failure Analyzer
# ---------------------------------------------------------------------------

class FailureAnalyzer:
    """Analyzes failed evaluation results to identify patterns and suggest fixes."""

    def categorize_failures(self, failures: list[EvalResult]) -> dict[str, int]:
        """Count failures by failure_type."""
        categories: dict[str, int] = {}
        for f in failures:
            if f.failure_type:
                categories[f.failure_type] = categories.get(f.failure_type, 0) + 1
        return categories

    def find_root_cause(self, failure: EvalResult) -> str:
        """Suggest a root cause based on which score is lowest."""
        scores = {
            "faithfulness": failure.faithfulness,
            "relevance": failure.relevance,
            "completeness": failure.completeness,
        }
        lowest = min(scores, key=scores.get)
        if lowest == "faithfulness":
            return "Context is missing or irrelevant — improve retrieval"
        elif lowest == "relevance":
            return "Answer does not address the question — improve prompt clarity"
        else:
            return "Answer is missing key information — increase context window or improve generation"

    def generate_improvement_log(self, failures: list, suggestions: list[str]) -> str:
        """
        Generate a Markdown table logging failures and improvement actions.

        Format:
        | Failure ID | Type | Root Cause | Suggested Fix | Status |
        """
        header = "| Failure ID | Type | Root Cause | Suggested Fix | Status |\n"
        separator = "|------------|------|------------|---------------|--------|\n"
        rows = []
        for i, failure in enumerate(failures):
            failure_id = f"F{i + 1:03d}"
            failure_type = failure.failure_type or "unknown"
            root_cause = self.find_root_cause(failure)
            suggested_fix = suggestions[i] if i < len(suggestions) else "Review and fix"
            rows.append(
                f"| {failure_id} | {failure_type} | {root_cause} | {suggested_fix} | Open |"
            )
        return header + separator + "\n".join(rows)

    def generate_improvement_suggestions(self, failures: list[EvalResult]) -> list[str]:
        """
        Generate a prioritized list of improvement suggestions based on failure patterns.

        Returns at least 3 suggestions.
        """
        if not failures:
            return []

        categories = self.categorize_failures(failures)
        suggestions: list[str] = []

        if categories.get("hallucination", 0) > 0:
            suggestions.append(
                "Implement hallucination checker to filter unsupported claims"
            )
        if categories.get("irrelevant", 0) > 0:
            suggestions.append(
                "Improve prompt clarity and intent detection to address off-topic answers"
            )
        if categories.get("incomplete", 0) > 0:
            suggestions.append(
                "Increase chunk size in RAG pipeline to reduce context fragmentation"
            )
        if categories.get("off_topic", 0) > 0:
            suggestions.append(
                "Add routing logic to better handle ambiguous or out-of-scope queries"
            )

        # Ensure at least 3 suggestions with general fallbacks
        fallbacks = [
            "Add few-shot examples showing complete answers to improve completeness",
            "Implement retrieval reranking to surface more relevant context first",
            "Add evaluation monitoring in CI/CD to catch regressions early",
        ]
        for fb in fallbacks:
            if len(suggestions) >= 3:
                break
            if fb not in suggestions:
                suggestions.append(fb)

        return suggestions


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    qa_pairs = [
        QAPair(
            question="What is RAG?",
            expected_answer="RAG stands for Retrieval-Augmented Generation, which combines retrieval with text generation.",
            context="RAG is a technique that retrieves relevant documents and uses them to ground LLM generation.",
            metadata={"difficulty": "easy", "category": "definition"},
        ),
        QAPair(
            question="What is the capital of France?",
            expected_answer="Paris is the capital of France.",
            context="France is a country in Western Europe. Its capital city is Paris.",
            metadata={"difficulty": "easy", "category": "factual"},
        ),
        QAPair(
            question="Explain backpropagation and why it matters for training",
            expected_answer="Backpropagation is an algorithm for training neural networks by computing gradients efficiently.",
            context="Neural networks learn through gradient descent. Backpropagation efficiently computes these gradients layer by layer.",
            metadata={"difficulty": "medium", "category": "explanation"},
        ),
        QAPair(
            question="Should I use RAG or fine-tuning for my chatbot?",
            expected_answer="It depends: RAG is better for frequently updated knowledge, fine-tuning for consistent style/behavior.",
            context="RAG retrieves external documents at inference time. Fine-tuning modifies model weights during training.",
            metadata={"difficulty": "hard", "category": "comparison"},
        ),
        QAPair(
            question="What is the meaning of life?",
            expected_answer="This question is outside the scope of this system.",
            context="This is an AI assistant specialized in technology topics.",
            metadata={"difficulty": "adversarial", "category": "out_of_scope"},
        ),
    ]

    evaluator = RAGASEvaluator()
    runner = BenchmarkRunner()

    def mock_agent(question: str) -> str:
        return f"Based on my knowledge: {question[:30]}... The answer involves key concepts."

    results = runner.run(qa_pairs, mock_agent, evaluator)
    report = runner.generate_report(results)
    print("=== Benchmark Report ===")
    for k, v in report.items():
        print(f"  {k}: {v}")

    failures = runner.identify_failures(results, threshold=0.5)
    print(f"\n=== Failures ({len(failures)}) ===")
    analyzer = FailureAnalyzer()

    categories = analyzer.categorize_failures(failures)
    print("Failure Categories:", categories)

    for f in failures:
        cause = analyzer.find_root_cause(f)
        print(f"  Root cause: {cause}")

    suggestions = analyzer.generate_improvement_suggestions(failures)
    print("\nImprovement Suggestions:")
    for s in suggestions:
        print(f"  - {s}")

    log = analyzer.generate_improvement_log(failures, suggestions)
    print("\n=== Improvement Log ===")
    print(log)
