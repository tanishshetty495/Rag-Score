"""
Internal data models for the new metric system.
These are stdlib-only dataclasses (no pydantic) for internal use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievedChunk:
    """A retrieved chunk with document ID, text, and optional score."""

    doc_id: str | None = None
    text: str = ""
    score: float | None = None


@dataclass
class EvaluationCase:
    """
    Internal representation of a test case.
    Convertible to and from the existing TestCase type.
    """

    id: str
    question: str
    contexts: list[str] = field(default_factory=list)
    answer: str | None = None
    reference: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary suitable for JSON serialization."""
        return {
            "id": self.id,
            "question": self.question,
            "contexts": self.contexts,
            "answer": self.answer,
            "reference": self.reference,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationCase:
        """Create an EvaluationCase from a dictionary."""
        return cls(
            id=data["id"],
            question=data["question"],
            contexts=data.get("contexts", []),
            answer=data.get("answer"),
            reference=data.get("reference"),
            metadata=data.get("metadata", {}),
        )

    def to_test_case(self) -> TestCase:
        """
        Convert to the existing TestCase type (from rag_score.core.types).
        Note: This imports TestCase to avoid circular dependencies at runtime.
        """
        from rag_score.core.types import TestCase

        # Extract dataset_name, expected_doc_ids, and difficulty_category from metadata
        dataset_name = self.metadata.get("dataset_name", "default")
        expected_doc_ids = self.metadata.get("expected_doc_ids", [])
        difficulty_category = self.metadata.get("difficulty_category")
        # Metadata for TestCase should not include the extracted fields
        test_case_metadata = {
            k: v
            for k, v in self.metadata.items()
            if k not in ("dataset_name", "expected_doc_ids", "difficulty_category")
        }

        return TestCase(
            test_case_id=self.id,
            dataset_name=dataset_name,
            question=self.question,
            ground_truth_answer=self.reference,
            expected_doc_ids=expected_doc_ids,
            difficulty_category=difficulty_category,
            metadata=test_case_metadata,
        )

    @classmethod
    def from_test_case(cls, test_case: TestCase) -> EvaluationCase:
        """
        Create an EvaluationCase from the existing TestCase type.
        """
        # Store the TestCase-specific fields in metadata
        metadata = {
            **test_case.metadata,
            "dataset_name": test_case.dataset_name,
            "expected_doc_ids": list(test_case.expected_docids),  # ensure list
            "difficulty_category": test_case.difficulty_category,
        }
        return cls(
            id=test_case.test_case_id,
            question=test_case.question,
            contexts=[],  # contexts will be filled after retrieval
            answer=None,  # answer will be filled after generation
            reference=test_case.ground_truth_answer,
            metadata=metadata,
        )


@dataclass
class EvaluationResult:
    """
    Internal representation of the output of running an EvaluationCase through a pipeline.
    Convertible to and from the existing EvalResult type.
    """

    evaluation_id: str
    case_id: str
    retrieved_contexts: list[RetrievedChunk] = field(default_factory=list)
    generated_answer: str | None = None
    retrieval_latency_ms: float | None = None
    generation_latency_ms: float | None = None
    total_tokens: int | None = None
    estimated_cost_usd: float | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary suitable for JSON serialization."""
        return {
            "evaluation_id": self.evaluation_id,
            "case_id": self.case_id,
            "retrieved_contexts": [
                {
                    "doc_id": chunk.doc_id,
                    "text": chunk.text,
                    "score": chunk.score,
                }
                for chunk in self.retrieved_contexts
            ],
            "generated_answer": self.generated_answer,
            "retrieval_latency_ms": self.retrieval_latency_ms,
            "generation_latency_ms": self.generation_latency_ms,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationResult:
        """Create an EvaluationResult from a dictionary."""
        retrieved_chunks = [
            RetrievedChunk(
                doc_id=chunk.get("doc_id"),
                text=chunk.get("text", ""),
                score=chunk.get("score"),
            )
            for chunk in data.get("retrieved_contexts", [])
        ]
        return cls(
            evaluation_id=data["evaluation_id"],
            case_id=data["case_id"],
            retrieved_contexts=retrieved_chunks,
            generated_answer=data.get("generated_answer"),
            retrieval_latency_ms=data.get("retrieval_latency_ms"),
            generation_latency_ms=data.get("generation_latency_ms"),
            total_tokens=data.get("total_tokens"),
            estimated_cost_usd=data.get("estimated_cost_usd"),
            error=data.get("error"),
        )

    def to_eval_result(self) -> EvalResult:
        """
        Convert to the existing EvalResult type (from rag_score.core.types).
        """
        from rag_score.core.types import EvalResult

        return EvalResult(
            evaluation_id=self.evaluation_id,
            run_id="",  # run_id is not stored in EvaluationResult; we might need to add it?
            test_case_id=self.case_id,
            retrieved_context=self.retrieved_contexts,
            generated_answer=self.generated_answer,
            retrieval_latency_ms=self.retrieval_latency_ms,
            generation_latency_ms=self.generation_latency_ms,
            total_tokens=self.total_tokens,
            estimated_cost_usd=self.estimated_cost_usd,
            error=self.error,
        )

    @classmethod
    def from_eval_result(cls, eval_result: EvalResult) -> EvaluationResult:
        """
        Create an EvaluationResult from the existing EvalResult type.
        """
        return cls(
            evaluation_id=eval_result.evaluation_id,
            case_id=eval_result.test_case_id,
            retrieved_contexts=list(eval_result.retrieved_context),
            generated_answer=eval_result.generated_answer,
            retrieval_latency_ms=eval_result.retrieval_latency_ms,
            generation_latency_ms=eval_result.generation_latency_ms,
            total_tokens=eval_result.total_tokens,
            estimated_cost_usd=eval_result.estimated_cost_usd,
            error=eval_result.error,
        )


@dataclass
class MetricResult:
    """
    Internal representation of a metric's score on an EvaluationResult.
    Convertible to and from the existing MetricScore type.
    """

    score_id: str
    evaluation_id: str
    metric_name: str
    score_value: float
    judge_reasoning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary suitable for JSON serialization."""
        return {
            "score_id": self.score_id,
            "evaluation_id": self.evaluation_id,
            "metric_name": self.metric_name,
            "score_value": self.score_value,
            "judge_reasoning": self.judge_reasoning,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricResult:
        """Create a MetricResult from a dictionary."""
        return cls(
            score_id=data["score_id"],
            evaluation_id=data["evaluation_id"],
            metric_name=data["metric_name"],
            score_value=data["score_value"],
            judge_reasoning=data.get("judge_reasoning"),
        )

    def to_metric_score(self) -> MetricScore:
        """
        Convert to the existing MetricScore type (from rag_score.core.types).
        """
        from rag_score.core.types import MetricScore

        return MetricScore(
            score_id=self.score_id,
            evaluation_id=self.evaluation_id,
            metric_name=self.metric_name,
            score_value=self.score_value,
            judge_reasoning=self.judge_reasoning,
        )

    @classmethod
    def from_metric_score(cls, metric_score: MetricScore) -> MetricResult:
        """
        Create a MetricResult from the existing MetricScore type.
        """
        return cls(
            score_id=metric_score.score_id,
            evaluation_id=metric_score.evaluation_id,
            metric_name=metric_score.metric_name,
            score_value=metric_score.score_value,
            judge_reasoning=metric_score.judge_reasoning,
        )


@dataclass
class Failure:
    """
    Stub for a failure case (to be implemented later).
    """

    message: str = "Failure details to be implemented"

    def to_dict(self) -> dict[str, Any]:
        return {"message": self.message}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Failure:
        return cls(message=data.get("message", "Failure details to be implemented"))
