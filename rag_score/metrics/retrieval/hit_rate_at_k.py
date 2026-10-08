"""
Hit rate at k — fraction of test cases where at least one of the top-k retrieved
chunks has a doc_id that appears in the TestCase's expected_doc_ids.

Pure set-membership math, no dependencies, no network calls — this is
one of the "offline-first" metrics that works with zero API keys.
"""

from __future__ import annotations

from rag_score.core.metric import Metric
from rag_score.core.models import EvaluationCase


class HitRateAtK(Metric):
    def __init__(self, k: int = 5) -> None:
        if k <= 0:
            raise ValueError("k must be a positive integer")
        self.k = k
        # Keep the metric name specific per-k so a report can show
        # hit_rate_at_5 and hit_rate_at_10 side by side.
        self.name = f"hit_rate_at_{k}"

    def evaluate(self, case: EvaluationCase) -> MetricResult:
        # Get expected_doc_ids from metadata (stored when converting from TestCase)
        expected_doc_ids = set(case.metadata.get("expected_doc_ids", []))
        if not expected_doc_ids:
            # No ground truth -> undefined, return 0.0
            score_value = 0.0

        # Get retrieved_doc_ids from metadata (stored when creating EvaluationCase from pipeline output)
        retrieved_doc_ids = case.metadata.get("retrieved_doc_ids", [])
        if not retrieved_doc_ids:
            score_value = 0.0

        # Take top k retrieved_doc_ids (if available)
        top_k = retrieved_doc_ids[: self.k]
        # Filter out None doc_ids
        top_k_ids = [doc_id for doc_id in top_k if doc_id is not None]
        if not top_k_ids:
            score_value = 0.0

        # Check if any of the top_k_ids is in expected_doc_ids
        hit = any(doc_id in expected_doc_ids for doc_id in top_k_ids)
        score_value = 1.0 if hit else 0.0

        # Return a MetricResult
        return MetricResult(
            score_id="",  # We don't have an ID from the legacy metric
            evaluation_id="",  # We don't store this
            metric_name=self.name,
            score_value=score_value,
            judge_reasoning=None,  # These metrics don't provide reasoning by default
        )
