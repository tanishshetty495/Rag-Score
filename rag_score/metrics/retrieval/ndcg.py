"""
nDCG (Normalized Discounted Cumulative Gain) at k.

Unlike precision/recall/MRR (which treat relevance as binary and
position-blind beyond "found or not"), nDCG rewards relevant results
appearing *earlier* in the ranking via a log2 discount, then normalizes
against the ideal ordering so scores are comparable across queries with
different numbers of relevant docs.

Relevance here is binary (doc_id in expected_doc_ids -> 1, else 0)
since TestCase doesn't carry graded relevance judgments in v1.
"""

from __future__ import annotations

import math

from rag_score.core.types import EvalResult, TestCase
from rag_score.metrics.base import Metric


def _dcg(relevances: list[int]) -> float:
    return sum(
        rel / math.log2(rank + 1)  # rank is 1-indexed, so first term is rel/log2(2)=rel
        for rank, rel in enumerate(relevances, start=1)
    )


class NDCG(Metric):
    name = "ndcg_at_k"
    requires_api_key = False

    def __init__(self, k: int = 5) -> None:
        if k <= 0:
            raise ValueError("k must be a positive integer")
        self.k = k
        self.name = f"ndcg_at_{k}"

    async def score(self, test_case: TestCase, result: EvalResult) -> float:
        expected = set(test_case.expected_doc_ids)
        if not expected:
            return 0.0

        top_k = result.retrieved_context[: self.k]
        relevances = [
            1 if (chunk.doc_id is not None and chunk.doc_id in expected) else 0 for chunk in top_k
        ]

        actual_dcg = _dcg(relevances)

        # Ideal ranking: all relevant docs (up to k) first.
        ideal_relevances = [1] * min(len(expected), self.k) + [0] * max(0, self.k - len(expected))
        ideal_dcg = _dcg(ideal_relevances)

        if ideal_dcg == 0:
            return 0.0
        return actual_dcg / ideal_dcg
