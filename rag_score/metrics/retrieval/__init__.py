"""
Retrieval metrics for RAG evaluation.
"""

from __future__ import annotations

from rag_score.core.metric import register_metric
from rag_score.metrics.retrieval.context_coverage import ContextCoverage
from rag_score.metrics.retrieval.hit_rate_at_k import HitRateAtK
from rag_score.metrics.retrieval.mrr import MRR
from rag_score.metrics.retrieval.ndcg import NDCG
from rag_score.metrics.retrieval.precision_at_k import PrecisionAtK
from rag_score.metrics.retrieval.recall_at_k import RecallAtK

# Register existing metrics with the new registry
register_metric(MRR)
register_metric(NDCG)
register_metric(PrecisionAtK)
register_metric(RecallAtK)
register_metric(HitRateAtK)
register_metric(ContextCoverage)

__all__ = [
    "MRR",
    "NDCG",
    "ContextCoverage",
    "HitRateAtK",
    "PrecisionAtK",
    "RecallAtK",
]
