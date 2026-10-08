"""
Metrics for RAG evaluation.
"""

from __future__ import annotations

from rag_score.core.metric import register_metric
from rag_score.metrics.generation.answer_relevance import AnswerRelevance
from rag_score.metrics.generation.citation_completeness import CitationCompleteness
from rag_score.metrics.generation.citation_correctness import CitationCorrectness
from rag_score.metrics.generation.context_carry_over import ContextCarryOver
from rag_score.metrics.generation.context_precision import ContextPrecision
from rag_score.metrics.generation.faithfulness import Faithfulness
from rag_score.metrics.generation.local_answer_relevance import (
    LocalSemanticAnswerRelevance,
)
from rag_score.metrics.generation.local_faithfulness import (
    LocalSemanticFaithfulness,
)
from rag_score.metrics.generation.no_answer_abstention import NoAnswerAbstention
from rag_score.metrics.retrieval.context_coverage import ContextCoverage
from rag_score.metrics.retrieval.hit_rate_at_k import HitRateAtK
from rag_score.metrics.retrieval.mrr import MRR
from rag_score.metrics.retrieval.ndcg import NDCG
from rag_score.metrics.retrieval.precision_at_k import PrecisionAtK
from rag_score.metrics.retrieval.recall_at_k import RecallAtK

# Register existing metrics with the new registry
register_metric(AnswerRelevance)
register_metric(ContextCarryOver)
register_metric(ContextPrecision)
register_metric(Faithfulness)
register_metric(LocalSemanticAnswerRelevance)
register_metric(LocalSemanticFaithfulness)
register_metric(MRR)
register_metric(NDCG)
register_metric(PrecisionAtK)
register_metric(RecallAtK)
register_metric(HitRateAtK)
register_metric(ContextCoverage)
register_metric(CitationCorrectness)
register_metric(CitationCompleteness)
register_metric(NoAnswerAbstention)

__all__ = [
    "MRR",
    "NDCG",
    "AnswerRelevance",
    "CitationCompleteness",
    "CitationCorrectness",
    "ContextCarryOver",
    "ContextCoverage",
    "ContextPrecision",
    "Faithfulness",
    "HitRateAtK",
    "LocalSemanticAnswerRelevance",
    "LocalSemanticFaithfulness",
    "NoAnswerAbstention",
    "PrecisionAtK",
    "RecallAtK",
]
