"""
Generation metrics for RAG evaluation.
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

# Register existing metrics with the new registry
register_metric(AnswerRelevance)
register_metric(ContextCarryOver)
register_metric(ContextPrecision)
register_metric(Faithfulness)
register_metric(LocalSemanticAnswerRelevance)
register_metric(LocalSemanticFaithfulness)
register_metric(CitationCorrectness)
register_metric(CitationCompleteness)
register_metric(NoAnswerAbstention)

__all__ = [
    "AnswerRelevance",
    "CitationCompleteness",
    "CitationCorrectness",
    "ContextCarryOver",
    "ContextPrecision",
    "Faithfulness",
    "LocalSemanticAnswerRelevance",
    "LocalSemanticFaithfulness",
    "NoAnswerAbstention",
]
