"""
Top-level package for rag-score framework-agnostic RAG evaluation.

Public API:
- Core data structures and evaluation pipeline
- Metrics for retrieval and generation evaluation
- Agentic trajectory evaluation components
- LLM judge implementations
- Data export functionality
- Report generation
- Synthetic test set generation
- Response caching
- Telemetry
"""

from __future__ import annotations

from rag_score.agentic.metrics.tool_call_order import ToolCallOrderCorrectness
from rag_score.agentic.metrics.tool_selection_precision import ToolSelectionPrecision
from rag_score.agentic.metrics.tool_selection_recall import ToolSelectionRecall

# Agentic
from rag_score.agentic.types import TrajectoryEvalResult, TrajectoryTestCase

# Cache
from rag_score.cache import (
    CacheConfig,
    FileCacheBackend,
    InMemoryCacheBackend,
    JudgeCache,
    get_cache_backend,
)

# Core
from rag_score.core.dataset import load_dataset
from rag_score.core.runner import RunConfig, run_evaluation
from rag_score.core.types import (
    DimRun,
    DimTestCase,
    EvalResult,
    TestCase,
)

# Export
from rag_score.export.dataframe_export import (
    full_report_dataframe,
    results_to_dataframe,
    scores_to_dataframe,
    scores_to_wide_dataframe,
)
from rag_score.export.sqlite_export import export_to_sqlite
from rag_score.judges.anthropic_judge import AnthropicJudge

# Judges
from rag_score.judges.base import JudgeVerdict, LLMJudge
from rag_score.judges.local_judge import LocalJudge
from rag_score.judges.openai_judge import OpenAIJudge
from rag_score.metrics.generation.answer_relevance import AnswerRelevance
from rag_score.metrics.generation.context_carry_over import ContextCarryOver
from rag_score.metrics.generation.context_precision import ContextPrecision
from rag_score.metrics.generation.faithfulness import Faithfulness
from rag_score.metrics.generation.local_answer_relevance import (
    LocalSemanticAnswerRelevance,
)
from rag_score.metrics.generation.local_faithfulness import (
    LocalSemanticFaithfulness,
)

# Metrics
from rag_score.metrics.retrieval.mrr import MRR
from rag_score.metrics.retrieval.ndcg import NDCG
from rag_score.metrics.retrieval.precision_at_k import PrecisionAtK
from rag_score.metrics.retrieval.recall_at_k import RecallAtK

# Report
from rag_score.report.html_report import generate_html_report

# Synthesize
from rag_score.synthesize import synthesize_test_set

# Telemetry
from rag_score.telemetry import TelemetryConfig, count_tokens, estimate_cost

# Define what gets exported with "from rag_score import *"
__all__ = [
    "MRR",
    "NDCG",
    "AnswerRelevance",
    "AnthropicJudge",
    "CacheConfig",
    "ContextCarryOver",
    "ContextPrecision",
    "DimRun",
    "DimTestCase",
    "EvalResult",
    "Faithfulness",
    "FileCacheBackend",
    "InMemoryCacheBackend",
    "JudgeCache",
    "JudgeVerdict",
    "LLMJudge",
    "LocalJudge",
    "LocalSemanticAnswerRelevance",
    "LocalSemanticFaithfulness",
    "OpenAIJudge",
    "PrecisionAtK",
    "RecallAtK",
    "RunConfig",
    "TelemetryConfig",
    "TestCase",
    "ToolCallOrderCorrectness",
    "ToolSelectionPrecision",
    "ToolSelectionRecall",
    "TrajectoryEvalResult",
    "TrajectoryTestCase",
    "count_tokens",
    "estimate_cost",
    "export_to_sqlite",
    "full_report_dataframe",
    "generate_html_report",
    "get_cache_backend",
    "load_dataset",
    "results_to_dataframe",
    "run_evaluation",
    "scores_to_dataframe",
    "scores_to_wide_dataframe",
    "synthesize_test_set",
]