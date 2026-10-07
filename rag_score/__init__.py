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

# Core
from rag_score.core.dataset import load_dataset
from rag_score.core.runner import RunConfig, run_evaluation
from rag_score.core.types import (
    DimRun,
    DimTestCase,
    EvalResult,
    FactEvaluation,
    FactMetricScore,
    TestCase,
)

# Metrics
from rag_score.metrics.retrieval.mrr import MRR
from rag_score.metrics.retrieval.ndcg import NDCG
from rag_score.metrics.retrieval.precision_at_k import PrecisionAtK
from rag_score.metrics.retrieval.recall_at_k import RecallAtK
from rag_score.metrics.generation.answer_relevance import AnswerRelevance
from rag_score.metrics.generation.context_precision import ContextPrecision
from rag_score.metrics.generation.context_carry_over import ContextCarryOver
from rag_score.metrics.generation.faithfulness import Faithfulness
from rag_score.metrics.generation.local_answer_relevance import (
    LocalSemanticAnswerRelevance,
)
from rag_score.metrics.generation.local_faithfulness import (
    LocalSemanticFaithfulness,
)

# Agentic
from rag_score.agentic.types import TrajectoryEvalResult, TrajectoryTestCase
from rag_score.agentic.metrics.tool_call_order import ToolCallOrderCorrectness
from rag_score.agentic.metrics.tool_selection_precision import ToolSelectionPrecision
from rag_score.agentic.metrics.tool_selection_recall import ToolSelectionRecall

# Judges
from rag_score.judges.base import LLMJudge, JudgeVerdict
from rag_score.judges.anthropic_judge import AnthropicJudge
from rag_score.judges.local_judge import LocalJudge
from rag_score.judges.openai_judge import OpenAIJudge

# Export
from rag_score.export.dataframe_export import (
    full_report_dataframe,
    scores_to_dataframe,
    scores_to_wide_dataframe,
    results_to_dataframe,
)
from rag_score.export.sqlite_export import export_to_sqlite

# Report
from rag_score.report.html_report import generate_html_report

# Synthesize
from rag_score.synthesize import synthesize_test_set

# Cache
from rag_score.cache import (
    CacheConfig,
    FileCacheBackend,
    InMemoryCacheBackend,
    JudgeCache,
    get_cache_backend,
)

# Telemetry
from rag_score.telemetry import TelemetryConfig, count_tokens, estimate_cost

# Define what gets exported with "from rag_score import *"
__all__ = [
    # Core
    "TestCase",
    "EvalResult",
    "load_dataset",
    "RunConfig",
    "run_evaluation",
    "DimRun",
    "DimTestCase",

    # Metrics
    "PrecisionAtK",
    "RecallAtK",
    "MRR",
    "NDCG",
    "Faithfulness",
    "AnswerRelevance",
    "ContextPrecision",
    "ContextCarryOver",
    "LocalSemanticAnswerRelevance",
    "LocalSemanticFaithfulness",

    # Agentic
    "TrajectoryTestCase",
    "TrajectoryEvalResult",
    "ToolSelectionRecall",
    "ToolSelectionPrecision",
    "ToolCallOrderCorrectness",

    # Judges
    "LLMJudge",
    "JudgeVerdict",
    "OpenAIJudge",
    "AnthropicJudge",
    "LocalJudge",

    # Export
    "export_to_sqlite",
    "results_to_dataframe",
    "scores_to_dataframe",
    "scores_to_wide_dataframe",
    "full_report_dataframe",

    # Report
    "generate_html_report",

    # Synthesize
    "synthesize_test_set",

    # Cache
    "JudgeCache",
    "InMemoryCacheBackend",
    "FileCacheBackend",
    "CacheConfig",
    "get_cache_backend",

    # Telemetry
    "TelemetryConfig",
    "count_tokens",
    "estimate_cost",
]