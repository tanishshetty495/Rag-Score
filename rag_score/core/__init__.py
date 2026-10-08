"""
Core data structures and evaluation pipeline for rag-score.
"""

from __future__ import annotations

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

__all__ = [
    "DimRun",
    "DimTestCase",
    "EvalResult",
    "FactEvaluation",
    "FactMetricScore",
    "RunConfig",
    "TestCase",
    "load_dataset",
    "run_evaluation",
]
