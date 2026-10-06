"""
Core data models for rag-score.

These are the shapes that flow through the whole pipeline:
TestCase (input) -> EvalResult (raw run output) -> MetricScore (scored output).
The dim_/fact_ models mirror the SQLite star schema used for BI export,
so a DataFrame built from these can be written straight to SQLite with
executemany() and no reshaping.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def _new_id() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Input: what a user provides in their dataset (test_set.json)
# ---------------------------------------------------------------------------


class TestCase(BaseModel):
    """A single question in the evaluation dataset."""

    __test__ = False  # tell pytest this isn't a test class despite the name

    test_case_id: str = Field(default_factory=_new_id)
    dataset_name: str = "default"
    question: str
    ground_truth_answer: str | None = None
    # IDs of documents/chunks considered relevant, for precision/recall/MRR/nDCG.
    expected_doc_ids: list[str] = Field(default_factory=list)
    difficulty_category: str | None = None  # e.g. "easy" | "multi-hop" | "adversarial"
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Mid: raw output of running a retriever/generator against a TestCase
# ---------------------------------------------------------------------------


class RetrievedChunk(BaseModel):
    """A single retrieved chunk, with enough identity to score against
    expected_doc_ids (doc_id) while still carrying the raw text for
    LLM-judge metrics (faithfulness, context precision)."""

    doc_id: str | None = None  # None if the adapter can't supply stable IDs
    text: str
    score: float | None = None  # retriever's own similarity/rank score, if any


class EvalResult(BaseModel):
    """The full output of running one TestCase through a pipeline once.
    This is what every Metric.score() receives alongside the TestCase."""

    evaluation_id: str = Field(default_factory=_new_id)
    run_id: str
    test_case_id: str

    retrieved_context: list[RetrievedChunk] = Field(default_factory=list)
    generated_answer: str | None = None

    retrieval_latency_ms: float | None = None
    generation_latency_ms: float | None = None
    total_tokens: int | None = None
    estimated_cost_usd: float | None = None

    error: str | None = None  # populated if the pipeline raised; metrics should skip


class MetricScore(BaseModel):
    """One metric's verdict on one EvalResult."""

    score_id: str = Field(default_factory=_new_id)
    evaluation_id: str
    metric_name: str
    score_value: float
    # Populated by LLM-judge metrics for auditability; None for pure-math metrics
    # like precision_at_k.
    judge_reasoning: str | None = None


# ---------------------------------------------------------------------------
# Star schema dimension/fact rows (BI export: SQLite / Pandas / Power BI)
# ---------------------------------------------------------------------------


class DimRun(BaseModel):
    run_id: str = Field(default_factory=_new_id)
    run_timestamp: datetime = Field(default_factory=_utcnow)
    project_name: str
    dataset_name: str
    retriever_config: str = ""  # free-text/JSON-serialized config, kept flat for SQLite
    generator_config: str = ""
    environment: str = "local"  # e.g. "local" | "ci" | "prod"


class DimTestCase(BaseModel):
    test_case_id: str
    dataset_name: str
    question: str
    ground_truth_answer: str | None = None
    expected_doc_ids: str = ""  # JSON-encoded list, flattened for SQLite storage
    difficulty_category: str | None = None


class FactEvaluation(BaseModel):
    evaluation_id: str
    run_id: str
    test_case_id: str
    retrieved_context: str = ""  # JSON-encoded list[RetrievedChunk]
    generated_answer: str | None = None
    retrieval_latency_ms: float | None = None
    generation_latency_ms: float | None = None
    total_tokens: int | None = None
    estimated_cost_usd: float | None = None


class FactMetricScore(BaseModel):
    score_id: str
    evaluation_id: str
    metric_name: str
    score_value: float
    judge_reasoning: str | None = None
