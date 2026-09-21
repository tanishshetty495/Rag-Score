"""Tests for export/dataframe_export.py.

This module had zero test coverage before the telemetry feature - the
first several test classes below close that pre-existing gap (basic
shape/content of each export function); the TestTelemetryColumns class
at the end covers what telemetry specifically adds.
"""

from __future__ import annotations

from rag_score.core.runner import RunConfig, RunReport, run_evaluation
from rag_score.export.dataframe_export import (
    full_report_dataframe,
    results_to_dataframe,
    scores_to_dataframe,
    scores_to_wide_dataframe,
)
from rag_score.metrics.retrieval.mrr import MRR
from rag_score.metrics.retrieval.precision_at_k import PrecisionAtK
from rag_score.telemetry import TelemetryConfig


class TestResultsToDataframe:
    async def test_one_row_per_result(self, sample_test_cases, fake_retriever, fake_generator):
        config = RunConfig(run_id="r1")
        report = await run_evaluation(
            sample_test_cases, fake_retriever, fake_generator, [MRR()], config
        )
        df = results_to_dataframe(report)
        assert len(df) == len(sample_test_cases)

    async def test_contains_expected_columns(
        self, sample_test_cases, fake_retriever, fake_generator
    ):
        config = RunConfig(run_id="r1")
        report = await run_evaluation(
            sample_test_cases, fake_retriever, fake_generator, [MRR()], config
        )
        df = results_to_dataframe(report)
        for col in ("evaluation_id", "run_id", "test_case_id", "generated_answer"):
            assert col in df.columns

    def test_empty_report_returns_empty_dataframe(self):
        df = results_to_dataframe(RunReport(run_id="empty"))
        assert len(df) == 0


class TestScoresToDataframe:
    async def test_one_row_per_score(self, sample_test_cases, fake_retriever, fake_generator):
        metrics = [PrecisionAtK(k=3), MRR()]
        config = RunConfig(run_id="r1")
        report = await run_evaluation(
            sample_test_cases, fake_retriever, fake_generator, metrics, config
        )
        df = scores_to_dataframe(report)
        assert len(df) == len(sample_test_cases) * len(metrics)

    async def test_contains_metric_name_and_score_value(
        self, sample_test_cases, fake_retriever, fake_generator
    ):
        config = RunConfig(run_id="r1")
        report = await run_evaluation(
            sample_test_cases, fake_retriever, fake_generator, [MRR()], config
        )
        df = scores_to_dataframe(report)
        assert "metric_name" in df.columns
        assert "score_value" in df.columns
        assert (df["metric_name"] == "mrr").all()


class TestScoresToWideDataframe:
    async def test_one_column_per_metric(self, sample_test_cases, fake_retriever, fake_generator):
        metrics = [PrecisionAtK(k=3), MRR()]
        config = RunConfig(run_id="r1")
        report = await run_evaluation(
            sample_test_cases, fake_retriever, fake_generator, metrics, config
        )
        df = scores_to_wide_dataframe(report)
        assert "precision_at_3" in df.columns
        assert "mrr" in df.columns
        assert len(df) == len(sample_test_cases)

    def test_empty_report_returns_empty_dataframe(self):
        df = scores_to_wide_dataframe(RunReport(run_id="empty"))
        assert len(df) == 0


class TestFullReportDataframe:
    async def test_joins_question_and_scores(
        self, sample_test_cases, fake_retriever, fake_generator
    ):
        config = RunConfig(run_id="r1")
        report = await run_evaluation(
            sample_test_cases, fake_retriever, fake_generator, [MRR()], config
        )
        df = full_report_dataframe(report, sample_test_cases)
        assert "question" in df.columns
        assert "mrr" in df.columns
        assert len(df) == len(sample_test_cases)

    async def test_questions_match_original_test_cases(
        self, sample_test_cases, fake_retriever, fake_generator
    ):
        config = RunConfig(run_id="r1")
        report = await run_evaluation(
            sample_test_cases, fake_retriever, fake_generator, [MRR()], config
        )
        df = full_report_dataframe(report, sample_test_cases)
        expected_questions = {tc.question for tc in sample_test_cases}
        assert set(df["question"]) == expected_questions


class TestTelemetryColumns:
    async def test_sec_columns_derived_correctly(
        self, sample_test_cases, fake_retriever, fake_generator
    ):
        config = RunConfig(run_id="r1")
        report = await run_evaluation(
            sample_test_cases, fake_retriever, fake_generator, [MRR()], config
        )
        df = results_to_dataframe(report)
        assert "retrieval_latency_sec" in df.columns
        assert "generation_latency_sec" in df.columns
        for _, row in df.iterrows():
            assert row["retrieval_latency_sec"] == row["retrieval_latency_ms"] / 1000.0
            assert row["generation_latency_sec"] == row["generation_latency_ms"] / 1000.0

    async def test_telemetry_columns_populated_when_enabled(
        self, sample_test_cases, fake_retriever, fake_generator
    ):
        config = RunConfig(run_id="r1", telemetry=TelemetryConfig(model_name="gpt-4o-mini"))
        report = await run_evaluation(
            sample_test_cases, fake_retriever, fake_generator, [MRR()], config
        )
        df = results_to_dataframe(report)
        assert (df["total_tokens"] > 0).all()
        assert (df["estimated_cost_usd"] > 0).all()

    async def test_telemetry_columns_null_when_disabled(
        self, sample_test_cases, fake_retriever, fake_generator
    ):
        config = RunConfig(run_id="r1")  # telemetry off
        report = await run_evaluation(
            sample_test_cases, fake_retriever, fake_generator, [MRR()], config
        )
        df = results_to_dataframe(report)
        assert df["total_tokens"].isna().all()
        assert df["estimated_cost_usd"].isna().all()
