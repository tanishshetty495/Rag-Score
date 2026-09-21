"""Tests for export/sqlite_export.py - the star-schema writer."""

from __future__ import annotations

import sqlite3

import pytest

from rag_score.core.runner import RunConfig, run_evaluation
from rag_score.core.types import DimRun
from rag_score.export.sqlite_export import export_to_sqlite
from rag_score.metrics.retrieval.mrr import MRR
from rag_score.metrics.retrieval.precision_at_k import PrecisionAtK
from rag_score.telemetry import TelemetryConfig


class TestExportToSqlite:
    async def test_creates_all_four_tables(
        self, tmp_path, sample_test_cases, fake_retriever, fake_generator
    ):
        config = RunConfig(run_id="r1")
        report = await run_evaluation(
            sample_test_cases, fake_retriever, fake_generator, [MRR()], config
        )
        run = DimRun(run_id="r1", project_name="p", dataset_name="d")

        db_path = tmp_path / "test.db"
        export_to_sqlite(db_path, run, sample_test_cases, report)

        conn = sqlite3.connect(str(db_path))
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert {"dim_runs", "dim_test_cases", "fact_evaluations", "fact_metric_scores"} <= tables

    async def test_row_counts_match(
        self, tmp_path, sample_test_cases, fake_retriever, fake_generator
    ):
        metrics = [PrecisionAtK(k=3), MRR()]
        config = RunConfig(run_id="r1")
        report = await run_evaluation(
            sample_test_cases, fake_retriever, fake_generator, metrics, config
        )
        run = DimRun(run_id="r1", project_name="p", dataset_name="d")

        db_path = tmp_path / "test.db"
        export_to_sqlite(db_path, run, sample_test_cases, report)

        conn = sqlite3.connect(str(db_path))
        assert conn.execute("SELECT COUNT(*) FROM dim_test_cases").fetchone()[0] == len(
            sample_test_cases
        )
        assert conn.execute("SELECT COUNT(*) FROM fact_evaluations").fetchone()[0] == len(
            sample_test_cases
        )
        assert conn.execute("SELECT COUNT(*) FROM fact_metric_scores").fetchone()[0] == len(
            sample_test_cases
        ) * len(metrics)

    async def test_multiple_runs_accumulate_not_overwrite(
        self, tmp_path, sample_test_cases, fake_retriever, fake_generator
    ):
        db_path = tmp_path / "history.db"

        for run_id in ("run-a", "run-b"):
            config = RunConfig(run_id=run_id)
            report = await run_evaluation(
                sample_test_cases, fake_retriever, fake_generator, [MRR()], config
            )
            run = DimRun(run_id=run_id, project_name="p", dataset_name="d")
            export_to_sqlite(db_path, run, sample_test_cases, report)

        conn = sqlite3.connect(str(db_path))
        assert conn.execute("SELECT COUNT(*) FROM dim_runs").fetchone()[0] == 2
        # test cases shared across both runs should be deduped, not doubled
        assert conn.execute("SELECT COUNT(*) FROM dim_test_cases").fetchone()[0] == len(
            sample_test_cases
        )
        assert conn.execute("SELECT COUNT(*) FROM fact_evaluations").fetchone()[0] == len(
            sample_test_cases
        ) * 2

    async def test_creates_parent_directories(
        self, tmp_path, sample_test_cases, fake_retriever, fake_generator
    ):
        config = RunConfig(run_id="r1")
        report = await run_evaluation(
            sample_test_cases, fake_retriever, fake_generator, [MRR()], config
        )
        run = DimRun(run_id="r1", project_name="p", dataset_name="d")

        nested_path = tmp_path / "a" / "b" / "c" / "test.db"
        export_to_sqlite(nested_path, run, sample_test_cases, report)
        assert nested_path.exists()

    async def test_generated_sec_columns_match_ms_columns(
        self, tmp_path, sample_test_cases, fake_retriever, fake_generator
    ):
        config = RunConfig(run_id="r1", telemetry=TelemetryConfig())
        report = await run_evaluation(
            sample_test_cases, fake_retriever, fake_generator, [MRR()], config
        )
        run = DimRun(run_id="r1", project_name="p", dataset_name="d")

        db_path = tmp_path / "test.db"
        export_to_sqlite(db_path, run, sample_test_cases, report)

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT retrieval_latency_ms, retrieval_latency_sec, "
            "generation_latency_ms, generation_latency_sec FROM fact_evaluations"
        ).fetchall()
        for retrieval_ms, retrieval_sec, generation_ms, generation_sec in rows:
            assert retrieval_sec == pytest.approx(retrieval_ms / 1000.0)
            assert generation_sec == pytest.approx(generation_ms / 1000.0)

    async def test_telemetry_fields_exported_when_enabled(
        self, tmp_path, sample_test_cases, fake_retriever, fake_generator
    ):
        config = RunConfig(run_id="r1", telemetry=TelemetryConfig(model_name="gpt-4o-mini"))
        report = await run_evaluation(
            sample_test_cases, fake_retriever, fake_generator, [MRR()], config
        )
        run = DimRun(run_id="r1", project_name="p", dataset_name="d")

        db_path = tmp_path / "test.db"
        export_to_sqlite(db_path, run, sample_test_cases, report)

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT total_tokens, estimated_cost_usd FROM fact_evaluations"
        ).fetchall()
        assert all(tokens is not None and tokens > 0 for tokens, _ in rows)
        assert all(cost is not None and cost > 0 for _, cost in rows)

    async def test_telemetry_fields_null_when_disabled(
        self, tmp_path, sample_test_cases, fake_retriever, fake_generator
    ):
        config = RunConfig(run_id="r1")  # telemetry off
        report = await run_evaluation(
            sample_test_cases, fake_retriever, fake_generator, [MRR()], config
        )
        run = DimRun(run_id="r1", project_name="p", dataset_name="d")

        db_path = tmp_path / "test.db"
        export_to_sqlite(db_path, run, sample_test_cases, report)

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT total_tokens, estimated_cost_usd FROM fact_evaluations"
        ).fetchall()
        assert all(tokens is None for tokens, _ in rows)
        assert all(cost is None for _, cost in rows)
