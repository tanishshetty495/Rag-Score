"""
Regression/golden tests for export functionality:
- SQLite schema validation
- DataFrame shapes and columns
- HTML report generation
"""

from __future__ import annotations

import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from rag_score.core.runner import RunReport
from rag_score.core.types import (
    DimRun,
    DimTestCase,
    EvalResult,
    FactEvaluation,
    FactMetricScore,
    MetricScore,
    RetrievedChunk,
    TestCase,
)
from rag_score.export.sqlite_export import export_to_sqlite


class TestSQLiteExportRegression:
    """Tests for SQLite export."""

    def create_sample_data(self):
        """Create sample data for testing export."""
        tc = TestCase(
            question="test query",
            ground_truth_answer="expected answer",
            expected_doc_ids=["doc1", "doc2"],
            dataset_name="test_dataset",
        )

        retrieved = [
            RetrievedChunk(doc_id="doc1", text="chunk 1", score=0.9),
            RetrievedChunk(doc_id="doc3", text="chunk 3", score=0.7),
        ]

        eval_result = EvalResult(
            run_id="test_run",
            test_case_id=tc.test_case_id,
            retrieved_context=retrieved,
            generated_answer="generated answer",
            retrieval_latency_ms=100.0,
            generation_latency_ms=500.0,
        )

        metrics = [
            MetricScore(
                evaluation_id=eval_result.evaluation_id,
                metric_name="precision_at_5",
                score_value=0.5,
            ),
            MetricScore(
                evaluation_id=eval_result.evaluation_id,
                metric_name="recall_at_5",
                score_value=1.0,
            ),
            MetricScore(
                evaluation_id=eval_result.evaluation_id,
                metric_name="faithfulness",
                score_value=0.8,
            ),
        ]

        run = DimRun(
            run_id="test_run",
            run_timestamp=datetime.now(timezone.utc),
            project_name="test_project",
            dataset_name="test_dataset",
            retriever_config="{}",
            generator_config="{}",
            environment="test",
        )

        report = RunReport(
            run_id="test_run",
            results=[eval_result],
            scores=metrics,
        )

        test_cases = [tc]

        return run, test_cases, report

    def test_export_creates_database(self):
        """Export creates a valid SQLite database."""
        run, test_cases, report = self.create_sample_data()

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            export_to_sqlite(db_path, run, test_cases, report)

            # Verify file exists and is a valid SQLite database
            assert Path(db_path).exists()
            assert Path(db_path).stat().st_size > 0

            # Verify we can connect and query
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cursor.fetchall()}
            assert "dim_runs" in tables
            assert "dim_test_cases" in tables
            assert "fact_evaluations" in tables
            assert "fact_metric_scores" in tables

            conn.close()
        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_dim_runs_table_schema(self):
        """Dim runs table has correct schema and data."""
        run, test_cases, report = self.create_sample_data()

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            export_to_sqlite(db_path, run, test_cases, report)

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check columns
            cursor.execute("PRAGMA table_info(dim_runs)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}
            expected_columns = {
                "run_id", "run_timestamp", "project_name", "dataset_name",
                "retriever_config", "generator_config", "environment"
            }
            assert set(columns.keys()) == expected_columns

            # Check data
            cursor.execute("SELECT * FROM dim_runs")
            rows = cursor.fetchall()
            assert len(rows) == 1
            row = rows[0]
            assert row[0] == "test_run"  # run_id
            assert row[2] == "test_project"  # project_name
            assert row[3] == "test_dataset"  # dataset_name

            conn.close()
        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_dim_test_cases_table_schema(self):
        """Dim test cases table has correct schema and data."""
        run, test_cases, report = self.create_sample_data()

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            export_to_sqlite(db_path, run, test_cases, report)

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check columns
            cursor.execute("PRAGMA table_info(dim_test_cases)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}
            expected_columns = {
                "test_case_id", "dataset_name", "question", "ground_truth_answer",
                "expected_doc_ids", "difficulty_category"
            }
            assert set(columns.keys()) == expected_columns

            # Check data
            cursor.execute("SELECT * FROM dim_test_cases")
            rows = cursor.fetchall()
            assert len(rows) == 1
            row = rows[0]
            assert row[2] == "test query"  # question
            assert row[3] == "expected answer"  # ground_truth_answer
            assert "doc1" in row[4]  # expected_doc_ids (JSON)

            conn.close()
        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_fact_evaluations_table_schema(self):
        """Fact evaluations table has correct schema and data."""
        run, test_cases, report = self.create_sample_data()

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            export_to_sqlite(db_path, run, test_cases, report)

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check columns
            cursor.execute("PRAGMA table_info(fact_evaluations)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}
            expected_columns = {
                "evaluation_id", "run_id", "test_case_id", "retrieved_context",
                "generated_answer", "retrieval_latency_ms", "generation_latency_ms",
                "total_tokens", "estimated_cost_usd"
            }
            assert set(columns.keys()) == expected_columns

            # Check data
            cursor.execute("SELECT * FROM fact_evaluations")
            rows = cursor.fetchall()
            assert len(rows) == 1
            row = rows[0]
            assert row[1] == "test_run"  # run_id
            assert row[4] == "generated answer"  # generated_answer
            assert row[5] == 100.0  # retrieval_latency_ms
            assert row[6] == 500.0  # generation_latency_ms

            conn.close()
        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_fact_metric_scores_table_schema(self):
        """Fact metric scores table has correct schema and data."""
        run, test_cases, report = self.create_sample_data()

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            export_to_sqlite(db_path, run, test_cases, report)

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check columns
            cursor.execute("PRAGMA table_info(fact_metric_scores)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}
            expected_columns = {
                "score_id", "evaluation_id", "metric_name", "score_value", "judge_reasoning"
            }
            assert set(columns.keys()) == expected_columns

            # Check data - should have 3 metrics
            cursor.execute("SELECT metric_name, score_value FROM fact_metric_scores ORDER BY metric_name")
            rows = cursor.fetchall()
            assert len(rows) == 3
            metric_dict = {row[0]: row[1] for row in rows}
            assert metric_dict["faithfulness"] == 0.8
            assert metric_dict["precision_at_5"] == 0.5
            assert metric_dict["recall_at_5"] == 1.0

            conn.close()
        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_foreign_keys_work(self):
        """Foreign key relationships work correctly."""
        run, test_cases, report = self.create_sample_data()

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            export_to_sqlite(db_path, run, test_cases, report)

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Enable foreign key checking
            cursor.execute("PRAGMA foreign_keys = ON")

            # This should not raise an error
            cursor.execute("PRAGMA foreign_key_check")
            violations = cursor.fetchall()
            assert len(violations) == 0

            conn.close()
        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_multiple_results_export(self):
        """Exporting multiple test cases works correctly."""
        tc1 = TestCase(question="query 1", expected_doc_ids=["doc1"], dataset_name="test")
        tc2 = TestCase(question="query 2", expected_doc_ids=["doc2"], dataset_name="test")

        eval1 = EvalResult(
            run_id="test_run",
            test_case_id=tc1.test_case_id,
            retrieved_context=[RetrievedChunk(doc_id="doc1", text="chunk 1")],
            generated_answer="answer 1",
        )
        eval2 = EvalResult(
            run_id="test_run",
            test_case_id=tc2.test_case_id,
            retrieved_context=[RetrievedChunk(doc_id="doc2", text="chunk 2")],
            generated_answer="answer 2",
        )

        metrics = [
            MetricScore(evaluation_id=eval1.evaluation_id, metric_name="precision_at_5", score_value=1.0),
            MetricScore(evaluation_id=eval2.evaluation_id, metric_name="precision_at_5", score_value=0.5),
        ]

        run = DimRun(
            run_id="test_run",
            run_timestamp=datetime.now(timezone.utc),
            project_name="test_project",
            dataset_name="test",
            retriever_config="{}",
            generator_config="{}",
            environment="test",
        )

        report = RunReport(run_id="test_run", results=[eval1, eval2], scores=metrics)
        test_cases = [tc1, tc2]

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            export_to_sqlite(db_path, run, test_cases, report)

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM dim_test_cases")
            assert cursor.fetchone()[0] == 2

            cursor.execute("SELECT COUNT(*) FROM fact_evaluations")
            assert cursor.fetchone()[0] == 2

            cursor.execute("SELECT COUNT(*) FROM fact_metric_scores")
            assert cursor.fetchone()[0] == 2

            cursor.execute("SELECT COUNT(*) FROM dim_runs")
            assert cursor.fetchone()[0] == 1

            conn.close()
        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_empty_results_export(self):
        """Exporting empty results list works."""
        run = DimRun(
            run_id="empty_run",
            run_timestamp=datetime.now(timezone.utc),
            project_name="test_project",
            dataset_name="test",
            retriever_config="{}",
            generator_config="{}",
            environment="test",
        )

        test_cases = []
        report = RunReport(run_id="empty_run", results=[], scores=[])

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            export_to_sqlite(db_path, run, test_cases, report)

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM dim_test_cases")
            assert cursor.fetchone()[0] == 0

            cursor.execute("SELECT COUNT(*) FROM fact_evaluations")
            assert cursor.fetchone()[0] == 0

            cursor.execute("SELECT COUNT(*) FROM fact_metric_scores")
            assert cursor.fetchone()[0] == 0

            cursor.execute("SELECT COUNT(*) FROM dim_runs")
            assert cursor.fetchone()[0] == 1

            conn.close()
        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_results_with_errors(self):
        """Export handles results with errors."""
        tc = TestCase(question="test", expected_doc_ids=[], dataset_name="test")

        eval_result = EvalResult(
            run_id="test_run",
            test_case_id=tc.test_case_id,
            retrieved_context=[],
            generated_answer="",
            error="Retrieval failed: connection timeout",
        )

        run = DimRun(
            run_id="test_run",
            run_timestamp=datetime.now(timezone.utc),
            project_name="test_project",
            dataset_name="test",
            retriever_config="{}",
            generator_config="{}",
            environment="test",
        )

        report = RunReport(run_id="test_run", results=[eval_result], scores=[])
        test_cases = [tc]

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            export_to_sqlite(db_path, run, test_cases, report)

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # The error is not stored in fact_evaluations, but the row exists
            cursor.execute("SELECT COUNT(*) FROM fact_evaluations")
            assert cursor.fetchone()[0] == 1

            conn.close()
        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_multiple_runs_append(self):
        """Multiple runs can be appended to the same database."""
        run1, test_cases1, report1 = self.create_sample_data()
        run1.run_id = "run_1"

        # Create second run
        run2 = DimRun(
            run_id="run_2",
            run_timestamp=datetime.now(timezone.utc),
            project_name="test_project",
            dataset_name="test_dataset",
            retriever_config="{}",
            generator_config="{}",
            environment="test",
        )
        tc2 = TestCase(question="query 2", expected_doc_ids=["doc2"], dataset_name="test_dataset")
        eval2 = EvalResult(
            run_id="run_2",
            test_case_id=tc2.test_case_id,
            retrieved_context=[RetrievedChunk(doc_id="doc2", text="chunk 2")],
            generated_answer="answer 2",
        )
        report2 = RunReport(run_id="run_2", results=[eval2], scores=[])
        test_cases2 = [tc2]

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            # First run
            export_to_sqlite(db_path, run1, test_cases1, report1)
            # Second run
            export_to_sqlite(db_path, run2, test_cases2, report2)

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM dim_runs")
            assert cursor.fetchone()[0] == 2

            cursor.execute("SELECT run_id FROM dim_runs ORDER BY run_id")
            runs = [row[0] for row in cursor.fetchall()]
            assert "run_1" in runs
            assert "run_2" in runs

            conn.close()
        finally:
            Path(db_path).unlink(missing_ok=True)


class TestHTMLReportRegression:
    """Tests for HTML report generation."""

    def test_generate_html_report(self):
        """HTML report generation works."""
        from rag_score.report.html_report import generate_html_report

        tc = TestCase(question="test", expected_doc_ids=["doc1"])

        eval_result = EvalResult(
            run_id="test_run",
            test_case_id=tc.test_case_id,
            retrieved_context=[RetrievedChunk(doc_id="doc1", text="chunk")],
            generated_answer="answer",
        )

        metrics = [
            MetricScore(evaluation_id=eval_result.evaluation_id, metric_name="precision_at_5", score_value=1.0)
        ]

        report = RunReport(run_id="test_run", results=[eval_result], scores=metrics)
        test_cases = [tc]

        run = DimRun(
            run_id="test_run",
            run_timestamp=datetime.now(timezone.utc),
            project_name="test_project",
            dataset_name="test_dataset",
            retriever_config="{}",
            generator_config="{}",
            environment="test",
        )

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            html_path = f.name

        try:
            generate_html_report(html_path, run, test_cases, report)

            assert Path(html_path).exists()
            content = Path(html_path).read_text()
            assert "<html" in content.lower()
            assert "precision_at_5" in content
            assert "1.000" in content or "1.0" in content
        finally:
            Path(html_path).unlink(missing_ok=True)

    def test_html_report_with_multiple_metrics(self):
        """HTML report includes all metrics."""
        from rag_score.report.html_report import generate_html_report

        tc = TestCase(question="test", expected_doc_ids=["doc1"])

        eval_result = EvalResult(
            run_id="test_run",
            test_case_id=tc.test_case_id,
            retrieved_context=[RetrievedChunk(doc_id="doc1", text="chunk")],
            generated_answer="answer",
        )

        metrics = [
            MetricScore(evaluation_id=eval_result.evaluation_id, metric_name="precision_at_5", score_value=1.0),
            MetricScore(evaluation_id=eval_result.evaluation_id, metric_name="recall_at_5", score_value=0.5),
            MetricScore(evaluation_id=eval_result.evaluation_id, metric_name="faithfulness", score_value=0.8),
        ]

        report = RunReport(run_id="test_run", results=[eval_result], scores=metrics)
        test_cases = [tc]

        run = DimRun(
            run_id="test_run",
            run_timestamp=datetime.now(timezone.utc),
            project_name="test_project",
            dataset_name="test_dataset",
            retriever_config="{}",
            generator_config="{}",
            environment="test",
        )

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            html_path = f.name

        try:
            generate_html_report(html_path, run, test_cases, report)

            content = Path(html_path).read_text()
            assert "precision_at_5" in content
            assert "recall_at_5" in content
            assert "faithfulness" in content
        finally:
            Path(html_path).unlink(missing_ok=True)