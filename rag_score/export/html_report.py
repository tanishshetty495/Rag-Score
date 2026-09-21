"""Tests for report/html_report.py."""

from __future__ import annotations

from rag_score.core.runner import RunConfig, run_evaluation
from rag_score.core.types import DimRun
from rag_score.metrics.generation.faithfulness import Faithfulness
from rag_score.metrics.retrieval.precision_at_k import PrecisionAtK
from rag_score.report.html_report import generate_html_report
from rag_score.telemetry import TelemetryConfig


class TestGenerateHtmlReport:
    async def test_writes_a_file(
        self, tmp_path, sample_test_cases, fake_retriever, fake_generator
    ):
        config = RunConfig(run_id="r1")
        report = await run_evaluation(
            sample_test_cases, fake_retriever, fake_generator, [PrecisionAtK(k=3)], config
        )
        run = DimRun(run_id="r1", project_name="p", dataset_name="d")

        output_path = tmp_path / "report.html"
        written = generate_html_report(output_path, run, sample_test_cases, report)
        assert written.exists()
        assert written == output_path

    async def test_contains_run_id_and_questions(
        self, tmp_path, sample_test_cases, fake_retriever, fake_generator
    ):
        config = RunConfig(run_id="my-unique-run-id")
        report = await run_evaluation(
            sample_test_cases, fake_retriever, fake_generator, [PrecisionAtK(k=3)], config
        )
        run = DimRun(run_id="my-unique-run-id", project_name="p", dataset_name="d")

        output_path = tmp_path / "report.html"
        generate_html_report(output_path, run, sample_test_cases, report)

        html = output_path.read_text(encoding="utf-8")
        assert "my-unique-run-id" in html
        for tc in sample_test_cases:
            assert tc.question in html

    async def test_judge_reasoning_appears_as_tooltip(
        self, tmp_path, sample_test_cases, fake_retriever, fake_generator, fake_judge
    ):
        config = RunConfig(run_id="r1")
        report = await run_evaluation(
            sample_test_cases, fake_retriever, fake_generator,
            [Faithfulness(judge=fake_judge)], config,
        )
        run = DimRun(run_id="r1", project_name="p", dataset_name="d")

        output_path = tmp_path / "report.html"
        generate_html_report(output_path, run, sample_test_cases, report)

        html = output_path.read_text(encoding="utf-8")
        assert 'title="Looks reasonable."' in html

    async def test_error_rows_show_error_badge_not_scores(
        self, tmp_path, sample_test_cases, failing_retriever, fake_generator
    ):
        config = RunConfig(run_id="r1", continue_on_error=True)
        report = await run_evaluation(
            sample_test_cases, failing_retriever, fake_generator, [PrecisionAtK(k=3)], config
        )
        run = DimRun(run_id="r1", project_name="p", dataset_name="d")

        output_path = tmp_path / "report.html"
        generate_html_report(output_path, run, sample_test_cases, report)

        html = output_path.read_text(encoding="utf-8")
        assert "badge-error" in html

    async def test_creates_parent_directories(
        self, tmp_path, sample_test_cases, fake_retriever, fake_generator
    ):
        config = RunConfig(run_id="r1")
        report = await run_evaluation(
            sample_test_cases, fake_retriever, fake_generator, [PrecisionAtK(k=3)], config
        )
        run = DimRun(run_id="r1", project_name="p", dataset_name="d")

        nested_path = tmp_path / "a" / "b" / "report.html"
        generate_html_report(nested_path, run, sample_test_cases, report)
        assert nested_path.exists()

    async def test_chart_svg_present_with_scores(
        self, tmp_path, sample_test_cases, fake_retriever, fake_generator
    ):
        config = RunConfig(run_id="r1")
        report = await run_evaluation(
            sample_test_cases, fake_retriever, fake_generator, [PrecisionAtK(k=3)], config
        )
        run = DimRun(run_id="r1", project_name="p", dataset_name="d")

        output_path = tmp_path / "report.html"
        generate_html_report(output_path, run, sample_test_cases, report)

        html = output_path.read_text(encoding="utf-8")
        assert "<svg" in html
        assert "precision_at_3" in html

    async def test_no_chart_section_when_no_scores(
        self, tmp_path, sample_test_cases, failing_retriever, fake_generator
    ):
        # All results errored -> no scores -> chart should be skipped
        # entirely rather than rendering an empty/broken SVG.
        config = RunConfig(run_id="r1", continue_on_error=True)
        report = await run_evaluation(
            sample_test_cases, failing_retriever, fake_generator, [PrecisionAtK(k=3)], config
        )
        run = DimRun(run_id="r1", project_name="p", dataset_name="d")

        output_path = tmp_path / "report.html"
        generate_html_report(output_path, run, sample_test_cases, report)

        html = output_path.read_text(encoding="utf-8")
        assert "<svg" not in html

    async def test_telemetry_cards_present_when_enabled(
        self, tmp_path, sample_test_cases, fake_retriever, fake_generator
    ):
        config = RunConfig(run_id="r1", telemetry=TelemetryConfig(model_name="gpt-4o-mini"))
        report = await run_evaluation(
            sample_test_cases, fake_retriever, fake_generator, [PrecisionAtK(k=3)], config
        )
        run = DimRun(run_id="r1", project_name="p", dataset_name="d")

        output_path = tmp_path / "report.html"
        generate_html_report(output_path, run, sample_test_cases, report)

        html = output_path.read_text(encoding="utf-8")
        assert "Total tokens" in html
        assert "Est. cost" in html

    async def test_telemetry_cards_absent_when_disabled(
        self, tmp_path, sample_test_cases, fake_retriever, fake_generator
    ):
        config = RunConfig(run_id="r1")  # telemetry off
        report = await run_evaluation(
            sample_test_cases, fake_retriever, fake_generator, [PrecisionAtK(k=3)], config
        )
        run = DimRun(run_id="r1", project_name="p", dataset_name="d")

        output_path = tmp_path / "report.html"
        generate_html_report(output_path, run, sample_test_cases, report)

        html = output_path.read_text(encoding="utf-8")
        assert "Total tokens" not in html
        assert "Est. cost" not in html
