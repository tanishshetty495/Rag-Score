"""Tests for core/runner.py - the async evaluation orchestrator."""

from __future__ import annotations

from rag_score.core.runner import RunConfig, run_evaluation
from rag_score.metrics.retrieval.mrr import MRR
from rag_score.metrics.retrieval.precision_at_k import PrecisionAtK
from rag_score.telemetry import TelemetryConfig


class TestRunEvaluation:
    async def test_produces_one_result_per_test_case(
        self, sample_test_cases, fake_retriever, fake_generator
    ):
        config = RunConfig(run_id="r1", max_concurrency=4)
        report = await run_evaluation(
            sample_test_cases, fake_retriever, fake_generator, [MRR()], config
        )
        assert len(report.results) == len(sample_test_cases)

    async def test_produces_scores_for_every_metric_and_result(
        self, sample_test_cases, fake_retriever, fake_generator
    ):
        metrics = [PrecisionAtK(k=3), MRR()]
        config = RunConfig(run_id="r1", max_concurrency=4)
        report = await run_evaluation(
            sample_test_cases, fake_retriever, fake_generator, metrics, config
        )
        # 2 test cases x 2 metrics = 4 scores, since both succeed
        assert len(report.scores) == len(sample_test_cases) * len(metrics)

    async def test_failed_retrieval_is_captured_not_raised(
        self, sample_test_cases, failing_retriever, fake_generator
    ):
        config = RunConfig(run_id="r1", max_concurrency=4, continue_on_error=True)
        report = await run_evaluation(
            sample_test_cases, failing_retriever, fake_generator, [MRR()], config
        )
        assert len(report.results) == len(sample_test_cases)
        assert all(r.error is not None for r in report.results)

    async def test_failed_results_are_not_scored(
        self, sample_test_cases, failing_retriever, fake_generator
    ):
        config = RunConfig(run_id="r1", max_concurrency=4, continue_on_error=True)
        report = await run_evaluation(
            sample_test_cases, failing_retriever, fake_generator, [MRR()], config
        )
        # Errors should be skipped, not scored as 0.0
        assert len(report.scores) == 0

    async def test_run_id_propagates_to_all_results(
        self, sample_test_cases, fake_retriever, fake_generator
    ):
        config = RunConfig(run_id="my-specific-run-id", max_concurrency=4)
        report = await run_evaluation(
            sample_test_cases, fake_retriever, fake_generator, [MRR()], config
        )
        assert all(r.run_id == "my-specific-run-id" for r in report.results)

    async def test_empty_dataset_produces_empty_report(self, fake_retriever, fake_generator):
        config = RunConfig(run_id="r1")
        report = await run_evaluation([], fake_retriever, fake_generator, [MRR()], config)
        assert report.results == []
        assert report.scores == []

    async def test_latency_is_recorded(self, sample_test_cases, fake_retriever, fake_generator):
        config = RunConfig(run_id="r1")
        report = await run_evaluation(
            sample_test_cases, fake_retriever, fake_generator, [MRR()], config
        )
        assert all(r.retrieval_latency_ms is not None for r in report.results)
        assert all(r.generation_latency_ms is not None for r in report.results)

    async def test_llm_judge_reasoning_flows_into_metric_score(
        self, sample_test_cases, fake_retriever, fake_generator, fake_judge
    ):
        from rag_score.metrics.generation.faithfulness import Faithfulness

        config = RunConfig(run_id="r1")
        report = await run_evaluation(
            sample_test_cases, fake_retriever, fake_generator,
            [Faithfulness(judge=fake_judge)], config,
        )
        assert len(report.scores) == len(sample_test_cases)
        assert all(s.judge_reasoning == "Looks reasonable." for s in report.scores)

    async def test_pure_math_metric_has_no_reasoning(
        self, sample_test_cases, fake_retriever, fake_generator
    ):
        config = RunConfig(run_id="r1")
        report = await run_evaluation(
            sample_test_cases, fake_retriever, fake_generator, [MRR()], config
        )
        assert all(s.judge_reasoning is None for s in report.scores)

    async def test_on_progress_called_once_per_test_case(
        self, sample_test_cases, fake_retriever, fake_generator
    ):
        progress_calls = []
        config = RunConfig(run_id="r1")
        await run_evaluation(
            sample_test_cases, fake_retriever, fake_generator, [MRR()], config,
            on_progress=lambda: progress_calls.append(1),
        )
        assert len(progress_calls) == len(sample_test_cases)

    async def test_on_progress_called_even_on_failure(
        self, sample_test_cases, failing_retriever, fake_generator
    ):
        # A failed test case still represents forward progress through
        # the dataset - on_progress should fire regardless of success.
        progress_calls = []
        config = RunConfig(run_id="r1", continue_on_error=True)
        await run_evaluation(
            sample_test_cases, failing_retriever, fake_generator, [MRR()], config,
            on_progress=lambda: progress_calls.append(1),
        )
        assert len(progress_calls) == len(sample_test_cases)

    async def test_none_on_progress_does_not_crash(
        self, sample_test_cases, fake_retriever, fake_generator
    ):
        config = RunConfig(run_id="r1")
        report = await run_evaluation(
            sample_test_cases, fake_retriever, fake_generator, [MRR()], config,
        )
        assert len(report.results) == len(sample_test_cases)

    async def test_telemetry_off_by_default_leaves_fields_none(
        self, sample_test_cases, fake_retriever, fake_generator
    ):
        """The critical backward-compatibility check for this feature:
        not opting in must produce byte-for-byte the same behavior as
        before telemetry existed."""
        config = RunConfig(run_id="r1")  # no telemetry= passed
        report = await run_evaluation(
            sample_test_cases, fake_retriever, fake_generator, [MRR()], config
        )
        assert all(r.total_tokens is None for r in report.results)
        assert all(r.estimated_cost_usd is None for r in report.results)

    async def test_telemetry_on_populates_tokens_and_cost(
        self, sample_test_cases, fake_retriever, fake_generator
    ):
        config = RunConfig(run_id="r1", telemetry=TelemetryConfig(model_name="gpt-4o-mini"))
        report = await run_evaluation(
            sample_test_cases, fake_retriever, fake_generator, [MRR()], config
        )
        assert all(r.total_tokens is not None for r in report.results)
        assert all(r.total_tokens > 0 for r in report.results)
        assert all(r.estimated_cost_usd is not None for r in report.results)
        assert all(r.estimated_cost_usd > 0 for r in report.results)

    async def test_telemetry_on_unpriced_model_counts_tokens_but_not_cost(
        self, sample_test_cases, fake_retriever, fake_generator
    ):
        config = RunConfig(run_id="r1", telemetry=TelemetryConfig(model_name="unpriced-model"))
        report = await run_evaluation(
            sample_test_cases, fake_retriever, fake_generator, [MRR()], config
        )
        assert all(r.total_tokens is not None for r in report.results)
        assert all(r.estimated_cost_usd is None for r in report.results)

    async def test_telemetry_does_not_affect_latency_tracking(
        self, sample_test_cases, fake_retriever, fake_generator
    ):
        config = RunConfig(run_id="r1", telemetry=TelemetryConfig())
        report = await run_evaluation(
            sample_test_cases, fake_retriever, fake_generator, [MRR()], config
        )
        assert all(r.retrieval_latency_ms is not None for r in report.results)
        assert all(r.generation_latency_ms is not None for r in report.results)

    async def test_failed_result_has_no_telemetry(
        self, sample_test_cases, failing_retriever, fake_generator
    ):
        config = RunConfig(
            run_id="r1", telemetry=TelemetryConfig(), continue_on_error=True
        )
        report = await run_evaluation(
            sample_test_cases, failing_retriever, fake_generator, [MRR()], config
        )
        # Retrieval failed before generation ran, so there's no answer
        # text to count tokens for - telemetry fields correctly stay
        # unset rather than reporting a nonsensical partial count.
        assert all(r.total_tokens is None for r in report.results)
