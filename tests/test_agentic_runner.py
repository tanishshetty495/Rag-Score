"""Tests for agentic/runner.py - the trajectory evaluation orchestrator."""

from __future__ import annotations

from rag_score.agentic.adapters import CallableAgentAdapter
from rag_score.agentic.metrics.tool_selection_recall import ToolSelectionRecall
from rag_score.agentic.runner import TrajectoryRunConfig, run_trajectory_evaluation
from rag_score.agentic.types import ToolCall, TrajectoryTestCase


async def _working_agent(query: str):
    return [ToolCall(tool_name="search")], f"answer to {query}"


async def _failing_agent(query: str):
    raise RuntimeError("simulated agent failure")


class TestRunTrajectoryEvaluation:
    async def test_produces_one_result_per_test_case(self):
        test_cases = [
            TrajectoryTestCase(question="q1", expected_tool_sequence=["search"]),
            TrajectoryTestCase(question="q2", expected_tool_sequence=["search"]),
        ]
        agent = CallableAgentAdapter(_working_agent)
        config = TrajectoryRunConfig(run_id="r1")
        report = await run_trajectory_evaluation(test_cases, agent, [ToolSelectionRecall()], config)
        assert len(report.results) == 2

    async def test_produces_scores_for_every_metric_and_result(self):
        test_cases = [TrajectoryTestCase(question="q1", expected_tool_sequence=["search"])]
        agent = CallableAgentAdapter(_working_agent)
        config = TrajectoryRunConfig(run_id="r1")
        report = await run_trajectory_evaluation(test_cases, agent, [ToolSelectionRecall()], config)
        assert len(report.scores) == 1

    async def test_failed_agent_run_is_captured_not_raised(self):
        test_cases = [TrajectoryTestCase(question="q1", expected_tool_sequence=["search"])]
        agent = CallableAgentAdapter(_failing_agent)
        config = TrajectoryRunConfig(run_id="r1", continue_on_error=True)
        report = await run_trajectory_evaluation(test_cases, agent, [ToolSelectionRecall()], config)
        assert len(report.results) == 1
        assert report.results[0].error is not None
        assert "simulated agent failure" in report.results[0].error

    async def test_failed_results_are_not_scored(self):
        test_cases = [TrajectoryTestCase(question="q1", expected_tool_sequence=["search"])]
        agent = CallableAgentAdapter(_failing_agent)
        config = TrajectoryRunConfig(run_id="r1", continue_on_error=True)
        report = await run_trajectory_evaluation(test_cases, agent, [ToolSelectionRecall()], config)
        assert len(report.scores) == 0

    async def test_run_id_propagates_to_all_results(self):
        test_cases = [TrajectoryTestCase(question="q1"), TrajectoryTestCase(question="q2")]
        agent = CallableAgentAdapter(_working_agent)
        config = TrajectoryRunConfig(run_id="my-specific-run-id")
        report = await run_trajectory_evaluation(test_cases, agent, [ToolSelectionRecall()], config)
        assert all(r.run_id == "my-specific-run-id" for r in report.results)

    async def test_empty_dataset_produces_empty_report(self):
        agent = CallableAgentAdapter(_working_agent)
        config = TrajectoryRunConfig(run_id="r1")
        report = await run_trajectory_evaluation([], agent, [ToolSelectionRecall()], config)
        assert report.results == []
        assert report.scores == []

    async def test_latency_is_recorded(self):
        test_cases = [TrajectoryTestCase(question="q1")]
        agent = CallableAgentAdapter(_working_agent)
        config = TrajectoryRunConfig(run_id="r1")
        report = await run_trajectory_evaluation(test_cases, agent, [ToolSelectionRecall()], config)
        assert report.results[0].total_latency_ms is not None

    async def test_tool_calls_and_final_answer_recorded(self):
        test_cases = [TrajectoryTestCase(question="what is x?")]
        agent = CallableAgentAdapter(_working_agent)
        config = TrajectoryRunConfig(run_id="r1")
        report = await run_trajectory_evaluation(test_cases, agent, [ToolSelectionRecall()], config)
        result = report.results[0]
        assert result.tool_calls == [ToolCall(tool_name="search")]
        assert result.final_answer == "answer to what is x?"

    async def test_wrong_order_scenario_end_to_end(self):
        """Regression test for the exact scenario manually verified
        during development: correct tools, wrong order should score
        high on recall/precision but zero on order correctness."""
        from rag_score.agentic.metrics.tool_call_order import ToolCallOrderCorrectness
        from rag_score.agentic.metrics.tool_selection_precision import (
            ToolSelectionPrecision,
        )

        async def wrong_order_agent(query: str):
            calls = [ToolCall(tool_name="calculator"), ToolCall(tool_name="search_weather")]
            return calls, "answer"

        test_cases = [
            TrajectoryTestCase(question="q", expected_tool_sequence=["search_weather", "calculator"])
        ]
        agent = CallableAgentAdapter(wrong_order_agent)
        metrics = [ToolSelectionRecall(), ToolSelectionPrecision(), ToolCallOrderCorrectness()]
        config = TrajectoryRunConfig(run_id="r1")
        report = await run_trajectory_evaluation(test_cases, agent, metrics, config)

        scores = {s.metric_name: s.score_value for s in report.scores}
        assert scores["tool_selection_recall"] == 1.0
        assert scores["tool_selection_precision"] == 1.0
        assert scores["tool_call_order_correctness"] == 0.0
