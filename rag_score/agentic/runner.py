"""
The trajectory evaluation runner.

Structurally similar to core/runner.py (concurrency-bounded fan-out,
isolate failures per test case, score in a separate pass) but adapted
for the agentic shape: one call to agent.run() per test case instead
of a separate retrieve+generate pair, since the agent's internal
tool-calling loop isn't something this library orchestrates.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from rag_score.agentic.adapters import AgentAdapter
from rag_score.agentic.metrics_base import TrajectoryMetric
from rag_score.agentic.types import TrajectoryEvalResult, TrajectoryTestCase
from rag_score.core.types import MetricScore


@dataclass
class TrajectoryRunConfig:
    run_id: str
    project_name: str = "default"
    max_concurrency: int = 8
    continue_on_error: bool = True


@dataclass
class TrajectoryRunReport:
    run_id: str
    results: list[TrajectoryEvalResult] = field(default_factory=list)
    scores: list[MetricScore] = field(default_factory=list)


async def _run_single(
    test_case: TrajectoryTestCase,
    agent: AgentAdapter,
    config: TrajectoryRunConfig,
    semaphore: asyncio.Semaphore,
    on_progress: Callable[[], None] | None,
) -> TrajectoryEvalResult:
    async with semaphore:
        result = TrajectoryEvalResult(run_id=config.run_id, test_case_id=test_case.test_case_id)
        try:
            t0 = time.perf_counter()
            tool_calls, final_answer = await agent.run(test_case.question)
            result.total_latency_ms = (time.perf_counter() - t0) * 1000
            result.tool_calls = tool_calls
            result.final_answer = final_answer
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            if not config.continue_on_error:
                raise
        finally:
            if on_progress is not None:
                on_progress()
        return result


async def _score_result(
    test_case: TrajectoryTestCase, result: TrajectoryEvalResult, metrics: list[TrajectoryMetric]
) -> list[MetricScore]:
    if result.error is not None:
        return []

    scored = await asyncio.gather(*(m.score_with_reasoning(test_case, result) for m in metrics))
    return [
        MetricScore(
            evaluation_id=result.evaluation_id,
            metric_name=m.name,
            score_value=score,
            judge_reasoning=reasoning,
        )
        for m, (score, reasoning) in zip(metrics, scored)
    ]


async def run_trajectory_evaluation(
    test_cases: list[TrajectoryTestCase],
    agent: AgentAdapter,
    metrics: list[TrajectoryMetric],
    config: TrajectoryRunConfig,
    on_progress: Callable[[], None] | None = None,
) -> TrajectoryRunReport:
    """Run every TrajectoryTestCase through the agent once and score
    the resulting trajectory against every metric. Same two-pass
    structure as run_evaluation(): agent runs first (concurrency-bounded),
    then scoring, so a slow metric doesn't block the next test case's
    agent run from starting.

    on_progress, if given, is called once (no arguments) each time a
    test case's agent run completes - see run_evaluation()'s docstring
    for the same parameter.
    """
    semaphore = asyncio.Semaphore(config.max_concurrency)

    eval_results = await asyncio.gather(
        *(_run_single(tc, agent, config, semaphore, on_progress) for tc in test_cases)
    )

    tc_by_id = {tc.test_case_id: tc for tc in test_cases}
    scored_lists = await asyncio.gather(
        *(_score_result(tc_by_id[r.test_case_id], r, metrics) for r in eval_results)
    )
    all_scores = [s for sublist in scored_lists for s in sublist]

    return TrajectoryRunReport(run_id=config.run_id, results=list(eval_results), scores=all_scores)
