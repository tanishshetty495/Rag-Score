"""
The evaluation runner.

Fans out across a whole dataset concurrently (bounded by a semaphore so
a rate-limited API or a local machine doesn't get hammered), runs the
user's retriever + generator adapters per TestCase, then scores each
resulting EvalResult against every requested Metric.

This is what turns "a few hundred test cases x an LLM judge" from a
multi-hour sequential slog into something that finishes in minutes.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from rag_score.adapters.base import GeneratorAdapter, RetrieverAdapter
from rag_score.core.types import EvalResult, MetricScore, RetrievedChunk, TestCase
from rag_score.metrics.base import Metric


@dataclass
class RunConfig:
    run_id: str
    project_name: str = "default"
    top_k: int = 5
    max_concurrency: int = 8
    # If a single test case's retrieve/generate call raises, log it into
    # EvalResult.error and keep going rather than aborting the whole run.
    continue_on_error: bool = True


@dataclass
class RunReport:
    run_id: str
    results: list[EvalResult] = field(default_factory=list)
    scores: list[MetricScore] = field(default_factory=list)


async def _run_single(
    test_case: TestCase,
    retriever: RetrieverAdapter,
    generator: GeneratorAdapter,
    config: RunConfig,
    semaphore: asyncio.Semaphore,
    on_progress: Callable[[], None] | None,
) -> EvalResult:
    async with semaphore:
        result = EvalResult(run_id=config.run_id, test_case_id=test_case.test_case_id)
        try:
            t0 = time.perf_counter()
            context: list[RetrievedChunk] = await retriever.retrieve(
                test_case.question, top_k=config.top_k
            )
            result.retrieval_latency_ms = (time.perf_counter() - t0) * 1000
            result.retrieved_context = context

            t1 = time.perf_counter()
            answer = await generator.generate(test_case.question, context)
            result.generation_latency_ms = (time.perf_counter() - t1) * 1000
            result.generated_answer = answer

        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            if not config.continue_on_error:
                raise
        finally:
            # Report progress once retrieval+generation is done for this
            # test case, regardless of success/failure - a failed case
            # still represents forward progress through the dataset, and
            # scoring (the second pass) is comparatively fast so isn't
            # tracked separately.
            if on_progress is not None:
                on_progress()

        return result


async def _score_result(
    test_case: TestCase, result: EvalResult, metrics: list[Metric]
) -> list[MetricScore]:
    if result.error is not None:
        # Don't score failed runs - a 0.0 would silently pollute averages
        # and look like a genuinely bad answer rather than a crash.
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


async def run_evaluation(
    test_cases: list[TestCase],
    retriever: RetrieverAdapter,
    generator: GeneratorAdapter,
    metrics: list[Metric],
    config: RunConfig,
    on_progress: Callable[[], None] | None = None,
) -> RunReport:
    """Run every TestCase through the pipeline and score it against every metric.

    Two concurrency-bounded fan-out passes:
      1. retrieve + generate for every test case
      2. score every resulting EvalResult against every metric
    kept separate (rather than interleaved) so a slow LLM judge doesn't
    block the next test case's retrieval/generation from starting.

    on_progress, if given, is called once (synchronously, with no
    arguments) each time a test case finishes its retrieve+generate
    phase - e.g. `click.progressbar`'s `.update(1)` bound method. Kept
    as a plain callback rather than an async generator/queue so callers
    that don't care about progress pay zero overhead.
    """
    semaphore = asyncio.Semaphore(config.max_concurrency)

    eval_results = await asyncio.gather(
        *(
            _run_single(tc, retriever, generator, config, semaphore, on_progress)
            for tc in test_cases
        )
    )

    tc_by_id = {tc.test_case_id: tc for tc in test_cases}
    scored_lists = await asyncio.gather(
        *(
            _score_result(tc_by_id[r.test_case_id], r, metrics)
            for r in eval_results
        )
    )
    all_scores = [s for sublist in scored_lists for s in sublist]

    return RunReport(run_id=config.run_id, results=list(eval_results), scores=all_scores)
