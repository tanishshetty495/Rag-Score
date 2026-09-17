"""
Trajectory metric base class - parallel to metrics/base.py, but for
TrajectoryTestCase/TrajectoryEvalResult instead of TestCase/EvalResult.

Kept as a separate hierarchy rather than making the existing Metric
generic over both shapes: the two evaluation modes have different
enough inputs (a tool-call sequence vs. retrieved chunks) that sharing
a base class would mean every metric author has to know which shape
they're getting, for no real reuse benefit - the abstract contract
(score → float, optionally with reasoning) is simple enough that
duplicating it here is cheaper than the generic-typing complexity of
sharing one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag_score.agentic.types import TrajectoryEvalResult, TrajectoryTestCase


class TrajectoryMetric(ABC):
    """Base class for every trajectory metric."""

    name: str = "base_trajectory_metric"
    requires_api_key: bool = False

    @abstractmethod
    async def score(self, test_case: TrajectoryTestCase, result: TrajectoryEvalResult) -> float:
        """Return a float score for this one TrajectoryTestCase/Result
        pair. Like Metric.score(), should return 0.0 (not raise) when
        inputs are insufficient (e.g. no expected_tool_sequence)."""
        raise NotImplementedError

    async def score_with_reasoning(
        self, test_case: TrajectoryTestCase, result: TrajectoryEvalResult
    ) -> tuple[float, str | None]:
        """Same pattern as Metric.score_with_reasoning() - most
        trajectory metrics are pure math with nothing to explain, but
        a future LLM-judge trajectory metric can override this."""
        score = await self.score(test_case, result)
        return score, None
