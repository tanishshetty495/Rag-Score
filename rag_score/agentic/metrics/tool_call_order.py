"""
Tool Call Order Correctness - were the expected tools called in the
right relative order, allowing extra/unrelated calls in between?

Uses a subsequence check rather than exact-sequence match: an agent
that calls [search, log_event, calculator] when only [search,
calculator] was expected still gets full credit here, since the
expected tools appeared in the right relative order - the extra
logging call isn't a routing mistake. Exact-match would unfairly
punish any agent that does anything beyond the minimum expected
tools, which is often reasonable behavior (retries, logging,
intermediate validation steps) rather than a real error.
"""

from __future__ import annotations

from rag_score.agentic.metrics_base import TrajectoryMetric
from rag_score.agentic.types import TrajectoryEvalResult, TrajectoryTestCase


def _is_subsequence(expected: list[str], actual: list[str]) -> bool:
    """True if `expected` appears in `actual` in order, with any other
    elements allowed in between. Classic subsequence check: consume
    the actual iterator looking for each expected element in turn."""
    it = iter(actual)
    return all(item in it for item in expected)


class ToolCallOrderCorrectness(TrajectoryMetric):
    name = "tool_call_order_correctness"
    requires_api_key = False

    async def score(self, test_case: TrajectoryTestCase, result: TrajectoryEvalResult) -> float:
        expected = test_case.expected_tool_sequence
        if not expected:
            return 0.0

        actual = [call.tool_name for call in result.tool_calls]
        return 1.0 if _is_subsequence(expected, actual) else 0.0
