"""
Tool Selection Recall - of the tools the test case expected to be
called, what fraction actually were (anywhere in the trajectory,
order not considered)?

Mirrors recall_at_k's naming and semantics deliberately: same
question ("did we get the right things"), just over a set of tool
names instead of a set of retrieved doc IDs.
"""

from __future__ import annotations

from rag_score.agentic.metrics_base import TrajectoryMetric
from rag_score.agentic.types import TrajectoryEvalResult, TrajectoryTestCase


class ToolSelectionRecall(TrajectoryMetric):
    name = "tool_selection_recall"
    requires_api_key = False

    async def score(self, test_case: TrajectoryTestCase, result: TrajectoryEvalResult) -> float:
        expected = set(test_case.expected_tool_sequence)
        if not expected:
            return 0.0

        actual = {call.tool_name for call in result.tool_calls}
        return len(expected & actual) / len(expected)
