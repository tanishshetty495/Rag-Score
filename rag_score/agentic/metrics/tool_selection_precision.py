"""
Tool Selection Precision - of the tools the agent actually called,
what fraction were expected? Complements Recall: recall asks "did we
get the right tools", precision asks "how much wasted/wrong tool use
was there" - an agent that calls every available tool "just in case"
would score high on recall but low on precision.
"""

from __future__ import annotations

from rag_score.agentic.metrics_base import TrajectoryMetric
from rag_score.agentic.types import TrajectoryEvalResult, TrajectoryTestCase


class ToolSelectionPrecision(TrajectoryMetric):
    name = "tool_selection_precision"
    requires_api_key = False

    async def score(self, test_case: TrajectoryTestCase, result: TrajectoryEvalResult) -> float:
        if not result.tool_calls:
            return 0.0

        expected = set(test_case.expected_tool_sequence)
        if not expected:
            return 0.0

        actual = [call.tool_name for call in result.tool_calls]
        relevant_calls = sum(1 for name in actual if name in expected)
        return relevant_calls / len(actual)
