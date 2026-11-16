"""
Regression/golden tests for agentic trajectory metrics with hand-computed expected values.
"""

from __future__ import annotations

import pytest

from rag_score.agentic.metrics.tool_call_order import ToolCallOrderCorrectness
from rag_score.agentic.metrics.tool_selection_precision import ToolSelectionPrecision
from rag_score.agentic.metrics.tool_selection_recall import ToolSelectionRecall
from rag_score.agentic.types import ToolCall, TrajectoryEvalResult, TrajectoryTestCase


class TestToolSelectionRecallRegression:
    """Golden tests for ToolSelectionRecall."""

    @pytest.mark.asyncio
    async def test_all_expected_tools_called(self):
        """All expected tools called -> 1.0"""
        tc = TrajectoryTestCase(
            question="test",
            expected_tool_sequence=["search", "read", "write"],
        )
        result = TrajectoryEvalResult(
            run_id="run1",
            test_case_id=tc.test_case_id,
            tool_calls=[
                ToolCall(tool_name="search", tool_input={}, tool_output="result1"),
                ToolCall(tool_name="read", tool_input={}, tool_output="result2"),
                ToolCall(tool_name="write", tool_input={}, tool_output="result3"),
            ],
            final_answer="done",
        )
        metric = ToolSelectionRecall()
        score = await metric.score(tc, result)
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_partial_recall(self):
        """2 of 3 expected tools called -> 2/3"""
        tc = TrajectoryTestCase(
            question="test",
            expected_tool_sequence=["search", "read", "write"],
        )
        result = TrajectoryEvalResult(
            run_id="run1",
            test_case_id=tc.test_case_id,
            tool_calls=[
                ToolCall(tool_name="search", tool_input={}, tool_output="result1"),
                ToolCall(tool_name="read", tool_input={}, tool_output="result2"),
            ],
            final_answer="done",
        )
        metric = ToolSelectionRecall()
        score = await metric.score(tc, result)
        assert score == pytest.approx(2.0 / 3.0)

    @pytest.mark.asyncio
    async def test_none_called(self):
        """No tools called -> 0.0"""
        tc = TrajectoryTestCase(
            question="test",
            expected_tool_sequence=["search", "read"],
        )
        result = TrajectoryEvalResult(
            run_id="run1",
            test_case_id=tc.test_case_id,
            tool_calls=[],
            final_answer="done",
        )
        metric = ToolSelectionRecall()
        score = await metric.score(tc, result)
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_extra_tools_do_not_affect_recall(self):
        """Extra tools called don't reduce recall."""
        tc = TrajectoryTestCase(
            question="test",
            expected_tool_sequence=["search", "read"],
        )
        result = TrajectoryEvalResult(
            run_id="run1",
            test_case_id=tc.test_case_id,
            tool_calls=[
                ToolCall(tool_name="search", tool_input={}, tool_output="result1"),
                ToolCall(tool_name="read", tool_input={}, tool_output="result2"),
                ToolCall(tool_name="extra_tool", tool_input={}, tool_output="result3"),
                ToolCall(tool_name="another_extra", tool_input={}, tool_output="result4"),
            ],
            final_answer="done",
        )
        metric = ToolSelectionRecall()
        score = await metric.score(tc, result)
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_no_expected_tools_returns_zero(self):
        """No expected tools -> 0.0 (not 1.0)"""
        tc = TrajectoryTestCase(question="test", expected_tool_sequence=[])
        result = TrajectoryEvalResult(
            run_id="run1",
            test_case_id=tc.test_case_id,
            tool_calls=[ToolCall(tool_name="search", tool_input={}, tool_output="result1")],
            final_answer="done",
        )
        metric = ToolSelectionRecall()
        score = await metric.score(tc, result)
        assert score == 0.0

    def test_name(self):
        assert ToolSelectionRecall().name == "tool_selection_recall"


class TestToolSelectionPrecisionRegression:
    """Golden tests for ToolSelectionPrecision."""

    @pytest.mark.asyncio
    async def test_all_calls_relevant(self):
        """All called tools expected -> 1.0"""
        tc = TrajectoryTestCase(
            question="test",
            expected_tool_sequence=["search", "read", "write"],
        )
        result = TrajectoryEvalResult(
            run_id="run1",
            test_case_id=tc.test_case_id,
            tool_calls=[
                ToolCall(tool_name="search", tool_input={}, tool_output="result1"),
                ToolCall(tool_name="read", tool_input={}, tool_output="result2"),
            ],
            final_answer="done",
        )
        metric = ToolSelectionPrecision()
        score = await metric.score(tc, result)
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_extra_irrelevant_call_lowers_precision(self):
        """1 of 3 calls irrelevant -> 2/3"""
        tc = TrajectoryTestCase(
            question="test",
            expected_tool_sequence=["search", "read"],
        )
        result = TrajectoryEvalResult(
            run_id="run1",
            test_case_id=tc.test_case_id,
            tool_calls=[
                ToolCall(tool_name="search", tool_input={}, tool_output="result1"),
                ToolCall(tool_name="read", tool_input={}, tool_output="result2"),
                ToolCall(tool_name="irrelevant_tool", tool_input={}, tool_output="result3"),
            ],
            final_answer="done",
        )
        metric = ToolSelectionPrecision()
        score = await metric.score(tc, result)
        assert score == pytest.approx(2.0 / 3.0)

    @pytest.mark.asyncio
    async def test_no_tool_calls_returns_zero_not_divide_by_zero(self):
        """No tool calls -> 0.0 (not divide by zero)"""
        tc = TrajectoryTestCase(question="test", expected_tool_sequence=["search"])
        result = TrajectoryEvalResult(
            run_id="run1",
            test_case_id=tc.test_case_id,
            tool_calls=[],
            final_answer="done",
        )
        metric = ToolSelectionPrecision()
        score = await metric.score(tc, result)
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_no_expected_tools_returns_zero(self):
        """No expected tools -> 0.0"""
        tc = TrajectoryTestCase(question="test", expected_tool_sequence=[])
        result = TrajectoryEvalResult(
            run_id="run1",
            test_case_id=tc.test_case_id,
            tool_calls=[ToolCall(tool_name="search", tool_input={}, tool_output="result1")],
            final_answer="done",
        )
        metric = ToolSelectionPrecision()
        score = await metric.score(tc, result)
        assert score == 0.0

    def test_name(self):
        assert ToolSelectionPrecision().name == "tool_selection_precision"


class TestToolCallOrderCorrectnessRegression:
    """Golden tests for ToolCallOrderCorrectness (subsequence-based)."""

    @pytest.mark.asyncio
    async def test_correct_order_scores_one(self):
        """Expected order matches exactly -> 1.0"""
        tc = TrajectoryTestCase(question="test", expected_tool_sequence=["search", "read", "write"])
        result = TrajectoryEvalResult(
            run_id="run1",
            test_case_id=tc.test_case_id,
            tool_calls=[
                ToolCall(tool_name="search", tool_input={}, tool_output="r1"),
                ToolCall(tool_name="read", tool_input={}, tool_output="r2"),
                ToolCall(tool_name="write", tool_input={}, tool_output="r3"),
            ],
            final_answer="done",
        )
        metric = ToolCallOrderCorrectness()
        score = await metric.score(tc, result)
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_wrong_order_scores_zero(self):
        """Wrong relative order -> 0.0"""
        tc = TrajectoryTestCase(question="test", expected_tool_sequence=["search", "read", "write"])
        result = TrajectoryEvalResult(
            run_id="run1",
            test_case_id=tc.test_case_id,
            tool_calls=[
                ToolCall(tool_name="read", tool_input={}, tool_output="r1"),  # read before search
                ToolCall(tool_name="search", tool_input={}, tool_output="r2"),
                ToolCall(tool_name="write", tool_input={}, tool_output="r3"),
            ],
            final_answer="done",
        )
        metric = ToolCallOrderCorrectness()
        score = await metric.score(tc, result)
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_correct_order_with_extra_calls_still_scores_one(self):
        """Extra calls interspersed don't affect score if order preserved."""
        tc = TrajectoryTestCase(question="test", expected_tool_sequence=["search", "read", "write"])
        result = TrajectoryEvalResult(
            run_id="run1",
            test_case_id=tc.test_case_id,
            tool_calls=[
                ToolCall(tool_name="search", tool_input={}, tool_output="r1"),
                ToolCall(tool_name="extra", tool_input={}, tool_output="r2"),
                ToolCall(tool_name="read", tool_input={}, tool_output="r3"),
                ToolCall(tool_name="another_extra", tool_input={}, tool_output="r4"),
                ToolCall(tool_name="write", tool_input={}, tool_output="r5"),
            ],
            final_answer="done",
        )
        metric = ToolCallOrderCorrectness()
        score = await metric.score(tc, result)
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_no_expected_tools_returns_zero(self):
        """No expected tools -> 0.0"""
        tc = TrajectoryTestCase(question="test", expected_tool_sequence=[])
        result = TrajectoryEvalResult(
            run_id="run1",
            test_case_id=tc.test_case_id,
            tool_calls=[ToolCall(tool_name="search", tool_input={}, tool_output="r1")],
            final_answer="done",
        )
        metric = ToolCallOrderCorrectness()
        score = await metric.score(tc, result)
        assert score == 0.0

    def test_name(self):
        assert ToolCallOrderCorrectness().name == "tool_call_order_correctness"


class TestIsSubsequenceRegression:
    """Direct tests of the _is_subsequence helper (private, but tested for regression)."""

    def _is_subsequence(self, expected: list[str], actual: list[str]) -> bool:
        """Copy of the internal helper for testing."""
        it = iter(actual)
        return all(item in it for item in expected)

    def test_exact_match(self):
        assert self._is_subsequence(["a", "b", "c"], ["a", "b", "c"]) is True

    def test_extra_calls_interspersed_still_matches(self):
        assert self._is_subsequence(["a", "b", "c"], ["a", "x", "b", "y", "c"]) is True

    def test_wrong_order_fails(self):
        assert self._is_subsequence(["a", "b", "c"], ["b", "a", "c"]) is False

    def test_missing_element_fails(self):
        assert self._is_subsequence(["a", "b", "c"], ["a", "c"]) is False

    def test_empty_actual_with_nonempty_expected_fails(self):
        assert self._is_subsequence(["a", "b"], []) is False

    def test_duplicates_in_expected_consume_iterator_correctly(self):
        # Expected has duplicate, actual has it twice in order
        assert self._is_subsequence(["a", "a"], ["a", "a"]) is True
        # Expected has duplicate, actual has it once
        assert self._is_subsequence(["a", "a"], ["a"]) is False
        # Expected has duplicate, actual has it three times
        assert self._is_subsequence(["a", "a"], ["a", "a", "a"]) is True