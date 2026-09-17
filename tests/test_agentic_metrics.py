"""Tests for agentic/metrics/ - tool selection recall, precision, and
order correctness."""

from __future__ import annotations

from rag_score.agentic.metrics.tool_call_order import (
    ToolCallOrderCorrectness,
    _is_subsequence,
)
from rag_score.agentic.metrics.tool_selection_precision import ToolSelectionPrecision
from rag_score.agentic.metrics.tool_selection_recall import ToolSelectionRecall
from rag_score.agentic.types import ToolCall, TrajectoryEvalResult, TrajectoryTestCase


def _result(tool_names: list[str]) -> TrajectoryEvalResult:
    return TrajectoryEvalResult(
        run_id="r", test_case_id="tc",
        tool_calls=[ToolCall(tool_name=n) for n in tool_names],
    )


class TestToolSelectionRecall:
    async def test_all_expected_tools_called(self):
        tc = TrajectoryTestCase(question="q", expected_tool_sequence=["search", "calc"])
        result = _result(["search", "calc"])
        score = await ToolSelectionRecall().score(tc, result)
        assert score == 1.0

    async def test_partial_recall(self):
        tc = TrajectoryTestCase(question="q", expected_tool_sequence=["search", "calc"])
        result = _result(["search"])
        score = await ToolSelectionRecall().score(tc, result)
        assert score == 0.5

    async def test_none_called(self):
        tc = TrajectoryTestCase(question="q", expected_tool_sequence=["search"])
        result = _result([])
        score = await ToolSelectionRecall().score(tc, result)
        assert score == 0.0

    async def test_extra_tools_do_not_affect_recall(self):
        tc = TrajectoryTestCase(question="q", expected_tool_sequence=["search"])
        result = _result(["search", "log", "extra_tool"])
        score = await ToolSelectionRecall().score(tc, result)
        assert score == 1.0

    async def test_no_expected_tools_returns_zero(self):
        tc = TrajectoryTestCase(question="q")
        result = _result(["search"])
        score = await ToolSelectionRecall().score(tc, result)
        assert score == 0.0

    def test_name(self):
        assert ToolSelectionRecall().name == "tool_selection_recall"


class TestToolSelectionPrecision:
    async def test_all_calls_relevant(self):
        tc = TrajectoryTestCase(question="q", expected_tool_sequence=["search", "calc"])
        result = _result(["search", "calc"])
        score = await ToolSelectionPrecision().score(tc, result)
        assert score == 1.0

    async def test_extra_irrelevant_call_lowers_precision(self):
        tc = TrajectoryTestCase(question="q", expected_tool_sequence=["search"])
        result = _result(["search", "unrelated_tool"])
        score = await ToolSelectionPrecision().score(tc, result)
        assert score == 0.5

    async def test_no_tool_calls_returns_zero_not_divide_by_zero(self):
        tc = TrajectoryTestCase(question="q", expected_tool_sequence=["search"])
        result = _result([])
        score = await ToolSelectionPrecision().score(tc, result)
        assert score == 0.0

    async def test_no_expected_tools_returns_zero(self):
        tc = TrajectoryTestCase(question="q")
        result = _result(["search"])
        score = await ToolSelectionPrecision().score(tc, result)
        assert score == 0.0

    def test_name(self):
        assert ToolSelectionPrecision().name == "tool_selection_precision"


class TestIsSubsequence:
    """The core algorithm ToolCallOrderCorrectness relies on - tested
    directly since it's easy to get subtly wrong with iterator
    consumption and duplicate handling."""

    def test_exact_match(self):
        assert _is_subsequence(["a", "b"], ["a", "b"]) is True

    def test_extra_calls_interspersed_still_matches(self):
        assert _is_subsequence(["a", "b"], ["a", "x", "b"]) is True
        assert _is_subsequence(["a", "b"], ["x", "a", "x", "b", "x"]) is True

    def test_wrong_order_fails(self):
        assert _is_subsequence(["a", "b"], ["b", "a"]) is False

    def test_missing_element_fails(self):
        assert _is_subsequence(["a", "b"], ["a", "x"]) is False

    def test_empty_actual_with_nonempty_expected_fails(self):
        assert _is_subsequence(["a"], []) is False

    def test_duplicates_in_expected_consume_iterator_correctly(self):
        assert _is_subsequence(["a", "a"], ["a", "x", "a"]) is True
        assert _is_subsequence(["a", "a"], ["a", "x"]) is False


class TestToolCallOrderCorrectness:
    async def test_correct_order_scores_one(self):
        tc = TrajectoryTestCase(question="q", expected_tool_sequence=["search", "calc"])
        result = _result(["search", "calc"])
        score = await ToolCallOrderCorrectness().score(tc, result)
        assert score == 1.0

    async def test_wrong_order_scores_zero(self):
        tc = TrajectoryTestCase(question="q", expected_tool_sequence=["search", "calc"])
        result = _result(["calc", "search"])
        score = await ToolCallOrderCorrectness().score(tc, result)
        assert score == 0.0

    async def test_correct_order_with_extra_calls_still_scores_one(self):
        tc = TrajectoryTestCase(question="q", expected_tool_sequence=["search", "calc"])
        result = _result(["search", "log_event", "calc"])
        score = await ToolCallOrderCorrectness().score(tc, result)
        assert score == 1.0

    async def test_no_expected_tools_returns_zero(self):
        tc = TrajectoryTestCase(question="q")
        result = _result(["search"])
        score = await ToolCallOrderCorrectness().score(tc, result)
        assert score == 0.0

    def test_name(self):
        assert ToolCallOrderCorrectness().name == "tool_call_order_correctness"
