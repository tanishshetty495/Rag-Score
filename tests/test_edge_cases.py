"""
Regression/golden tests for edge cases:
- Empty retrieval
- No ground truth
- Duplicate doc_ids
- k > retrieved
- k = 0
- Empty answer
- Malformed JSON from judge
- Unicode handling
- Long text handling
- Zero test cases
"""

from __future__ import annotations

import pytest

from rag_score.agentic.metrics.tool_call_order import ToolCallOrderCorrectness
from rag_score.agentic.metrics.tool_selection_precision import ToolSelectionPrecision
from rag_score.agentic.metrics.tool_selection_recall import ToolSelectionRecall
from rag_score.agentic.types import ToolCall, TrajectoryEvalResult, TrajectoryTestCase
from rag_score.core.models import EvaluationCase
from rag_score.core.types import EvalResult, RetrievedChunk, TestCase
from rag_score.metrics.generation.citation_completeness import CitationCompleteness
from rag_score.metrics.generation.citation_correctness import CitationCorrectness
from rag_score.metrics.generation.no_answer_abstention import NoAnswerAbstention
from rag_score.metrics.retrieval.context_coverage import ContextCoverage
from rag_score.metrics.retrieval.hit_rate_at_k import HitRateAtK
from rag_score.metrics.retrieval.mrr import MRR
from rag_score.metrics.retrieval.ndcg import NDCG
from rag_score.metrics.retrieval.precision_at_k import PrecisionAtK
from rag_score.metrics.retrieval.recall_at_k import RecallAtK


class TestRetrievalEdgeCases:
    """Edge case tests for retrieval metrics (OLD interface: score)."""

    @pytest.mark.asyncio
    async def test_empty_retrieval_precision(self):
        """Precision with empty retrieval returns 0."""
        tc = TestCase(question="test", expected_doc_ids=["doc1"])
        result = EvalResult(
            run_id="run1",
            test_case_id=tc.test_case_id,
            retrieved_context=[],
            generated_answer="",
        )

        metric = PrecisionAtK(k=5)
        score = await metric.score(tc, result)
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_empty_retrieval_recall(self):
        """Recall with empty retrieval returns 0 when expected docs exist."""
        tc = TestCase(question="test", expected_doc_ids=["doc1"])
        result = EvalResult(
            run_id="run1",
            test_case_id=tc.test_case_id,
            retrieved_context=[],
            generated_answer="",
        )

        metric = RecallAtK(k=5)
        score = await metric.score(tc, result)
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_empty_retrieval_mrr(self):
        """MRR with empty retrieval returns 0."""
        tc = TestCase(question="test", expected_doc_ids=["doc1"])
        result = EvalResult(
            run_id="run1",
            test_case_id=tc.test_case_id,
            retrieved_context=[],
            generated_answer="",
        )

        metric = MRR()
        score = await metric.score(tc, result)
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_empty_retrieval_ndcg(self):
        """nDCG with empty retrieval returns 0."""
        tc = TestCase(question="test", expected_doc_ids=["doc1"])
        result = EvalResult(
            run_id="run1",
            test_case_id=tc.test_case_id,
            retrieved_context=[],
            generated_answer="",
        )

        metric = NDCG(k=5)
        score = await metric.score(tc, result)
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_no_ground_truth_precision(self):
        """Precision with no expected docs returns 0 (undefined)."""
        tc = TestCase(question="test", expected_doc_ids=[])
        result = EvalResult(
            run_id="run1",
            test_case_id=tc.test_case_id,
            retrieved_context=[RetrievedChunk(doc_id="doc1", text="chunk")],
            generated_answer="",
        )

        metric = PrecisionAtK(k=5)
        score = await metric.score(tc, result)
        assert score == 0.0  # undefined -> 0.0

    @pytest.mark.asyncio
    async def test_no_ground_truth_recall(self):
        """Recall with no expected docs returns 0 (undefined)."""
        tc = TestCase(question="test", expected_doc_ids=[])
        result = EvalResult(
            run_id="run1",
            test_case_id=tc.test_case_id,
            retrieved_context=[RetrievedChunk(doc_id="doc1", text="chunk")],
            generated_answer="",
        )

        metric = RecallAtK(k=5)
        score = await metric.score(tc, result)
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_duplicate_doc_ids_in_retrieved(self):
        """Duplicate doc_ids in retrieved are handled (uses len(top_k) as denominator)."""
        tc = TestCase(question="test", expected_doc_ids=["doc1"])
        result = EvalResult(
            run_id="run1",
            test_case_id=tc.test_case_id,
            retrieved_context=[
                RetrievedChunk(doc_id="doc1", text="chunk 1"),
                RetrievedChunk(doc_id="doc1", text="chunk 1 duplicate"),
                RetrievedChunk(doc_id="doc2", text="chunk 2"),
            ],
            generated_answer="",
        )

        metric = PrecisionAtK(k=5)
        score = await metric.score(tc, result)
        # Precision = relevant_count / len(top_k) = 2/3 (both doc1 count as relevant)
        assert score == pytest.approx(2.0 / 3.0)

    @pytest.mark.asyncio
    async def test_k_larger_than_retrieved(self):
        """k larger than number of retrieved docs works correctly."""
        tc = TestCase(question="test", expected_doc_ids=["doc1", "doc2"])
        result = EvalResult(
            run_id="run1",
            test_case_id=tc.test_case_id,
            retrieved_context=[RetrievedChunk(doc_id="doc1", text="chunk")],
            generated_answer="",
        )

        metric = PrecisionAtK(k=10)  # k > retrieved
        score = await metric.score(tc, result)
        assert score == 1.0  # 1 relevant / 1 retrieved

    @pytest.mark.asyncio
    async def test_k_zero_raises_error(self):
        """k=0 raises ValueError."""
        with pytest.raises(ValueError, match="k must be a positive integer"):
            PrecisionAtK(k=0)

    @pytest.mark.asyncio
    async def test_retrieved_chunks_without_doc_ids(self):
        """Retrieved chunks without doc_ids are handled."""
        tc = TestCase(question="test", expected_doc_ids=["doc1"])
        result = EvalResult(
            run_id="run1",
            test_case_id=tc.test_case_id,
            retrieved_context=[
                RetrievedChunk(doc_id=None, text="chunk 1"),
                RetrievedChunk(doc_id="doc1", text="chunk 2"),
            ],
            generated_answer="",
        )

        metric = PrecisionAtK(k=5)
        score = await metric.score(tc, result)
        # Only chunks with doc_ids can match expected_doc_ids
        # top_k = 2 chunks, but only 1 has doc_id, so relevant_count = 1, len(top_k) = 2
        assert score == pytest.approx(0.5)

    def test_hit_rate_at_k_edge_cases(self):
        """HitRateAtK handles edge cases (NEW interface: evaluate)."""
        # No expected docs
        case = EvaluationCase(
            id="tc",
            question="q",
            contexts=["a"],
            metadata={"expected_doc_ids": [], "retrieved_doc_ids": ["doc1"]},
        )
        metric = HitRateAtK(k=5)
        result = metric.evaluate(case)
        assert result.score_value == 0.0

        # Empty retrieval with expected docs
        case2 = EvaluationCase(
            id="tc2",
            question="q",
            contexts=[],
            metadata={"expected_doc_ids": ["doc1"], "retrieved_doc_ids": []},
        )
        result2 = metric.evaluate(case2)
        assert result2.score_value == 0.0

        # No retrieved doc_ids
        case3 = EvaluationCase(
            id="tc3",
            question="q",
            contexts=["x"],
            metadata={"expected_doc_ids": ["doc1"], "retrieved_doc_ids": [None]},
        )
        result3 = metric.evaluate(case3)
        assert result3.score_value == 0.0

    def test_context_coverage_edge_cases(self):
        """ContextCoverage handles edge cases (NEW interface: evaluate)."""
        # No reference answer
        case = EvaluationCase(
            id="tc",
            question="q",
            reference="",
            contexts=["some context"],
        )
        metric = ContextCoverage()
        result = metric.evaluate(case)
        assert result.score_value == 0.0

        # No context
        case2 = EvaluationCase(
            id="tc2",
            question="q",
            reference="expected answer",
            contexts=[],
        )
        result2 = metric.evaluate(case2)
        assert result2.score_value == 0.0

        # Reference with no key facts (after extraction)
        case3 = EvaluationCase(
            id="tc3",
            question="q",
            reference="!!!",  # No alphanumeric words
            contexts=["some context"],
        )
        result3 = metric.evaluate(case3)
        assert result3.score_value == 0.0


class TestGenerationEdgeCases:
    """Edge case tests for generation metrics (NEW interface: evaluate)."""

    def test_no_answer_abstention_empty_answer(self):
        """NoAnswerAbstention returns 1 for empty answer in no_answer case."""
        case = EvaluationCase(
            id="tc",
            question="q",
            answer="",
            contexts=["context"],
            metadata={"no_answer": True},
        )
        metric = NoAnswerAbstention()
        result = metric.evaluate(case)
        assert result.score_value == 1.0

    def test_no_answer_abstention_no_context(self):
        """NoAnswerAbstention returns 1 when answer contains abstention phrase, regardless of flag.

        Note: Current implementation checks for abstention phrases even without the no_answer flag.
        """
        case = EvaluationCase(
            id="tc",
            question="q",
            answer="I don't know",
            contexts=["context"],
            metadata={},  # no no_answer flag
        )
        metric = NoAnswerAbstention()
        result = metric.evaluate(case)
        # Current behavior: returns 1.0 when answer contains abstention phrase
        assert result.score_value == 1.0

    def test_citation_completeness_no_citations(self):
        """CitationCompleteness handles no citations."""
        case = EvaluationCase(
            id="tc",
            question="q",
            answer="Answer without citations",
            contexts=["context1", "context2"],
        )
        metric = CitationCompleteness()
        result = metric.evaluate(case)
        assert result.score_value == 0.0

    def test_citation_completeness_empty_answer(self):
        """CitationCompleteness handles empty answer."""
        case = EvaluationCase(
            id="tc",
            question="q",
            answer="",
            contexts=["context1", "context2"],
        )
        metric = CitationCompleteness()
        result = metric.evaluate(case)
        assert result.score_value == 0.0

    def test_citation_correctness_no_citations(self):
        """CitationCorrectness handles no citations (vacuously true -> 1.0)."""
        case = EvaluationCase(
            id="tc",
            question="q",
            answer="Answer without citations",
            contexts=["context1", "context2"],
        )
        metric = CitationCorrectness()
        result = metric.evaluate(case)
        assert result.score_value == 1.0

    def test_citation_correctness_empty_answer(self):
        """CitationCorrectness handles empty answer.

        Note: Current implementation returns 1.0 for empty answer (no citations = vacuously correct).
        """
        case = EvaluationCase(
            id="tc",
            question="q",
            answer="",
            contexts=["context1", "context2"],
        )
        metric = CitationCorrectness()
        result = metric.evaluate(case)
        # Current behavior: empty answer has no citations -> vacuously correct = 1.0
        assert result.score_value == 1.0


class TestUnicodeAndLongTextEdgeCases:
    """Edge case tests for unicode and long text."""

    @pytest.mark.asyncio
    async def test_unicode_in_query_and_context(self):
        """Unicode characters in query and context work correctly."""
        tc = TestCase(
            question="测试问题 🎉",
            expected_doc_ids=["doc1", "doc2"],  # Both docs are expected
            ground_truth_answer="测试答案 🎊"
        )
        result = EvalResult(
            run_id="run1",
            test_case_id=tc.test_case_id,
            retrieved_context=[
                RetrievedChunk(doc_id="doc1", text="这是中文文本 🎉🎊"),
                RetrievedChunk(doc_id="doc2", text="English text with emoji 🚀"),
            ],
            generated_answer="Generated answer with unicode 🌟",
        )

        metric = PrecisionAtK(k=5)
        score = await metric.score(tc, result)
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_very_long_text(self):
        """Very long text in context and answer works."""
        long_text = "word " * 10000  # ~50KB text
        tc = TestCase(question="test", expected_doc_ids=["doc1"])
        result = EvalResult(
            run_id="run1",
            test_case_id=tc.test_case_id,
            retrieved_context=[RetrievedChunk(doc_id="doc1", text=long_text)],
            generated_answer=long_text,
        )

        metric = PrecisionAtK(k=5)
        score = await metric.score(tc, result)
        assert score == 1.0

    def test_context_coverage_unicode(self):
        """ContextCoverage handles unicode."""
        case = EvaluationCase(
            id="tc",
            question="q",
            reference="测试答案",
            contexts=["这是相关的上下文 测试"],
        )
        metric = ContextCoverage()
        result = metric.evaluate(case)
        # Should not crash, score may vary
        assert 0.0 <= result.score_value <= 1.0

    def test_context_coverage_long_text(self):
        """ContextCoverage handles long text."""
        long_reference = "word " * 5000
        long_context = "word " * 5000
        case = EvaluationCase(
            id="tc",
            question="q",
            reference=long_reference,
            contexts=[long_context],
        )
        metric = ContextCoverage()
        result = metric.evaluate(case)
        assert 0.0 <= result.score_value <= 1.0


class TestAgenticEdgeCases:
    """Edge case tests for agentic metrics."""

    @pytest.mark.asyncio
    async def test_tool_selection_recall_empty_expected(self):
        """ToolSelectionRecall with empty expected tools returns 0."""
        tc = TrajectoryTestCase(question="test", expected_tool_sequence=[])
        result = TrajectoryEvalResult(
            run_id="run1",
            test_case_id=tc.test_case_id,
            tool_calls=[ToolCall(tool_name="search", tool_input={}, tool_output="result")],
            final_answer="done",
        )

        metric = ToolSelectionRecall()
        score = await metric.score(tc, result)
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_tool_selection_precision_empty_calls(self):
        """ToolSelectionPrecision with no tool calls returns 0."""
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
    async def test_tool_call_order_empty_expected(self):
        """ToolCallOrderCorrectness with empty expected returns 0."""
        tc = TrajectoryTestCase(question="test", expected_tool_sequence=[])
        result = TrajectoryEvalResult(
            run_id="run1",
            test_case_id=tc.test_case_id,
            tool_calls=[ToolCall(tool_name="search", tool_input={}, tool_output="result")],
            final_answer="done",
        )

        metric = ToolCallOrderCorrectness()
        score = await metric.score(tc, result)
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_tool_call_order_empty_actual(self):
        """ToolCallOrderCorrectness with empty actual calls returns 0."""
        tc = TrajectoryTestCase(question="test", expected_tool_sequence=["search", "read"])
        result = TrajectoryEvalResult(
            run_id="run1",
            test_case_id=tc.test_case_id,
            tool_calls=[],
            final_answer="done",
        )

        metric = ToolCallOrderCorrectness()
        score = await metric.score(tc, result)
        assert score == 0.0


class TestZeroTestCases:
    """Edge case tests for zero test cases."""

    @pytest.mark.asyncio
    async def test_empty_dataset_handling(self):
        """Empty dataset handling works."""
        from rag_score.adapters.base import CallableGeneratorAdapter, CallableRetrieverAdapter
        from rag_score.core.runner import RunConfig, run_evaluation
        from rag_score.metrics.retrieval.precision_at_k import PrecisionAtK

        async def dummy_retrieve(query: str, top_k: int = 5):
            return []

        async def dummy_generate(query: str, context):
            return "answer"

        retriever = CallableRetrieverAdapter(dummy_retrieve)
        generator = CallableGeneratorAdapter(dummy_generate)

        config = RunConfig(
            run_id="empty_test",
            project_name="test",
            max_concurrency=1,
            continue_on_error=True,
        )

        # Empty test cases list
        report = await run_evaluation([], retriever, generator, [PrecisionAtK(k=5)], config)

        assert report.run_id == "empty_test"
        assert len(report.results) == 0
        assert len(report.scores) == 0


class TestMalformedJudgeResponse:
    """Edge case tests for malformed judge responses."""

    @pytest.mark.asyncio
    async def test_judge_returns_non_json(self):
        """Judge returning non-JSON is handled."""
        # This is tested in test_regression_judge.py - just ensure it doesn't crash

    @pytest.mark.asyncio
    async def test_judge_returns_missing_score(self):
        """Judge returning JSON without score key is handled."""
        # This is tested in test_regression_judge.py

    @pytest.mark.asyncio
    async def test_judge_returns_score_out_of_range(self):
        """Judge returning score > 1 is clamped."""
        # This is tested in test_regression_judge.py


class TestSpecialCharacters:
    """Edge case tests for special characters."""

    @pytest.mark.asyncio
    async def test_newlines_in_text(self):
        """Newlines in context and answer are handled."""
        tc = TestCase(question="test\nwith\nnewlines", expected_doc_ids=["doc1"])
        result = EvalResult(
            run_id="run1",
            test_case_id=tc.test_case_id,
            retrieved_context=[
                RetrievedChunk(doc_id="doc1", text="line1\nline2\nline3")
            ],
            generated_answer="answer\nwith\nnewlines",
        )

        metric = PrecisionAtK(k=5)
        score = await metric.score(tc, result)
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_tabs_and_special_chars(self):
        """Tabs and special characters are handled."""
        tc = TestCase(question="test\twith\ttabs", expected_doc_ids=["doc1"])
        result = EvalResult(
            run_id="run1",
            test_case_id=tc.test_case_id,
            retrieved_context=[
                RetrievedChunk(doc_id="doc1", text="col1\tcol2\tcol3")
            ],
            generated_answer="answer\twith\ttabs",
        )

        metric = PrecisionAtK(k=5)
        score = await metric.score(tc, result)
        assert score == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])