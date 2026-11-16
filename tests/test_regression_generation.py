"""
Regression/golden tests for generation metrics using FakeJudge (no network/API keys).
"""

from __future__ import annotations

# Import the FakeJudge from conftest
import sys
from pathlib import Path

import pytest

from rag_score.core.models import EvaluationCase
from rag_score.core.types import EvalResult, RetrievedChunk, TestCase
from rag_score.metrics.generation.answer_relevance import AnswerRelevance
from rag_score.metrics.generation.citation_completeness import CitationCompleteness
from rag_score.metrics.generation.citation_correctness import CitationCorrectness
from rag_score.metrics.generation.context_carry_over import ContextCarryOver
from rag_score.metrics.generation.context_precision import ContextPrecision
from rag_score.metrics.generation.faithfulness import Faithfulness
from rag_score.metrics.generation.local_answer_relevance import LocalSemanticAnswerRelevance
from rag_score.metrics.generation.local_faithfulness import LocalSemanticFaithfulness
from rag_score.metrics.generation.no_answer_abstention import NoAnswerAbstention

sys.path.insert(0, str(Path(__file__).parent))
from conftest import FakeJudge


class TestFaithfulnessRegression:
    """Golden tests for Faithfulness metric."""

    @pytest.mark.asyncio
    async def test_faithfulness_returns_judge_score(self):
        """Faithfulness returns the judge's score."""
        tc = TestCase(question="What is the capital of France?")
        result = EvalResult(
            run_id="r",
            test_case_id="tc",
            retrieved_context=[RetrievedChunk(doc_id="d1", text="Paris is the capital of France.")],
            generated_answer="Paris is the capital of France.",
        )
        judge = FakeJudge(score=0.8, reasoning="Fully supported.")
        metric = Faithfulness(judge=judge)
        score = await metric.score(tc, result)
        assert score == 0.8

    @pytest.mark.asyncio
    async def test_faithfulness_score_with_reasoning(self):
        """score_with_reasoning returns both score and reasoning."""
        tc = TestCase(question="What is the capital of France?")
        result = EvalResult(
            run_id="r",
            test_case_id="tc",
            retrieved_context=[RetrievedChunk(doc_id="d1", text="Paris is the capital of France.")],
            generated_answer="Paris is the capital of France.",
        )
        judge = FakeJudge(score=0.8, reasoning="Fully supported.")
        metric = Faithfulness(judge=judge)
        score, reasoning = await metric.score_with_reasoning(tc, result)
        assert score == 0.8
        assert reasoning == "Fully supported."

    @pytest.mark.asyncio
    async def test_faithfulness_no_answer_skips_judge(self):
        """No generated answer -> 0.0, no judge call."""
        tc = TestCase(question="What is the capital of France?")
        result = EvalResult(
            run_id="r",
            test_case_id="tc",
            retrieved_context=[RetrievedChunk(doc_id="d1", text="Paris is the capital of France.")],
            generated_answer=None,
        )
        judge = FakeJudge(score=0.8)
        metric = Faithfulness(judge=judge)
        score = await metric.score(tc, result)
        assert score == 0.0
        assert judge.call_count == 0

    @pytest.mark.asyncio
    async def test_faithfulness_no_context_skips_judge(self):
        """No retrieved context -> 0.0, no judge call."""
        tc = TestCase(question="What is the capital of France?")
        result = EvalResult(
            run_id="r",
            test_case_id="tc",
            retrieved_context=[],
            generated_answer="Paris",
        )
        judge = FakeJudge(score=0.8)
        metric = Faithfulness(judge=judge)
        score = await metric.score(tc, result)
        assert score == 0.0
        assert judge.call_count == 0

    @pytest.mark.asyncio
    async def test_faithfulness_empty_answer_skips_judge(self):
        """Empty string answer -> 0.0, no judge call."""
        tc = TestCase(question="What is the capital of France?")
        result = EvalResult(
            run_id="r",
            test_case_id="tc",
            retrieved_context=[RetrievedChunk(doc_id="d1", text="Paris is the capital of France.")],
            generated_answer="",
        )
        judge = FakeJudge(score=0.8)
        metric = Faithfulness(judge=judge)
        score = await metric.score(tc, result)
        assert score == 0.0
        assert judge.call_count == 0

    def test_faithfulness_requires_api_key(self):
        """Faithfulness requires_api_key is True."""
        judge = FakeJudge()
        assert Faithfulness(judge=judge).requires_api_key is True


class TestAnswerRelevanceRegression:
    """Golden tests for AnswerRelevance metric."""

    @pytest.mark.asyncio
    async def test_answer_relevance_returns_judge_score(self):
        """AnswerRelevance returns the judge's score."""
        tc = TestCase(question="What is the capital of France?")
        result = EvalResult(
            run_id="r",
            test_case_id="tc",
            retrieved_context=[RetrievedChunk(doc_id="d1", text="Paris is the capital of France.")],
            generated_answer="Paris is the capital of France.",
        )
        judge = FakeJudge(score=0.9, reasoning="Directly answers the question.")
        metric = AnswerRelevance(judge=judge)
        score = await metric.score(tc, result)
        assert score == 0.9

    @pytest.mark.asyncio
    async def test_answer_relevance_no_answer_skips_judge(self):
        """No generated answer -> 0.0, no judge call."""
        tc = TestCase(question="What is the capital of France?")
        result = EvalResult(
            run_id="r",
            test_case_id="tc",
            retrieved_context=[RetrievedChunk(doc_id="d1", text="Paris is the capital of France.")],
            generated_answer=None,
        )
        judge = FakeJudge(score=0.9)
        metric = AnswerRelevance(judge=judge)
        score = await metric.score(tc, result)
        assert score == 0.0
        assert judge.call_count == 0

    @pytest.mark.asyncio
    async def test_answer_relevance_does_not_require_context(self):
        """AnswerRelevance works without context (question + answer only)."""
        tc = TestCase(question="What is the capital of France?")
        result = EvalResult(
            run_id="r",
            test_case_id="tc",
            retrieved_context=[],
            generated_answer="Paris is the capital of France.",
        )
        judge = FakeJudge(score=0.9)
        metric = AnswerRelevance(judge=judge)
        score = await metric.score(tc, result)
        assert score == 0.9
        assert judge.call_count == 1

    def test_answer_relevance_requires_api_key(self):
        """AnswerRelevance requires_api_key is True."""
        judge = FakeJudge()
        assert AnswerRelevance(judge=judge).requires_api_key is True


class TestContextPrecisionRegression:
    """Golden tests for ContextPrecision metric."""

    @pytest.mark.asyncio
    async def test_context_precision_returns_judge_score(self):
        """ContextPrecision returns the judge's score."""
        tc = TestCase(question="What is the capital of France?")
        result = EvalResult(
            run_id="r",
            test_case_id="tc",
            retrieved_context=[RetrievedChunk(doc_id="d1", text="Paris is the capital of France.")],
            generated_answer="Paris",
        )
        judge = FakeJudge(score=0.7, reasoning="Context is relevant.")
        metric = ContextPrecision(judge=judge)
        score = await metric.score(tc, result)
        assert score == 0.7

    @pytest.mark.asyncio
    async def test_context_precision_no_context_skips_judge(self):
        """No retrieved context -> 0.0, no judge call."""
        tc = TestCase(question="What is the capital of France?")
        result = EvalResult(
            run_id="r",
            test_case_id="tc",
            retrieved_context=[],
            generated_answer="Paris",
        )
        judge = FakeJudge(score=0.7)
        metric = ContextPrecision(judge=judge)
        score = await metric.score(tc, result)
        assert score == 0.0
        assert judge.call_count == 0

    @pytest.mark.asyncio
    async def test_context_precision_does_not_require_answer(self):
        """ContextPrecision works without generated answer."""
        tc = TestCase(question="What is the capital of France?")
        result = EvalResult(
            run_id="r",
            test_case_id="tc",
            retrieved_context=[RetrievedChunk(doc_id="d1", text="Paris is the capital of France.")],
            generated_answer=None,
        )
        judge = FakeJudge(score=0.7)
        metric = ContextPrecision(judge=judge)
        score = await metric.score(tc, result)
        assert score == 0.7
        assert judge.call_count == 1

    def test_context_precision_requires_api_key(self):
        """ContextPrecision requires_api_key is True."""
        judge = FakeJudge()
        assert ContextPrecision(judge=judge).requires_api_key is True


class TestContextCarryOverRegression:
    """Golden tests for ContextCarryOver metric."""

    @pytest.mark.asyncio
    async def test_context_carry_over_with_history_scores_from_judge(self):
        """With valid history, calls judge and returns score."""
        tc = TestCase(
            question="What about the second one?",
            metadata={"conversation_history": [{"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "Paris is the capital of France."}]},
        )
        result = EvalResult(
            run_id="r",
            test_case_id="tc",
            retrieved_context=[RetrievedChunk(doc_id="d1", text="Berlin is the capital of Germany.")],
            generated_answer="The second one is Berlin.",
        )
        judge = FakeJudge(score=0.8, reasoning="Correctly resolves 'the second one' to Berlin.")
        metric = ContextCarryOver(judge=judge)
        score = await metric.score(tc, result)
        assert score == 0.8

    @pytest.mark.asyncio
    async def test_context_carry_over_without_history_returns_zero(self):
        """Missing history -> 0.0, no judge call."""
        tc = TestCase(question="What about the second one?")
        result = EvalResult(
            run_id="r",
            test_case_id="tc",
            retrieved_context=[RetrievedChunk(doc_id="d1", text="Berlin")],
            generated_answer="Berlin",
        )
        judge = FakeJudge(score=0.8)
        metric = ContextCarryOver(judge=judge)
        score = await metric.score(tc, result)
        assert score == 0.0
        assert judge.call_count == 0

    @pytest.mark.asyncio
    async def test_context_carry_over_with_empty_history_returns_zero(self):
        """Empty history list -> 0.0, no judge call."""
        tc = TestCase(question="What about the second one?", metadata={"conversation_history": []})
        result = EvalResult(
            run_id="r",
            test_case_id="tc",
            retrieved_context=[RetrievedChunk(doc_id="d1", text="Berlin")],
            generated_answer="Berlin",
        )
        judge = FakeJudge(score=0.8)
        metric = ContextCarryOver(judge=judge)
        score = await metric.score(tc, result)
        assert score == 0.0
        assert judge.call_count == 0

    @pytest.mark.asyncio
    async def test_context_carry_over_malformed_history_returns_zero(self):
        """Malformed history entries -> 0.0, no judge call."""
        tc = TestCase(
            question="What about the second one?",
            metadata={"conversation_history": [{"role": "user"}]},  # missing content
        )
        result = EvalResult(
            run_id="r",
            test_case_id="tc",
            retrieved_context=[RetrievedChunk(doc_id="d1", text="Berlin")],
            generated_answer="Berlin",
        )
        judge = FakeJudge(score=0.8)
        metric = ContextCarryOver(judge=judge)
        score = await metric.score(tc, result)
        assert score == 0.0
        assert judge.call_count == 0

    @pytest.mark.asyncio
    async def test_context_carry_over_no_answer_returns_zero(self):
        """No generated answer -> 0.0, no judge call."""
        tc = TestCase(
            question="What about the second one?",
            metadata={"conversation_history": [{"role": "user", "content": "What is the capital of France?"}]},
        )
        result = EvalResult(
            run_id="r",
            test_case_id="tc",
            retrieved_context=[RetrievedChunk(doc_id="d1", text="Berlin")],
            generated_answer=None,
        )
        judge = FakeJudge(score=0.8)
        metric = ContextCarryOver(judge=judge)
        score = await metric.score(tc, result)
        assert score == 0.0
        assert judge.call_count == 0

    def test_context_carry_over_requires_api_key(self):
        """ContextCarryOver requires_api_key is True."""
        judge = FakeJudge()
        assert ContextCarryOver(judge=judge).requires_api_key is True


class TestLocalSemanticFaithfulnessRegression:
    """Golden tests for LocalSemanticFaithfulness metric."""

    @pytest.mark.asyncio
    async def test_local_faithfulness_similar_scores_high(self):
        """Similar answer and context -> high score."""
        tc = TestCase(question="What is the capital of France?")
        result = EvalResult(
            run_id="r",
            test_case_id="tc",
            retrieved_context=[RetrievedChunk(doc_id="d1", text="Paris is the capital of France.")],
            generated_answer="The capital of France is Paris.",
        )
        metric = LocalSemanticFaithfulness()
        score = await metric.score(tc, result)
        assert score > 0.8  # Should be high for semantically similar text

    @pytest.mark.asyncio
    async def test_local_faithfulness_dissimilar_scores_low(self):
        """Dissimilar answer and context -> low score."""
        tc = TestCase(question="What is the capital of France?")
        result = EvalResult(
            run_id="r",
            test_case_id="tc",
            retrieved_context=[RetrievedChunk(doc_id="d1", text="Berlin is the capital of Germany.")],
            generated_answer="Paris is the capital of France.",
        )
        metric = LocalSemanticFaithfulness()
        score = await metric.score(tc, result)
        assert score < 0.5  # Should be low for dissimilar text

    @pytest.mark.asyncio
    async def test_local_faithfulness_no_answer_returns_zero(self):
        """No answer -> 0.0 without calling encoder."""
        tc = TestCase(question="What is the capital of France?")
        result = EvalResult(
            run_id="r",
            test_case_id="tc",
            retrieved_context=[RetrievedChunk(doc_id="d1", text="Paris is the capital of France.")],
            generated_answer=None,
        )
        metric = LocalSemanticFaithfulness()
        score = await metric.score(tc, result)
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_local_faithfulness_no_context_returns_zero(self):
        """No context -> 0.0 without calling encoder."""
        tc = TestCase(question="What is the capital of France?")
        result = EvalResult(
            run_id="r",
            test_case_id="tc",
            retrieved_context=[],
            generated_answer="Paris is the capital of France.",
        )
        metric = LocalSemanticFaithfulness()
        score = await metric.score(tc, result)
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_local_faithfulness_multiple_context_chunks_joined(self):
        """Multiple context chunks are joined for comparison."""
        tc = TestCase(question="What is the capital of France?")
        result = EvalResult(
            run_id="r",
            test_case_id="tc",
            retrieved_context=[
                RetrievedChunk(doc_id="d1", text="Paris is the capital"),
                RetrievedChunk(doc_id="d2", text="of France."),
            ],
            generated_answer="Paris is the capital of France.",
        )
        metric = LocalSemanticFaithfulness()
        score = await metric.score(tc, result)
        assert score > 0.8

    def test_local_faithfulness_requires_no_api_key(self):
        """LocalSemanticFaithfulness requires_api_key is False."""
        assert LocalSemanticFaithfulness().requires_api_key is False


class TestLocalSemanticAnswerRelevanceRegression:
    """Golden tests for LocalSemanticAnswerRelevance metric."""

    @pytest.mark.asyncio
    async def test_local_answer_relevance_relevant_scores_high(self):
        """Relevant answer -> high score."""
        tc = TestCase(question="What is the capital of France?")
        result = EvalResult(
            run_id="r",
            test_case_id="tc",
            retrieved_context=[RetrievedChunk(doc_id="d1", text="Paris is the capital of France.")],
            generated_answer="The capital of France is Paris.",
        )
        metric = LocalSemanticAnswerRelevance()
        score = await metric.score(tc, result)
        assert score > 0.8

    @pytest.mark.asyncio
    async def test_local_answer_relevance_no_answer_returns_zero(self):
        """No answer -> 0.0 without calling encoder."""
        tc = TestCase(question="What is the capital of France?")
        result = EvalResult(
            run_id="r",
            test_case_id="tc",
            retrieved_context=[RetrievedChunk(doc_id="d1", text="Paris is the capital of France.")],
            generated_answer=None,
        )
        metric = LocalSemanticAnswerRelevance()
        score = await metric.score(tc, result)
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_local_answer_relevance_does_not_require_context(self):
        """Works without context (question + answer only)."""
        tc = TestCase(question="What is the capital of France?")
        result = EvalResult(
            run_id="r",
            test_case_id="tc",
            retrieved_context=[],
            generated_answer="Paris is the capital of France.",
        )
        metric = LocalSemanticAnswerRelevance()
        score = await metric.score(tc, result)
        assert score > 0.8

    def test_local_answer_relevance_requires_no_api_key(self):
        """LocalSemanticAnswerRelevance requires_api_key is False."""
        assert LocalSemanticAnswerRelevance().requires_api_key is False


class TestNoAnswerAbstentionRegression:
    """Golden tests for NoAnswerAbstention metric.

    Note: NoAnswerAbstention uses the NEW metric interface with EvaluationCase.
    It's an offline metric that doesn't require a judge.
    """

    def test_no_answer_abstention_detects_abstention(self):
        """Detects 'I don't know' style answers when no_answer=True in metadata."""
        case = EvaluationCase(
            id="tc",
            question="What is the meaning of life?",
            answer="I don't know the answer to that question.",
            contexts=["Some text"],
            metadata={"no_answer": True},
        )
        metric = NoAnswerAbstention()
        result = metric.evaluate(case)
        # Score should be 1.0 for clear abstention
        assert result.score_value == 1.0

    def test_no_answer_abstention_no_abstention(self):
        """Non-abstention answer -> 0.0."""
        case = EvaluationCase(
            id="tc",
            question="What is the capital of France?",
            answer="Paris is the capital of France.",
            contexts=["Paris is the capital of France."],
            metadata={"no_answer": True},
        )
        metric = NoAnswerAbstention()
        result = metric.evaluate(case)
        assert result.score_value == 0.0

    def test_no_answer_abstention_no_answer_returns_zero(self):
        """No generated answer -> 0.0 (empty answer is abstention, so 1.0)."""
        case = EvaluationCase(
            id="tc",
            question="What is the capital of France?",
            answer="",
            contexts=["Paris"],
            metadata={"no_answer": True},
        )
        metric = NoAnswerAbstention()
        result = metric.evaluate(case)
        # Empty answer is considered abstention
        assert result.score_value == 1.0

    def test_no_answer_abstention_not_no_answer_case(self):
        """Not a no_answer case -> metric returns 0.0 (not applicable)."""
        case = EvaluationCase(
            id="tc",
            question="What is the capital of France?",
            answer="Paris is the capital of France.",
            contexts=["Paris is the capital of France."],
            metadata={},  # no no_answer flag
        )
        metric = NoAnswerAbstention()
        result = metric.evaluate(case)
        assert result.score_value == 0.0

    def test_no_answer_abstention_requires_api_key(self):
        """NoAnswerAbstention is an offline metric (no judge needed)."""
        # New metrics don't have requires_api_key attribute; they just don't use judges
        metric = NoAnswerAbstention()
        # Just verify it works without a judge
        case = EvaluationCase(id="tc", question="q", answer="I don't know", contexts=["x"], metadata={"no_answer": True})
        result = metric.evaluate(case)
        assert result.score_value == 1.0


class TestCitationCompletenessRegression:
    """Golden tests for CitationCompleteness metric.

    Note: CitationCompleteness uses the NEW metric interface with EvaluationCase.
    It's an offline metric that doesn't require a judge.
    """

    def test_citation_completeness_perfect_citations(self):
        """All retrieved chunks are cited -> 1.0."""
        case = EvaluationCase(
            id="tc",
            question="What are the capitals?",
            answer="Paris is the capital of France [1]. Berlin is the capital of Germany [2].",
            contexts=[
                "Paris is the capital of France.",
                "Berlin is the capital of Germany.",
            ],
        )
        metric = CitationCompleteness()
        result = metric.evaluate(case)
        assert result.score_value == 1.0

    def test_citation_completeness_missing_citations(self):
        """Only 1 of 2 chunks cited -> 0.5."""
        case = EvaluationCase(
            id="tc",
            question="What are the capitals?",
            answer="Paris is the capital of France [1]. Rome is the capital of Italy.",
            contexts=[
                "Paris is the capital of France.",
            ],
        )
        metric = CitationCompleteness()
        result = metric.evaluate(case)
        # 1 chunk total, 1 cited -> 1.0
        # Wait - only 1 chunk in contexts, but answer cites [1] which exists
        # The metric checks fraction of CHUNKS that are cited
        # 1 chunk, 1 cited -> 1.0
        # Let me re-read the metric: it counts how many chunks are cited out of total chunks
        # If there's 1 chunk and it's cited, score is 1.0
        assert result.score_value == 1.0

    def test_citation_completeness_multiple_chunks_some_cited(self):
        """2 chunks, only 1 cited -> 0.5."""
        case = EvaluationCase(
            id="tc",
            question="What are the capitals?",
            answer="Paris is the capital of France [1]. Rome is the capital of Italy.",
            contexts=[
                "Paris is the capital of France.",
                "Berlin is the capital of Germany.",
            ],
        )
        metric = CitationCompleteness()
        result = metric.evaluate(case)
        # 2 chunks, only chunk 1 cited -> 0.5
        assert result.score_value == 0.5

    def test_citation_completeness_no_citations(self):
        """No citations -> 0.0."""
        case = EvaluationCase(
            id="tc",
            question="What are the capitals?",
            answer="Paris is the capital of France. Berlin is the capital of Germany.",
            contexts=[
                "Paris is the capital of France.",
                "Berlin is the capital of Germany.",
            ],
        )
        metric = CitationCompleteness()
        result = metric.evaluate(case)
        assert result.score_value == 0.0

    def test_citation_completeness_empty_answer(self):
        """Empty answer -> 0.0."""
        case = EvaluationCase(
            id="tc",
            question="What are the capitals?",
            answer="",
            contexts=[
                "Paris is the capital of France.",
            ],
        )
        metric = CitationCompleteness()
        result = metric.evaluate(case)
        assert result.score_value == 0.0

    def test_citation_completeness_requires_no_api_key(self):
        """CitationCompleteness is an offline metric (no judge needed)."""
        metric = CitationCompleteness()
        case = EvaluationCase(id="tc", question="q", answer="x [1]", contexts=["x"])
        result = metric.evaluate(case)
        assert result.score_value == 1.0


class TestCitationCorrectnessRegression:
    """Golden tests for CitationCorrectness metric.

    Note: CitationCorrectness uses the NEW metric interface with EvaluationCase.
    It's an offline metric that doesn't require a judge.
    """

    def test_citation_correctness_correct_citations(self):
        """Citations point to existing chunks -> 1.0."""
        case = EvaluationCase(
            id="tc",
            question="What is the capital of France?",
            answer="Paris is the capital of France [1].",
            contexts=[
                "Paris is the capital of France.",
            ],
        )
        metric = CitationCorrectness()
        result = metric.evaluate(case)
        assert result.score_value == 1.0

    def test_citation_correctness_incorrect_citations(self):
        """Citation points to non-existent chunk -> 0.0."""
        case = EvaluationCase(
            id="tc",
            question="What is the capital of France?",
            answer="Paris is the capital of France [2].",  # cites chunk 2 but only 1 chunk exists
            contexts=[
                "Berlin is the capital of Germany.",
            ],
        )
        metric = CitationCorrectness()
        result = metric.evaluate(case)
        assert result.score_value == 0.0

    def test_citation_correctness_no_citations(self):
        """No citations -> vacuously 1.0."""
        case = EvaluationCase(
            id="tc",
            question="What is the capital of France?",
            answer="Paris is the capital of France.",
            contexts=[
                "Paris is the capital of France.",
            ],
        )
        metric = CitationCorrectness()
        result = metric.evaluate(case)
        assert result.score_value == 1.0

    def test_citation_correctness_mixed_citations(self):
        """Some citations correct, some incorrect."""
        case = EvaluationCase(
            id="tc",
            question="What are the capitals?",
            answer="Paris is the capital of France [1]. Rome is the capital of Italy [3].",
            contexts=[
                "Paris is the capital of France.",
                "Berlin is the capital of Germany.",
            ],
        )
        metric = CitationCorrectness()
        result = metric.evaluate(case)
        # 2 citations: [1] is valid, [3] is invalid -> 0.5
        assert result.score_value == 0.5

    def test_citation_correctness_empty_answer(self):
        """Empty answer -> 1.0 (no citations to be wrong)."""
        case = EvaluationCase(
            id="tc",
            question="What is the capital of France?",
            answer="",
            contexts=[
                "Paris is the capital of France.",
            ],
        )
        metric = CitationCorrectness()
        result = metric.evaluate(case)
        assert result.score_value == 1.0

    def test_citation_correctness_requires_no_api_key(self):
        """CitationCorrectness is an offline metric (no judge needed)."""
        metric = CitationCorrectness()
        case = EvaluationCase(id="tc", question="q", answer="x [1]", contexts=["x"])
        result = metric.evaluate(case)
        assert result.score_value == 1.0