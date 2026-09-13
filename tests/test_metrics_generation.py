"""Tests for the LLM-judge generation metrics. All use FakeJudge - zero
network access, zero API keys - since the metrics' own logic (prompt
formatting, edge-case handling) is what's under test, not any real
model's judgment quality."""

from __future__ import annotations

from rag_score.core.types import EvalResult, RetrievedChunk
from rag_score.metrics.generation.answer_relevance import AnswerRelevance
from rag_score.metrics.generation.context_precision import ContextPrecision
from rag_score.metrics.generation.faithfulness import Faithfulness


class TestFaithfulness:
    async def test_returns_judge_score(self, sample_test_cases, sample_eval_result, fake_judge):
        metric = Faithfulness(judge=fake_judge)
        score = await metric.score(sample_test_cases[0], sample_eval_result)
        assert score == 0.8

    async def test_score_with_reasoning_returns_both(
        self, sample_test_cases, sample_eval_result, fake_judge
    ):
        metric = Faithfulness(judge=fake_judge)
        score, reasoning = await metric.score_with_reasoning(sample_test_cases[0], sample_eval_result)
        assert score == 0.8
        assert reasoning == "Looks reasonable."

    async def test_no_answer_skips_judge_call(self, sample_test_cases, fake_judge):
        result = EvalResult(
            run_id="r", test_case_id="tc",
            retrieved_context=[RetrievedChunk(doc_id="d", text="x")],
            generated_answer=None,
        )
        metric = Faithfulness(judge=fake_judge)
        score = await metric.score(sample_test_cases[0], result)
        assert score == 0.0
        assert fake_judge.call_count == 0  # shouldn't waste an API call on empty input

    async def test_no_context_skips_judge_call(self, sample_test_cases, fake_judge):
        result = EvalResult(
            run_id="r", test_case_id="tc",
            retrieved_context=[],
            generated_answer="some answer",
        )
        metric = Faithfulness(judge=fake_judge)
        score = await metric.score(sample_test_cases[0], result)
        assert score == 0.0
        assert fake_judge.call_count == 0

    def test_requires_api_key_flag(self, fake_judge):
        assert Faithfulness(judge=fake_judge).requires_api_key is True


class TestAnswerRelevance:
    async def test_returns_judge_score(self, sample_test_cases, sample_eval_result, fake_judge):
        metric = AnswerRelevance(judge=fake_judge)
        score = await metric.score(sample_test_cases[0], sample_eval_result)
        assert score == 0.8

    async def test_no_answer_skips_judge_call(self, sample_test_cases, fake_judge):
        result = EvalResult(run_id="r", test_case_id="tc", generated_answer=None)
        metric = AnswerRelevance(judge=fake_judge)
        score = await metric.score(sample_test_cases[0], result)
        assert score == 0.0
        assert fake_judge.call_count == 0

    async def test_does_not_require_context(self, sample_test_cases, fake_judge):
        # Unlike Faithfulness, relevance only needs question + answer
        result = EvalResult(
            run_id="r", test_case_id="tc", retrieved_context=[], generated_answer="an answer"
        )
        metric = AnswerRelevance(judge=fake_judge)
        score = await metric.score(sample_test_cases[0], result)
        assert score == 0.8
        assert fake_judge.call_count == 1


class TestContextPrecision:
    async def test_returns_judge_score(self, sample_test_cases, sample_eval_result, fake_judge):
        metric = ContextPrecision(judge=fake_judge)
        score = await metric.score(sample_test_cases[0], sample_eval_result)
        assert score == 0.8

    async def test_no_context_skips_judge_call(self, sample_test_cases, fake_judge):
        result = EvalResult(run_id="r", test_case_id="tc", retrieved_context=[])
        metric = ContextPrecision(judge=fake_judge)
        score = await metric.score(sample_test_cases[0], result)
        assert score == 0.0
        assert fake_judge.call_count == 0

    async def test_does_not_require_generated_answer(self, sample_test_cases, fake_judge):
        # Context precision is purely about retrieval, not generation
        result = EvalResult(
            run_id="r", test_case_id="tc",
            retrieved_context=[RetrievedChunk(doc_id="d", text="x")],
            generated_answer=None,
        )
        metric = ContextPrecision(judge=fake_judge)
        score = await metric.score(sample_test_cases[0], result)
        assert score == 0.8
