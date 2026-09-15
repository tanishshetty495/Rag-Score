"""Tests for LocalSemanticFaithfulness and LocalSemanticAnswerRelevance.
Uses an injected FakeEncoder throughout - zero network access, zero
model download, consistent with the rest of the suite's offline-first
testing philosophy."""

from __future__ import annotations

from rag_score.core.types import EvalResult, RetrievedChunk, TestCase
from rag_score.metrics.generation.local_answer_relevance import (
    LocalSemanticAnswerRelevance,
)
from rag_score.metrics.generation.local_faithfulness import LocalSemanticFaithfulness


class _FakeEncoder:
    """Deterministic fake encoder - maps known strings to fixed
    vectors so similarity scores are predictable in tests."""

    def __init__(self, vectors: dict[str, list[float]] | None = None):
        self.call_count = 0
        self.vectors = vectors or {}

    def encode(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        return [self.vectors.get(t, [0.5, 0.5, 0.0]) for t in texts]


class TestLocalSemanticFaithfulness:
    async def test_similar_answer_and_context_scores_high(self):
        encoder = _FakeEncoder({
            "the answer": [1.0, 0.0, 0.0],
            "the context": [1.0, 0.0, 0.0],
        })
        metric = LocalSemanticFaithfulness(encoder=encoder)
        tc = TestCase(question="q")
        result = EvalResult(
            run_id="r", test_case_id="tc",
            retrieved_context=[RetrievedChunk(doc_id="d", text="the context")],
            generated_answer="the answer",
        )
        score = await metric.score(tc, result)
        assert score == 1.0

    async def test_dissimilar_answer_and_context_scores_low(self):
        encoder = _FakeEncoder({
            "the answer": [1.0, 0.0, 0.0],
            "unrelated context": [0.0, 1.0, 0.0],
        })
        metric = LocalSemanticFaithfulness(encoder=encoder)
        tc = TestCase(question="q")
        result = EvalResult(
            run_id="r", test_case_id="tc",
            retrieved_context=[RetrievedChunk(doc_id="d", text="unrelated context")],
            generated_answer="the answer",
        )
        score = await metric.score(tc, result)
        assert score == 0.0

    async def test_no_answer_returns_zero_without_calling_encoder(self):
        encoder = _FakeEncoder()
        metric = LocalSemanticFaithfulness(encoder=encoder)
        tc = TestCase(question="q")
        result = EvalResult(
            run_id="r", test_case_id="tc",
            retrieved_context=[RetrievedChunk(doc_id="d", text="x")],
            generated_answer=None,
        )
        score = await metric.score(tc, result)
        assert score == 0.0
        assert encoder.call_count == 0

    async def test_no_context_returns_zero_without_calling_encoder(self):
        encoder = _FakeEncoder()
        metric = LocalSemanticFaithfulness(encoder=encoder)
        tc = TestCase(question="q")
        result = EvalResult(run_id="r", test_case_id="tc", retrieved_context=[], generated_answer="x")
        score = await metric.score(tc, result)
        assert score == 0.0
        assert encoder.call_count == 0

    async def test_multiple_context_chunks_are_joined(self):
        encoder = _FakeEncoder()
        metric = LocalSemanticFaithfulness(encoder=encoder)
        tc = TestCase(question="q")
        result = EvalResult(
            run_id="r", test_case_id="tc",
            retrieved_context=[
                RetrievedChunk(doc_id="d1", text="first chunk"),
                RetrievedChunk(doc_id="d2", text="second chunk"),
            ],
            generated_answer="an answer",
        )
        await metric.score(tc, result)
        # encode() should have been called with the answer plus the
        # joined context as a single string, not called once per chunk
        assert encoder.call_count == 1

    def test_requires_api_key_is_false(self):
        assert LocalSemanticFaithfulness().requires_api_key is False

    def test_name(self):
        assert LocalSemanticFaithfulness().name == "local_faithfulness"


class TestLocalSemanticAnswerRelevance:
    async def test_relevant_answer_scores_high(self):
        encoder = _FakeEncoder({
            "what is x": [1.0, 0.0, 0.0],
            "x is y": [1.0, 0.0, 0.0],
        })
        metric = LocalSemanticAnswerRelevance(encoder=encoder)
        tc = TestCase(question="what is x")
        result = EvalResult(run_id="r", test_case_id="tc", generated_answer="x is y")
        score = await metric.score(tc, result)
        assert score == 1.0

    async def test_no_answer_returns_zero_without_calling_encoder(self):
        encoder = _FakeEncoder()
        metric = LocalSemanticAnswerRelevance(encoder=encoder)
        tc = TestCase(question="q")
        result = EvalResult(run_id="r", test_case_id="tc", generated_answer=None)
        score = await metric.score(tc, result)
        assert score == 0.0
        assert encoder.call_count == 0

    async def test_does_not_require_context(self):
        encoder = _FakeEncoder()
        metric = LocalSemanticAnswerRelevance(encoder=encoder)
        tc = TestCase(question="q")
        result = EvalResult(run_id="r", test_case_id="tc", retrieved_context=[], generated_answer="an answer")
        await metric.score(tc, result)
        assert encoder.call_count == 1  # ran despite no context

    def test_name(self):
        assert LocalSemanticAnswerRelevance().name == "local_answer_relevance"
