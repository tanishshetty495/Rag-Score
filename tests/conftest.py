"""
Shared fixtures. Deliberately builds everything from mock/callable
adapters rather than real network calls - the whole point of
offline-first metrics is that the test suite itself needs zero API
keys and zero network access to run in CI.
"""

from __future__ import annotations

import pytest

from rag_score.adapters.base import CallableGeneratorAdapter, CallableRetrieverAdapter
from rag_score.core.types import EvalResult, RetrievedChunk, TestCase
from rag_score.judges.base import LLMJudge


@pytest.fixture
def sample_test_cases() -> list[TestCase]:
    return [
        TestCase(
            question="What is the refund policy?",
            expected_doc_ids=["doc_1", "doc_2"],
            dataset_name="demo",
        ),
        TestCase(
            question="How do I reset my password?",
            expected_doc_ids=["doc_7"],
            dataset_name="demo",
        ),
    ]


@pytest.fixture
def sample_eval_result() -> EvalResult:
    """A retrieval with one relevant hit at rank 2 of 3 - a good
    general-purpose fixture for testing partial-credit metrics."""
    return EvalResult(
        run_id="test-run",
        test_case_id="tc-1",
        retrieved_context=[
            RetrievedChunk(doc_id="doc_9", text="irrelevant"),
            RetrievedChunk(doc_id="doc_1", text="relevant chunk"),
            RetrievedChunk(doc_id="doc_5", text="irrelevant"),
        ],
        generated_answer="Refunds within 30 days.",
    )


@pytest.fixture
def perfect_eval_result() -> EvalResult:
    """All relevant docs at the top of the ranking - should score 1.0
    on every metric."""
    return EvalResult(
        run_id="test-run",
        test_case_id="tc-1",
        retrieved_context=[
            RetrievedChunk(doc_id="doc_1", text="a"),
            RetrievedChunk(doc_id="doc_2", text="b"),
        ],
        generated_answer="Refunds within 30 days.",
    )


@pytest.fixture
def empty_eval_result() -> EvalResult:
    """Retriever returned nothing - metrics should degrade to 0.0, not crash."""
    return EvalResult(run_id="test-run", test_case_id="tc-1", retrieved_context=[])


async def _fake_retrieve(query: str, top_k: int = 5) -> list[RetrievedChunk]:
    if "refund" in query.lower():
        return [
            RetrievedChunk(doc_id="doc_9", text="irrelevant"),
            RetrievedChunk(doc_id="doc_1", text="Refunds within 30 days."),
            RetrievedChunk(doc_id="doc_2", text="Refund process details."),
        ][:top_k]
    return [RetrievedChunk(doc_id="doc_7", text="Click forgot password.")][:top_k]


async def _fake_generate(query: str, context: list[RetrievedChunk]) -> str:
    return f"Answer using {len(context)} retrieved chunks."


@pytest.fixture
def fake_retriever() -> CallableRetrieverAdapter:
    return CallableRetrieverAdapter(_fake_retrieve)


@pytest.fixture
def fake_generator() -> CallableGeneratorAdapter:
    return CallableGeneratorAdapter(_fake_generate)


@pytest.fixture
def failing_retriever() -> CallableRetrieverAdapter:
    async def _fail(query: str, top_k: int = 5):
        raise RuntimeError("simulated retriever failure")

    return CallableRetrieverAdapter(_fail)


class FakeJudge(LLMJudge):
    """A deterministic judge for testing generation metrics without any
    network access or API key - always returns the same verdict,
    formatted the way a real model actually responds (wrapped in a
    markdown fence) so the full judge() parsing path gets exercised."""

    def __init__(self, score: float = 0.8, reasoning: str = "Looks reasonable.") -> None:
        self._score = score
        self._reasoning = reasoning
        self.call_count = 0

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.call_count += 1
        import json

        payload = json.dumps({"score": self._score, "reasoning": self._reasoning})
        return f"```json\n{payload}\n```"


@pytest.fixture
def fake_judge() -> FakeJudge:
    return FakeJudge()
