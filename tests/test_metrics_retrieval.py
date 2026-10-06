"""Tests for precision@k, recall@k, mrr, ndcg - the offline-first metrics."""

from __future__ import annotations

import pytest

from rag_score.core.types import EvalResult, RetrievedChunk, TestCase
from rag_score.metrics.retrieval.mrr import MRR
from rag_score.metrics.retrieval.ndcg import NDCG
from rag_score.metrics.retrieval.precision_at_k import PrecisionAtK
from rag_score.metrics.retrieval.recall_at_k import RecallAtK


class TestPrecisionAtK:
    async def test_partial_match(self, sample_test_cases, sample_eval_result):
        tc = sample_test_cases[0]  # expects doc_1, doc_2
        metric = PrecisionAtK(k=3)
        score = await metric.score(tc, sample_eval_result)
        assert score == pytest.approx(1 / 3)  # only doc_1 of 3 retrieved is relevant

    async def test_perfect_match(self, sample_test_cases, perfect_eval_result):
        tc = sample_test_cases[0]
        metric = PrecisionAtK(k=2)
        score = await metric.score(tc, perfect_eval_result)
        assert score == pytest.approx(1.0)

    async def test_no_expected_docs_returns_zero(self, empty_eval_result):
        tc = TestCase(question="q", expected_doc_ids=[])
        metric = PrecisionAtK(k=3)
        score = await metric.score(tc, empty_eval_result)
        assert score == 0.0

    async def test_empty_retrieval_returns_zero(self, sample_test_cases, empty_eval_result):
        metric = PrecisionAtK(k=3)
        score = await metric.score(sample_test_cases[0], empty_eval_result)
        assert score == 0.0

    def test_rejects_non_positive_k(self):
        with pytest.raises(ValueError):
            PrecisionAtK(k=0)

    def test_name_includes_k(self):
        assert PrecisionAtK(k=7).name == "precision_at_7"


class TestRecallAtK:
    async def test_partial_match(self, sample_test_cases, sample_eval_result):
        tc = sample_test_cases[0]  # expects doc_1, doc_2 - only doc_1 retrieved
        metric = RecallAtK(k=3)
        score = await metric.score(tc, sample_eval_result)
        assert score == pytest.approx(0.5)

    async def test_perfect_match(self, sample_test_cases, perfect_eval_result):
        tc = sample_test_cases[0]
        metric = RecallAtK(k=2)
        score = await metric.score(tc, perfect_eval_result)
        assert score == pytest.approx(1.0)

    async def test_k_smaller_than_expected_docs(self, sample_test_cases, perfect_eval_result):
        tc = sample_test_cases[0]  # expects 2 docs
        metric = RecallAtK(k=1)  # only looks at top 1
        score = await metric.score(tc, perfect_eval_result)
        assert score == pytest.approx(0.5)  # found 1 of 2 expected


class TestMRR:
    async def test_relevant_at_rank_2(self, sample_test_cases, sample_eval_result):
        tc = sample_test_cases[0]
        metric = MRR()
        score = await metric.score(tc, sample_eval_result)
        assert score == pytest.approx(0.5)  # 1/rank, rank=2

    async def test_relevant_at_rank_1(self, sample_test_cases, perfect_eval_result):
        tc = sample_test_cases[0]
        metric = MRR()
        score = await metric.score(tc, perfect_eval_result)
        assert score == pytest.approx(1.0)

    async def test_no_relevant_found(self, sample_test_cases):
        tc = sample_test_cases[0]
        result = EvalResult(
            run_id="r",
            test_case_id="tc",
            retrieved_context=[RetrievedChunk(doc_id="doc_999", text="x")],
        )
        metric = MRR()
        score = await metric.score(tc, result)
        assert score == 0.0


class TestNDCG:
    async def test_perfect_ranking_scores_one(self, sample_test_cases, perfect_eval_result):
        tc = sample_test_cases[0]
        metric = NDCG(k=2)
        score = await metric.score(tc, perfect_eval_result)
        assert score == pytest.approx(1.0)

    async def test_partial_ranking_between_zero_and_one(
        self, sample_test_cases, sample_eval_result
    ):
        tc = sample_test_cases[0]
        metric = NDCG(k=3)
        score = await metric.score(tc, sample_eval_result)
        assert 0.0 < score < 1.0

    async def test_no_relevant_docs_in_expected_returns_zero(self):
        tc = TestCase(question="q", expected_doc_ids=[])
        result = EvalResult(run_id="r", test_case_id="tc", retrieved_context=[])
        metric = NDCG(k=3)
        score = await metric.score(tc, result)
        assert score == 0.0
