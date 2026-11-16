"""
Regression/golden tests for retrieval metrics with hand-computed expected values.
These lock in the exact current behavior of each retrieval metric.
"""

from __future__ import annotations

import pytest

from rag_score.core.models import EvaluationCase
from rag_score.core.types import EvalResult, RetrievedChunk, TestCase
from rag_score.metrics.retrieval.context_coverage import ContextCoverage
from rag_score.metrics.retrieval.hit_rate_at_k import HitRateAtK
from rag_score.metrics.retrieval.mrr import MRR
from rag_score.metrics.retrieval.ndcg import NDCG
from rag_score.metrics.retrieval.precision_at_k import PrecisionAtK
from rag_score.metrics.retrieval.recall_at_k import RecallAtK


class TestPrecisionAtKRegression:
    """Golden tests for Precision@k with hand-computed expected values."""

    @pytest.mark.asyncio
    async def test_precision_at_3_partial_match(self):
        """3 retrieved, 1 relevant at position 2 -> 1/3 = 0.333..."""
        tc = TestCase(question="q", expected_doc_ids=["doc_1", "doc_2"])
        result = EvalResult(
            run_id="r",
            test_case_id="tc",
            retrieved_context=[
                RetrievedChunk(doc_id="doc_9", text="irrelevant"),
                RetrievedChunk(doc_id="doc_1", text="relevant"),
                RetrievedChunk(doc_id="doc_5", text="irrelevant"),
            ],
        )
        metric = PrecisionAtK(k=3)
        score = await metric.score(tc, result)
        assert score == pytest.approx(1.0 / 3.0)

    @pytest.mark.asyncio
    async def test_precision_at_5_all_relevant(self):
        """5 retrieved, all 5 relevant -> 1.0"""
        tc = TestCase(question="q", expected_doc_ids=["d1", "d2", "d3", "d4", "d5"])
        result = EvalResult(
            run_id="r",
            test_case_id="tc",
            retrieved_context=[
                RetrievedChunk(doc_id=f"d{i}", text=f"text {i}") for i in range(1, 6)
            ],
        )
        metric = PrecisionAtK(k=5)
        score = await metric.score(tc, result)
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_precision_at_k_no_expected_docs(self):
        """No ground truth -> 0.0 (undefined, not an error)"""
        tc = TestCase(question="q", expected_doc_ids=[])
        result = EvalResult(run_id="r", test_case_id="tc", retrieved_context=[])
        metric = PrecisionAtK(k=5)
        score = await metric.score(tc, result)
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_precision_at_k_empty_retrieval(self):
        """Empty retrieval -> 0.0"""
        tc = TestCase(question="q", expected_doc_ids=["doc_1"])
        result = EvalResult(run_id="r", test_case_id="tc", retrieved_context=[])
        metric = PrecisionAtK(k=5)
        score = await metric.score(tc, result)
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_precision_at_k_retrieved_docs_have_no_ids(self):
        """Retrieved chunks have doc_id=None -> 0.0"""
        tc = TestCase(question="q", expected_doc_ids=["doc_1"])
        result = EvalResult(
            run_id="r",
            test_case_id="tc",
            retrieved_context=[RetrievedChunk(doc_id=None, text="x")],
        )
        metric = PrecisionAtK(k=5)
        score = await metric.score(tc, result)
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_precision_at_k_k_larger_than_retrieved(self):
        """k=10 but only 3 retrieved, 2 relevant -> 2/3"""
        tc = TestCase(question="q", expected_doc_ids=["doc_1", "doc_2", "doc_3"])
        result = EvalResult(
            run_id="r",
            test_case_id="tc",
            retrieved_context=[
                RetrievedChunk(doc_id="doc_1", text="a"),
                RetrievedChunk(doc_id="doc_9", text="b"),
                RetrievedChunk(doc_id="doc_2", text="c"),
            ],
        )
        metric = PrecisionAtK(k=10)
        score = await metric.score(tc, result)
        assert score == pytest.approx(2.0 / 3.0)

    @pytest.mark.asyncio
    async def test_precision_at_k_duplicate_doc_ids_in_retrieved(self):
        """Duplicate doc_ids in retrieved count separately (per position)"""
        tc = TestCase(question="q", expected_doc_ids=["doc_1"])
        result = EvalResult(
            run_id="r",
            test_case_id="tc",
            retrieved_context=[
                RetrievedChunk(doc_id="doc_1", text="a"),
                RetrievedChunk(doc_id="doc_1", text="a again"),
                RetrievedChunk(doc_id="doc_9", text="b"),
            ],
        )
        metric = PrecisionAtK(k=3)
        score = await metric.score(tc, result)
        # Both doc_1 count as relevant at their positions
        assert score == pytest.approx(2.0 / 3.0)

    def test_precision_at_k_rejects_zero_k(self):
        """k must be positive"""
        with pytest.raises(ValueError):
            PrecisionAtK(k=0)

    def test_precision_at_k_name_includes_k(self):
        """Metric name includes k for side-by-side reporting"""
        assert PrecisionAtK(k=7).name == "precision_at_7"


class TestRecallAtKRegression:
    """Golden tests for Recall@k with hand-computed expected values."""

    @pytest.mark.asyncio
    async def test_recall_at_3_partial_match(self):
        """3 expected, 1 found in top 3 -> 1/3 = 0.333..."""
        tc = TestCase(question="q", expected_doc_ids=["doc_1", "doc_2", "doc_3"])
        result = EvalResult(
            run_id="r",
            test_case_id="tc",
            retrieved_context=[
                RetrievedChunk(doc_id="doc_9", text="irrelevant"),
                RetrievedChunk(doc_id="doc_1", text="relevant"),
                RetrievedChunk(doc_id="doc_5", text="irrelevant"),
            ],
        )
        metric = RecallAtK(k=3)
        score = await metric.score(tc, result)
        assert score == pytest.approx(1.0 / 3.0)

    @pytest.mark.asyncio
    async def test_recall_at_k_all_found(self):
        """All expected docs found in top k -> 1.0"""
        tc = TestCase(question="q", expected_doc_ids=["doc_1", "doc_2"])
        result = EvalResult(
            run_id="r",
            test_case_id="tc",
            retrieved_context=[
                RetrievedChunk(doc_id="doc_1", text="a"),
                RetrievedChunk(doc_id="doc_2", text="b"),
            ],
        )
        metric = RecallAtK(k=5)
        score = await metric.score(tc, result)
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_recall_at_k_k_smaller_than_expected(self):
        """k=1 but 2 expected docs, 1 found -> 0.5"""
        tc = TestCase(question="q", expected_doc_ids=["doc_1", "doc_2"])
        result = EvalResult(
            run_id="r",
            test_case_id="tc",
            retrieved_context=[RetrievedChunk(doc_id="doc_1", text="a")],
        )
        metric = RecallAtK(k=1)
        score = await metric.score(tc, result)
        assert score == 0.5

    @pytest.mark.asyncio
    async def test_recall_at_k_no_expected_docs(self):
        """No ground truth -> 0.0"""
        tc = TestCase(question="q", expected_doc_ids=[])
        result = EvalResult(run_id="r", test_case_id="tc", retrieved_context=[])
        metric = RecallAtK(k=5)
        score = await metric.score(tc, result)
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_recall_at_k_empty_retrieval(self):
        """Empty retrieval -> 0.0"""
        tc = TestCase(question="q", expected_doc_ids=["doc_1"])
        result = EvalResult(run_id="r", test_case_id="tc", retrieved_context=[])
        metric = RecallAtK(k=5)
        score = await metric.score(tc, result)
        assert score == 0.0


class TestMRRRegression:
    """Golden tests for MRR with hand-computed expected values."""

    @pytest.mark.asyncio
    async def test_mrr_relevant_at_rank_1(self):
        """First result relevant -> 1.0"""
        tc = TestCase(question="q", expected_doc_ids=["doc_1"])
        result = EvalResult(
            run_id="r",
            test_case_id="tc",
            retrieved_context=[RetrievedChunk(doc_id="doc_1", text="a")],
        )
        metric = MRR()
        score = await metric.score(tc, result)
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_mrr_relevant_at_rank_3(self):
        """First relevant at rank 3 -> 1/3 = 0.333..."""
        tc = TestCase(question="q", expected_doc_ids=["doc_1"])
        result = EvalResult(
            run_id="r",
            test_case_id="tc",
            retrieved_context=[
                RetrievedChunk(doc_id="doc_9", text="a"),
                RetrievedChunk(doc_id="doc_8", text="b"),
                RetrievedChunk(doc_id="doc_1", text="c"),
            ],
        )
        metric = MRR()
        score = await metric.score(tc, result)
        assert score == pytest.approx(1.0 / 3.0)

    @pytest.mark.asyncio
    async def test_mrr_no_relevant_found(self):
        """No relevant docs in retrieval -> 0.0"""
        tc = TestCase(question="q", expected_doc_ids=["doc_1"])
        result = EvalResult(
            run_id="r",
            test_case_id="tc",
            retrieved_context=[RetrievedChunk(doc_id="doc_9", text="a")],
        )
        metric = MRR()
        score = await metric.score(tc, result)
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_mrr_no_expected_docs(self):
        """No ground truth -> 0.0"""
        tc = TestCase(question="q", expected_doc_ids=[])
        result = EvalResult(run_id="r", test_case_id="tc", retrieved_context=[])
        metric = MRR()
        score = await metric.score(tc, result)
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_mrr_multiple_relevant_first_wins(self):
        """Multiple relevant docs, first one determines rank"""
        tc = TestCase(question="q", expected_doc_ids=["doc_1", "doc_2"])
        result = EvalResult(
            run_id="r",
            test_case_id="tc",
            retrieved_context=[
                RetrievedChunk(doc_id="doc_9", text="a"),
                RetrievedChunk(doc_id="doc_2", text="b"),  # rank 2
                RetrievedChunk(doc_id="doc_1", text="c"),  # rank 3
            ],
        )
        metric = MRR()
        score = await metric.score(tc, result)
        assert score == pytest.approx(1.0 / 2.0)


class TestNDCGRegression:
    """Golden tests for NDCG@k with hand-computed expected values."""

    @pytest.mark.asyncio
    async def test_ndcg_perfect_ranking(self):
        """All relevant at top -> 1.0"""
        tc = TestCase(question="q", expected_doc_ids=["doc_1", "doc_2"])
        result = EvalResult(
            run_id="r",
            test_case_id="tc",
            retrieved_context=[
                RetrievedChunk(doc_id="doc_1", text="a"),
                RetrievedChunk(doc_id="doc_2", text="b"),
            ],
        )
        metric = NDCG(k=2)
        score = await metric.score(tc, result)
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_ndcg_swapped_relevant(self):
        """Relevant at ranks 2 and 3 (k=3) -> DCG = 1/log2(3) + 1/log2(4)"""
        import math
        tc = TestCase(question="q", expected_doc_ids=["doc_1", "doc_2"])
        result = EvalResult(
            run_id="r",
            test_case_id="tc",
            retrieved_context=[
                RetrievedChunk(doc_id="doc_9", text="a"),
                RetrievedChunk(doc_id="doc_1", text="b"),
                RetrievedChunk(doc_id="doc_2", text="c"),
            ],
        )
        metric = NDCG(k=3)
        score = await metric.score(tc, result)
        # Ideal: rel at 1,2 -> DCG = 1 + 1/log2(3)
        # Actual: rel at 2,3 -> DCG = 1/log2(3) + 1/log2(4)
        # NDCG = actual / ideal
        ideal_dcg = 1.0 + 1.0 / math.log2(3)
        actual_dcg = 1.0 / math.log2(3) + 1.0 / math.log2(4)
        expected = actual_dcg / ideal_dcg
        assert score == pytest.approx(expected)

    @pytest.mark.asyncio
    async def test_ndcg_no_expected_docs(self):
        """No ground truth -> 0.0"""
        tc = TestCase(question="q", expected_doc_ids=[])
        result = EvalResult(run_id="r", test_case_id="tc", retrieved_context=[])
        metric = NDCG(k=5)
        score = await metric.score(tc, result)
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_ndcg_k_larger_than_expected(self):
        """k=5, 2 expected, both found at ranks 1,2"""
        tc = TestCase(question="q", expected_doc_ids=["doc_1", "doc_2"])
        result = EvalResult(
            run_id="r",
            test_case_id="tc",
            retrieved_context=[
                RetrievedChunk(doc_id="doc_1", text="a"),
                RetrievedChunk(doc_id="doc_2", text="b"),
                RetrievedChunk(doc_id="doc_9", text="c"),
            ],
        )
        metric = NDCG(k=5)
        score = await metric.score(tc, result)
        # Ideal: 1,1,0,0,0 -> DCG = 1 + 1/log2(3)
        # Actual: same -> 1.0
        assert score == 1.0


class TestHitRateAtKRegression:
    """Golden tests for HitRate@k with hand-computed expected values.

    Note: HitRateAtK uses the NEW metric interface with EvaluationCase.
    """

    def test_hit_rate_at_k_hit(self):
        """At least one relevant in top k -> 1.0"""
        case = EvaluationCase(
            id="tc",
            question="q",
            contexts=["a", "b"],
            metadata={"expected_doc_ids": ["doc_1"], "retrieved_doc_ids": ["doc_9", "doc_1"]},
        )
        metric = HitRateAtK(k=2)
        result = metric.evaluate(case)
        assert result.score_value == 1.0

    def test_hit_rate_at_k_miss(self):
        """No relevant in top k -> 0.0"""
        case = EvaluationCase(
            id="tc",
            question="q",
            contexts=["a"],
            metadata={"expected_doc_ids": ["doc_1"], "retrieved_doc_ids": ["doc_9"]},
        )
        metric = HitRateAtK(k=1)
        result = metric.evaluate(case)
        assert result.score_value == 0.0

    def test_hit_rate_at_k_no_expected_docs(self):
        """No ground truth -> 0.0"""
        case = EvaluationCase(
            id="tc",
            question="q",
            contexts=[],
            metadata={"expected_doc_ids": [], "retrieved_doc_ids": []},
        )
        metric = HitRateAtK(k=5)
        result = metric.evaluate(case)
        assert result.score_value == 0.0

    def test_hit_rate_at_k_empty_retrieval(self):
        """Empty retrieval -> 0.0"""
        case = EvaluationCase(
            id="tc",
            question="q",
            contexts=[],
            metadata={"expected_doc_ids": ["doc_1"], "retrieved_doc_ids": []},
        )
        metric = HitRateAtK(k=5)
        result = metric.evaluate(case)
        assert result.score_value == 0.0

    def test_hit_rate_at_k_retrieved_no_doc_ids(self):
        """Retrieved chunks have no doc_ids -> 0.0"""
        case = EvaluationCase(
            id="tc",
            question="q",
            contexts=["x"],
            metadata={"expected_doc_ids": ["doc_1"], "retrieved_doc_ids": [None]},
        )
        metric = HitRateAtK(k=5)
        result = metric.evaluate(case)
        assert result.score_value == 0.0


class TestContextCoverageRegression:
    """Golden tests for ContextCoverage with hand-computed expected values.

    Note: ContextCoverage uses the NEW metric interface with EvaluationCase.
    """

    def test_context_coverage_perfect(self):
        """All reference words in context -> 1.0"""
        case = EvaluationCase(
            id="tc",
            question="q",
            reference="Paris is the capital of France",
            contexts=["Paris is the capital city of France"],
        )
        metric = ContextCoverage()
        result = metric.evaluate(case)
        # Reference words: paris, is, the, capital, of, france (6)
        # Context words: paris, is, the, capital, city, of, france (7)
        # Intersection: paris, is, the, capital, of, france (6) -> 6/6 = 1.0
        assert result.score_value == 1.0

    def test_context_coverage_partial(self):
        """Half reference words in context -> 0.666..."""
        case = EvaluationCase(
            id="tc",
            question="q",
            reference="Paris is the capital of France",
            contexts=["Berlin is the capital of Germany"],
        )
        metric = ContextCoverage()
        result = metric.evaluate(case)
        # Reference words: paris, is, the, capital, of, france (6)
        # Context words: berlin, is, the, capital, of, germany (6)
        # Intersection: is, the, capital, of (4) -> 4/6 = 0.666...
        assert result.score_value == pytest.approx(4.0 / 6.0)

    def test_context_coverage_no_reference(self):
        """No ground truth answer -> 0.0"""
        case = EvaluationCase(
            id="tc",
            question="q",
            reference=None,
            contexts=[],
        )
        metric = ContextCoverage()
        result = metric.evaluate(case)
        assert result.score_value == 0.0

    def test_context_coverage_no_context(self):
        """No retrieved context -> 0.0"""
        case = EvaluationCase(
            id="tc",
            question="q",
            reference="Paris",
            contexts=[],
        )
        metric = ContextCoverage()
        result = metric.evaluate(case)
        assert result.score_value == 0.0

    def test_context_coverage_unicode(self):
        """Unicode text handled correctly"""
        case = EvaluationCase(
            id="tc",
            question="q",
            reference="Café résumé naïve",
            contexts=["Café and résumé"],
        )
        metric = ContextCoverage()
        result = metric.evaluate(case)
        # Should not crash, should compute something reasonable
        assert 0.0 <= result.score_value <= 1.0

    def test_context_coverage_very_long_text(self):
        """Very long reference and context"""
        case = EvaluationCase(
            id="tc",
            question="q",
            reference="word " * 10000,
            contexts=["word " * 10000],
        )
        metric = ContextCoverage()
        result = metric.evaluate(case)
        assert result.score_value == 1.0