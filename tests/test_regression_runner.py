"""
Regression/golden tests for the evaluation runner behavior:
- concurrency bound
- continue_on_error
- error recording
- multi-turn history
"""

from __future__ import annotations

import asyncio
import pytest

from rag_score.adapters.base import CallableGeneratorAdapter, CallableRetrieverAdapter
from rag_score.core.runner import RunConfig, run_evaluation
from rag_score.core.types import EvalResult, RetrievedChunk, TestCase
from rag_score.metrics.retrieval.precision_at_k import PrecisionAtK


class TestRunnerConcurrencyRegression:
    """Tests that concurrency is properly bounded."""

    @pytest.mark.asyncio
    async def test_concurrency_bound_respected(self):
        """Max concurrency limits simultaneous retrievals."""
        call_times = []
        max_concurrent = 0
        lock = asyncio.Lock()

        async def slow_retrieve(query: str, top_k: int = 5):
            nonlocal max_concurrent
            async with lock:
                call_times.append(asyncio.get_event_loop().time())
                if len(call_times) > max_concurrent:
                    max_concurrent = len(call_times)
            await asyncio.sleep(0.1)
            async with lock:
                call_times.pop()
            return [RetrievedChunk(doc_id="d1", text="x")]

        async def fast_generate(query: str, context):
            return "answer"

        test_cases = [TestCase(question=f"q{i}") for i in range(10)]
        retriever = CallableRetrieverAdapter(slow_retrieve)
        generator = CallableGeneratorAdapter(fast_generate)
        metrics = [PrecisionAtK(k=1)]

        config = RunConfig(run_id="test", max_concurrency=3)
        report = await run_evaluation(test_cases, retriever, generator, metrics, config)

        # Max concurrent should not exceed 3 (allowing some slack for timing)
        assert max_concurrent <= 3
        assert len(report.results) == 10


class TestRunnerContinueOnErrorRegression:
    """Tests for continue_on_error behavior."""

    @pytest.mark.asyncio
    async def test_continue_on_error_true_errors_recorded_not_raised(self):
        """With continue_on_error=True, errors are recorded in EvalResult."""
        call_count = 0

        async def failing_retrieve(query: str, top_k: int = 5):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise RuntimeError(f"Simulated failure {call_count}")
            return [RetrievedChunk(doc_id="d1", text="x")]

        async def fast_generate(query: str, context):
            return "answer"

        test_cases = [TestCase(question=f"q{i}") for i in range(5)]
        retriever = CallableRetrieverAdapter(failing_retrieve)
        generator = CallableGeneratorAdapter(fast_generate)
        metrics = [PrecisionAtK(k=1)]

        config = RunConfig(run_id="test", continue_on_error=True, max_concurrency=1)
        report = await run_evaluation(test_cases, retriever, generator, metrics, config)

        # All 5 test cases should have results
        assert len(report.results) == 5
        # First 3 should have errors
        errors = [r for r in report.results if r.error is not None]
        assert len(errors) == 3
        # Last 2 should succeed
        successes = [r for r in report.results if r.error is None]
        assert len(successes) == 2
        # Failed results should not be scored
        failed_eval_ids = {r.evaluation_id for r in errors}
        scored_failed = [s for s in report.scores if s.evaluation_id in failed_eval_ids]
        assert len(scored_failed) == 0

    @pytest.mark.asyncio
    async def test_continue_on_error_false_raises(self):
        """With continue_on_error=False, first error aborts the run."""
        call_count = 0

        async def failing_retrieve(query: str, top_k: int = 5):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Simulated failure")
            return [RetrievedChunk(doc_id="d1", text="x")]

        async def fast_generate(query: str, context):
            return "answer"

        test_cases = [TestCase(question=f"q{i}") for i in range(5)]
        retriever = CallableRetrieverAdapter(failing_retrieve)
        generator = CallableGeneratorAdapter(fast_generate)
        metrics = [PrecisionAtK(k=1)]

        config = RunConfig(run_id="test", continue_on_error=False, max_concurrency=1)

        with pytest.raises(RuntimeError, match="Simulated failure"):
            await run_evaluation(test_cases, retriever, generator, metrics, config)


class TestRunnerErrorRecordingRegression:
    """Tests that errors are properly recorded in EvalResult."""

    @pytest.mark.asyncio
    async def test_retrieval_error_recorded(self):
        """Retriever error captured in EvalResult.error."""
        async def failing_retrieve(query: str, top_k: int = 5):
            raise ValueError("Retriever broke")

        async def fast_generate(query: str, context):
            return "answer"

        test_cases = [TestCase(question="q1")]
        retriever = CallableRetrieverAdapter(failing_retrieve)
        generator = CallableGeneratorAdapter(fast_generate)
        metrics = [PrecisionAtK(k=1)]

        config = RunConfig(run_id="test", continue_on_error=True)
        report = await run_evaluation(test_cases, retriever, generator, metrics, config)

        assert len(report.results) == 1
        assert report.results[0].error is not None
        assert "ValueError" in report.results[0].error
        assert "Retriever broke" in report.results[0].error

    @pytest.mark.asyncio
    async def test_generation_error_recorded(self):
        """Generator error captured in EvalResult.error."""
        async def fast_retrieve(query: str, top_k: int = 5):
            return [RetrievedChunk(doc_id="d1", text="x")]

        async def failing_generate(query: str, context):
            raise RuntimeError("Generator broke")

        test_cases = [TestCase(question="q1")]
        retriever = CallableRetrieverAdapter(fast_retrieve)
        generator = CallableGeneratorAdapter(failing_generate)
        metrics = [PrecisionAtK(k=1)]

        config = RunConfig(run_id="test", continue_on_error=True)
        report = await run_evaluation(test_cases, retriever, generator, metrics, config)

        assert len(report.results) == 1
        assert report.results[0].error is not None
        assert "RuntimeError" in report.results[0].error
        assert "Generator broke" in report.results[0].error


class TestRunnerMultiTurnHistoryRegression:
    """Tests that conversation history is passed to generators that opt in."""

    @pytest.mark.asyncio
    async def test_history_passed_to_opt_in_generator(self):
        """Generator with 'history' parameter receives conversation history."""
        received_history = None

        async def fast_retrieve(query: str, top_k: int = 5):
            return [RetrievedChunk(doc_id="d1", text="context")]

        async def history_aware_generator(query: str, context, history=None):
            nonlocal received_history
            received_history = history
            return "answer"

        test_cases = [
            TestCase(
                question="followup",
                metadata={
                    "conversation_history": [
                        {"role": "user", "content": "first question"},
                        {"role": "assistant", "content": "first answer"},
                    ]
                },
            )
        ]
        retriever = CallableRetrieverAdapter(fast_retrieve)
        generator = CallableGeneratorAdapter(history_aware_generator)
        metrics = [PrecisionAtK(k=1)]

        config = RunConfig(run_id="test")
        await run_evaluation(test_cases, retriever, generator, metrics, config)

        assert received_history is not None
        assert len(received_history) == 2
        assert received_history[0]["role"] == "user"
        assert received_history[0]["content"] == "first question"
        assert received_history[1]["role"] == "assistant"
        assert received_history[1]["content"] == "first answer"

    @pytest.mark.asyncio
    async def test_history_not_passed_to_legacy_generator(self):
        """Generator without 'history' parameter does not receive history."""
        received_history = "NOT_SET"

        async def fast_retrieve(query: str, top_k: int = 5):
            return [RetrievedChunk(doc_id="d1", text="context")]

        async def legacy_generator(query: str, context):
            nonlocal received_history
            # This generator doesn't accept history parameter
            received_history = "CALLED_WITHOUT_HISTORY"
            return "answer"

        test_cases = [
            TestCase(
                question="followup",
                metadata={
                    "conversation_history": [
                        {"role": "user", "content": "first question"},
                    ]
                },
            )
        ]
        retriever = CallableRetrieverAdapter(fast_retrieve)
        generator = CallableGeneratorAdapter(legacy_generator)
        metrics = [PrecisionAtK(k=1)]

        config = RunConfig(run_id="test")
        await run_evaluation(test_cases, retriever, generator, metrics, config)

        # Legacy generator should be called without history kwarg
        assert received_history == "CALLED_WITHOUT_HISTORY"

    @pytest.mark.asyncio
    async def test_history_not_passed_when_test_case_lacks_history(self):
        """No history metadata -> generator not called with history."""
        received_history = "NOT_SET"

        async def fast_retrieve(query: str, top_k: int = 5):
            return [RetrievedChunk(doc_id="d1", text="context")]

        async def history_aware_generator(query: str, context, history=None):
            nonlocal received_history
            received_history = history
            return "answer"

        test_cases = [TestCase(question="first question")]  # No conversation_history
        retriever = CallableRetrieverAdapter(fast_retrieve)
        generator = CallableGeneratorAdapter(history_aware_generator)
        metrics = [PrecisionAtK(k=1)]

        config = RunConfig(run_id="test")
        await run_evaluation(test_cases, retriever, generator, metrics, config)

        assert received_history is None

    @pytest.mark.asyncio
    async def test_malformed_history_not_passed(self):
        """Malformed history entries -> treated as no history."""
        received_history = "NOT_SET"

        async def fast_retrieve(query: str, top_k: int = 5):
            return [RetrievedChunk(doc_id="d1", text="context")]

        async def history_aware_generator(query: str, context, history=None):
            nonlocal received_history
            received_history = history
            return "answer"

        test_cases = [
            TestCase(
                question="followup",
                metadata={"conversation_history": [{"role": "user"}]},  # Missing content
            )
        ]
        retriever = CallableRetrieverAdapter(fast_retrieve)
        generator = CallableGeneratorAdapter(history_aware_generator)
        metrics = [PrecisionAtK(k=1)]

        config = RunConfig(run_id="test")
        await run_evaluation(test_cases, retriever, generator, metrics, config)

        assert received_history is None


class TestRunnerProgressCallbackRegression:
    """Tests that on_progress is called correctly."""

    @pytest.mark.asyncio
    async def test_on_progress_called_once_per_test_case(self):
        """Progress callback called once per test case after retrieve+generate."""
        progress_count = 0

        async def fast_retrieve(query: str, top_k: int = 5):
            return [RetrievedChunk(doc_id="d1", text="x")]

        async def fast_generate(query: str, context):
            return "answer"

        test_cases = [TestCase(question=f"q{i}") for i in range(5)]
        retriever = CallableRetrieverAdapter(fast_retrieve)
        generator = CallableGeneratorAdapter(fast_generate)
        metrics = [PrecisionAtK(k=1)]

        def on_progress():
            nonlocal progress_count
            progress_count += 1

        config = RunConfig(run_id="test", max_concurrency=3)
        await run_evaluation(test_cases, retriever, generator, metrics, config, on_progress=on_progress)

        assert progress_count == 5

    @pytest.mark.asyncio
    async def test_on_progress_called_even_on_failure(self):
        """Progress callback called even when test case fails."""
        progress_count = 0
        call_count = 0

        async def sometimes_failing_retrieve(query: str, top_k: int = 5):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("fail")
            return [RetrievedChunk(doc_id="d1", text="x")]

        async def fast_generate(query: str, context):
            return "answer"

        test_cases = [TestCase(question=f"q{i}") for i in range(4)]
        retriever = CallableRetrieverAdapter(sometimes_failing_retrieve)
        generator = CallableGeneratorAdapter(fast_generate)
        metrics = [PrecisionAtK(k=1)]

        def on_progress():
            nonlocal progress_count
            progress_count += 1

        config = RunConfig(run_id="test", continue_on_error=True, max_concurrency=1)
        await run_evaluation(test_cases, retriever, generator, metrics, config, on_progress=on_progress)

        # All 4 test cases should trigger progress (2 fail, 2 succeed)
        assert progress_count == 4


class TestRunnerLatencyRecordingRegression:
    """Tests that latencies are recorded correctly."""

    @pytest.mark.asyncio
    async def test_retrieval_latency_recorded(self):
        """Retrieval latency in milliseconds."""
        async def slow_retrieve(query: str, top_k: int = 5):
            await asyncio.sleep(0.05)  # 50ms
            return [RetrievedChunk(doc_id="d1", text="x")]

        async def fast_generate(query: str, context):
            return "answer"

        test_cases = [TestCase(question="q1")]
        retriever = CallableRetrieverAdapter(slow_retrieve)
        generator = CallableGeneratorAdapter(fast_generate)
        metrics = [PrecisionAtK(k=1)]

        config = RunConfig(run_id="test")
        report = await run_evaluation(test_cases, retriever, generator, metrics, config)

        assert report.results[0].retrieval_latency_ms is not None
        # Should be at least 50ms (allowing some overhead)
        assert report.results[0].retrieval_latency_ms >= 40

    @pytest.mark.asyncio
    async def test_generation_latency_recorded(self):
        """Generation latency in milliseconds."""
        async def fast_retrieve(query: str, top_k: int = 5):
            return [RetrievedChunk(doc_id="d1", text="x")]

        async def slow_generate(query: str, context):
            await asyncio.sleep(0.03)  # 30ms
            return "answer"

        test_cases = [TestCase(question="q1")]
        retriever = CallableRetrieverAdapter(fast_retrieve)
        generator = CallableGeneratorAdapter(slow_generate)
        metrics = [PrecisionAtK(k=1)]

        config = RunConfig(run_id="test")
        report = await run_evaluation(test_cases, retriever, generator, metrics, config)

        assert report.results[0].generation_latency_ms is not None
        assert report.results[0].generation_latency_ms >= 25