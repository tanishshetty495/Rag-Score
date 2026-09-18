"""Tests for the retry/backoff logic added to LLMJudge.judge().
Uses a FlakyJudge that fails a configurable number of times before
succeeding, with a tiny retry_base_delay so tests run fast without
actually depending on real timing thresholds for pass/fail."""

from __future__ import annotations

import pytest

from rag_score.judges.base import LLMJudge


class _FlakyJudge(LLMJudge):
    def __init__(self, fail_count: int, max_retries: int = 2, retry_base_delay: float = 0.001):
        self.fail_count = fail_count
        self.call_count = 0
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.call_count += 1
        if self.call_count <= self.fail_count:
            raise ConnectionError(f"simulated failure #{self.call_count}")
        return '{"score": 0.9, "reasoning": "eventually succeeded"}'


class TestRetryBackoff:
    async def test_succeeds_after_transient_failures(self):
        judge = _FlakyJudge(fail_count=2, max_retries=2)
        verdict = await judge.judge("sys", "user")
        assert verdict.score == 0.9
        assert judge.call_count == 3  # 1 initial attempt + 2 retries

    async def test_no_retry_needed_on_immediate_success(self):
        judge = _FlakyJudge(fail_count=0, max_retries=2)
        await judge.judge("sys", "user")
        assert judge.call_count == 1  # no wasted retry attempts

    async def test_exhausting_retries_raises_last_exception(self):
        judge = _FlakyJudge(fail_count=10, max_retries=2)
        with pytest.raises(ConnectionError, match="simulated failure #3"):
            await judge.judge("sys", "user")
        assert judge.call_count == 3  # 1 initial + 2 retries, then gives up

    async def test_zero_max_retries_means_single_attempt(self):
        judge = _FlakyJudge(fail_count=1, max_retries=0)
        with pytest.raises(ConnectionError):
            await judge.judge("sys", "user")
        assert judge.call_count == 1

    async def test_default_retry_settings_on_base_class(self):
        assert LLMJudge.max_retries == 2
        assert LLMJudge.retry_base_delay == 1.0

    async def test_backoff_is_exponential(self):
        """Verify delays roughly double each retry rather than being
        constant - uses a slightly larger base_delay here specifically
        to make timing assertions meaningful without being flaky."""
        import time

        judge = _FlakyJudge(fail_count=2, max_retries=2, retry_base_delay=0.05)
        t0 = time.perf_counter()
        await judge.judge("sys", "user")
        elapsed = time.perf_counter() - t0
        # Expected delays: 0.05 * 2^0 + 0.05 * 2^1 = 0.05 + 0.10 = 0.15s
        assert 0.12 < elapsed < 0.35, f"unexpected total backoff time: {elapsed}"
