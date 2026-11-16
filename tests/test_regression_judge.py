"""
Regression/golden tests for LLM judges:
- Retry/backoff behavior
- JSON fence stripping
- Cache invalidation
"""

from __future__ import annotations

import pytest

from rag_score.cache import InMemoryCacheBackend
from rag_score.judges.base import JudgeVerdict, LLMJudge


class FakeJudgeWithRetry(LLMJudge):
    """A judge that fails a few times then succeeds."""

    provider = "test"
    model = "test-model"

    def __init__(self, fail_count: int = 2, score: float = 0.8, cache=None):
        self.fail_count = fail_count
        self._score = score
        self.call_count = 0
        super().__init__(cache=cache)

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.call_count += 1
        import json

        if self.call_count <= self.fail_count:
            raise RuntimeError(f"Simulated failure {self.call_count}")

        payload = json.dumps({"score": self._score, "reasoning": "Success after retries"})
        return f"```json\n{payload}\n```"


class FakeJudgeBadJSON(LLMJudge):
    """A judge that returns malformed JSON."""

    provider = "test"
    model = "test-model"

    def __init__(self, bad_response: str, cache=None):
        self.bad_response = bad_response
        self.call_count = 0
        super().__init__(cache=cache)

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.call_count += 1
        return self.bad_response


class TestJudgeRetryBackoffRegression:
    """Tests for retry/backoff behavior."""

    @pytest.mark.asyncio
    async def test_retry_on_transient_failure(self):
        """Judge retries on transient failures and eventually succeeds."""
        judge = FakeJudgeWithRetry(fail_count=2, score=0.7)
        verdict = await judge.judge("system", "user")
        assert verdict.score == 0.7
        assert judge.call_count == 3  # 2 failures + 1 success

    @pytest.mark.asyncio
    async def test_max_retries_exhausted_raises(self):
        """After max retries, the exception is propagated."""
        judge = FakeJudgeWithRetry(fail_count=5, score=0.7)  # More than max retries (3)
        with pytest.raises(RuntimeError, match="Simulated failure"):
            await judge.judge("system", "user")

    @pytest.mark.asyncio
    async def test_no_retry_on_success(self):
        """Successful first call doesn't retry."""
        judge = FakeJudgeWithRetry(fail_count=0, score=0.9)
        verdict = await judge.judge("system", "user")
        assert verdict.score == 0.9
        assert judge.call_count == 1

    @pytest.mark.asyncio
    async def test_backoff_timing(self):
        """Verify backoff delays increase between retries."""
        # This is a timing test - we just verify it doesn't hang
        # and completes in reasonable time (not minutes)
        judge = FakeJudgeWithRetry(fail_count=2, score=0.5)
        import time
        start = time.monotonic()
        verdict = await judge.judge("system", "user")
        elapsed = time.monotonic() - start
        # Should complete with exponential backoff (1s + 2s = 3s minimum)
        # but we allow more time for CI variability
        assert elapsed < 15
        assert verdict.score == 0.5


class TestJudgeJSONFenceStrippingRegression:
    """Tests for JSON fence stripping."""

    @pytest.mark.asyncio
    async def test_markdown_fence_stripped(self):
        """JSON wrapped in markdown fences is parsed correctly."""
        judge = FakeJudgeBadJSON("```json\n{\"score\": 0.75, \"reasoning\": \"Good\"}\n```")
        verdict = await judge.judge("system", "user")
        assert verdict.score == 0.75
        assert verdict.reasoning == "Good"

    @pytest.mark.asyncio
    async def test_no_fence_parsed_directly(self):
        """JSON without fences is parsed correctly."""
        judge = FakeJudgeBadJSON('{"score": 0.5, "reasoning": "OK"}')
        verdict = await judge.judge("system", "user")
        assert verdict.score == 0.5
        assert verdict.reasoning == "OK"

    @pytest.mark.asyncio
    async def test_extra_text_before_fence(self):
        """Extra text before fence is handled."""
        judge = FakeJudgeBadJSON('Here is the result:\n```json\n{"score": 0.8, "reasoning": "Great"}\n```')
        verdict = await judge.judge("system", "user")
        assert verdict.score == 0.8

    @pytest.mark.asyncio
    async def test_extra_text_after_fence(self):
        """Extra text after fence is handled."""
        judge = FakeJudgeBadJSON('```json\n{"score": 0.6, "reasoning": "Fair"}\n```\nEnd of response.')
        verdict = await judge.judge("system", "user")
        assert verdict.score == 0.6

    @pytest.mark.asyncio
    async def test_single_backticks(self):
        """Single backticks are handled."""
        judge = FakeJudgeBadJSON('`{"score": 0.4, "reasoning": "Poor"}`')
        verdict = await judge.judge("system", "user")
        assert verdict.score == 0.4

    @pytest.mark.asyncio
    async def test_malformed_json_raises(self):
        """Malformed JSON raises an error."""
        judge = FakeJudgeBadJSON('not json at all')
        with pytest.raises(ValueError, match="Response was not valid JSON"):
            await judge.judge("system", "user")

    @pytest.mark.asyncio
    async def test_missing_score_key_raises(self):
        """Missing score key raises an error."""
        judge = FakeJudgeBadJSON('{"reasoning": "No score"}')
        with pytest.raises(ValueError, match="missing 'score'"):
            await judge.judge("system", "user")

    @pytest.mark.asyncio
    async def test_score_out_of_range_clamped(self):
        """Score outside 0-1 range is clamped (divided by 10 if >1, then clamped to 0-1)."""
        # Current implementation: score > 1.0 is divided by 10 (if <=10) or 100 (if >10),
        # then clamped to [0, 1]
        judge = FakeJudgeBadJSON('{"score": 1.5, "reasoning": "Over"}')
        verdict = await judge.judge("system", "user")
        # 1.5 > 1.0, so divide by 10 -> 0.15, clamp to [0,1] -> 0.15
        assert verdict.score == 0.15

    @pytest.mark.asyncio
    async def test_score_out_of_range_100_clamped(self):
        """Score > 10 is divided by 100 then clamped."""
        judge = FakeJudgeBadJSON('{"score": 85, "reasoning": "Out of 100"}')
        verdict = await judge.judge("system", "user")
        # 85 > 10, so divide by 100 -> 0.85, clamp to [0,1] -> 0.85
        assert verdict.score == 0.85

    @pytest.mark.asyncio
    async def test_negative_score_clamped_to_zero(self):
        """Negative scores are clamped to 0."""
        judge = FakeJudgeBadJSON('{"score": -0.5, "reasoning": "Negative"}')
        verdict = await judge.judge("system", "user")
        assert verdict.score == 0.0


class TestJudgeCacheInvalidationRegression:
    """Tests for judge cache invalidation."""

    @pytest.mark.asyncio
    async def test_cache_stores_results(self):
        """Cache stores and returns results."""
        cache = InMemoryCacheBackend()
        judge = FakeJudgeWithRetry(fail_count=0, score=0.8, cache=cache)

        # First call
        verdict1 = await judge.judge("system", "user")
        assert verdict1.score == 0.8
        assert judge.call_count == 1

        # Second call with same prompts should use cache
        verdict2 = await judge.judge("system", "user")
        assert verdict2.score == 0.8
        assert judge.call_count == 1  # Not called again

    @pytest.mark.asyncio
    async def test_different_prompts_not_cached(self):
        """Different prompts are not cached together."""
        cache = InMemoryCacheBackend()
        judge = FakeJudgeWithRetry(fail_count=0, score=0.8, cache=cache)

        await judge.judge("system1", "user1")
        await judge.judge("system2", "user2")
        assert judge.call_count == 2  # Both called

    @pytest.mark.asyncio
    async def test_cache_key_includes_both_prompts(self):
        """Cache key includes both system and user prompts."""
        cache = InMemoryCacheBackend()
        judge = FakeJudgeWithRetry(fail_count=0, score=0.8, cache=cache)

        await judge.judge("same_system", "user1")
        await judge.judge("same_system", "user2")
        assert judge.call_count == 2  # Different user prompts = different cache entries

    @pytest.mark.asyncio
    async def test_cache_key_based_on_provider_and_model_only(self):
        """Cache key is based on provider and model, not other judge params.

        Two judges with same provider/model but different scores will share cache.
        """
        cache = InMemoryCacheBackend()
        judge1 = FakeJudgeWithRetry(fail_count=0, score=0.5, cache=cache)
        judge2 = FakeJudgeWithRetry(fail_count=0, score=0.9, cache=cache)

        await judge1.judge("system", "user")
        await judge2.judge("system", "user")
        # Both have same provider/model -> same cache key -> judge2 uses cache
        assert judge1.call_count == 1
        assert judge2.call_count == 0  # Cache hit from judge1

    @pytest.mark.asyncio
    async def test_no_cache_when_none(self):
        """No caching when cache=None."""
        judge = FakeJudgeWithRetry(fail_count=0, score=0.8, cache=None)

        await judge.judge("system", "user")
        await judge.judge("system", "user")
        assert judge.call_count == 2  # Called twice


class TestJudgeVerdictRegression:
    """Tests for JudgeVerdict dataclass."""

    def test_verdict_creation(self):
        """JudgeVerdict can be created with score and reasoning."""
        verdict = JudgeVerdict(score=0.8, reasoning="Good")
        assert verdict.score == 0.8
        assert verdict.reasoning == "Good"

    def test_verdict_reasoning_required(self):
        """JudgeVerdict requires reasoning field (not optional)."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            JudgeVerdict(score=0.5)  # missing reasoning

    def test_verdict_mutable(self):
        """JudgeVerdict is mutable (regular pydantic model, not frozen)."""
        verdict = JudgeVerdict(score=0.8, reasoning="test")
        verdict.score = 0.5  # This works - model is not frozen
        assert verdict.score == 0.5


class TestLLMJudgeBaseClass:
    """Tests for LLMJudge base class behavior."""

    @pytest.mark.asyncio
    async def test_judge_calls_complete(self):
        """judge() calls complete() with correct prompts."""
        class TrackingJudge(LLMJudge):
            def __init__(self):
                self.last_system = None
                self.last_user = None
                super().__init__()

            async def complete(self, system_prompt: str, user_prompt: str) -> str:
                self.last_system = system_prompt
                self.last_user = user_prompt
                import json
                return f"```json\n{json.dumps({'score': 0.5, 'reasoning': 'test'})}\n```"

        judge = TrackingJudge()
        await judge.judge("sys prompt", "user prompt")
        assert judge.last_system == "sys prompt"
        assert judge.last_user == "user prompt"

    @pytest.mark.asyncio
    async def test_judge_returns_verdict(self):
        """judge() returns JudgeVerdict."""
        class SimpleJudge(LLMJudge):
            async def complete(self, system_prompt: str, user_prompt: str) -> str:
                import json
                return f"```json\n{json.dumps({'score': 0.7, 'reasoning': 'ok'})}\n```"

        judge = SimpleJudge()
        verdict = await judge.judge("sys", "user")
        assert isinstance(verdict, JudgeVerdict)
        assert verdict.score == 0.7
        assert verdict.reasoning == "ok"