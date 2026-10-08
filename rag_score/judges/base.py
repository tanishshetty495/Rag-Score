"""
LLM judge interface.

An LLMJudge only needs to implement one method - complete() - that
sends a system+user prompt to whatever model backs it and returns the
raw text response. Everything else (formatting the judge prompt,
parsing the JSON verdict out of the response) is shared logic that
generation metrics (Faithfulness, AnswerRelevance, ContextPrecision)
reuse via judge().

Keeping complete() as the only abstract method is what makes it easy
to plug in a new provider: OpenAI and Anthropic implementations here
are ~20 lines each, and a local Ollama/vLLM judge in Phase 3 will be
the same shape.
"""

from __future__ import annotations

import asyncio
import json
import re
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from rag_score.cache import JudgeCache


class JudgeVerdict(BaseModel):
    """A judge's verdict on one generation-quality question: a 0-1
    score plus the reasoning behind it, so users can audit *why* a
    faithfulness or relevance score came out the way it did rather
    than trusting an opaque number."""

    score: float = Field(ge=0.0, le=1.0)
    reasoning: str


class LLMJudge(ABC):
    """Base class every LLM-judge provider implements.

    max_retries/retry_base_delay control automatic retry-with-backoff
    around complete() - the actual network call, which is where
    transient failures (rate limits, connection blips, timeouts) show
    up in practice. Retrying complete() specifically (rather than the
    whole judge() call) means a resampled response also gets a fresh
    chance at producing well-formed JSON, without retry logic needing
    to know anything about JSON parsing.

    Defaults are deliberately modest (2 retries, 1s base delay) so a
    genuinely broken setup (bad API key, wrong model name) still fails
    fast rather than silently hanging for a long backoff sequence.
    """

    max_retries: int = 2
    retry_base_delay: float = 1.0

    def __init__(self, cache: JudgeCache | None = None) -> None:
        self.cache = cache

    @abstractmethod
    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Send the prompts to the underlying model and return its raw
        text response. Implementations should NOT try to parse JSON
        here - that's judge()'s job, kept separate so every provider
        gets identical, well-tested parsing behavior."""
        raise NotImplementedError

    async def judge(self, system_prompt: str, user_prompt: str) -> JudgeVerdict:
        """Get a completion (retrying transient failures automatically)
        and parse it into a JudgeVerdict. On a malformed response after
        all retries (the model didn't return valid JSON), this raises
        rather than silently returning a fabricated 0.0 - a parsing
        failure is a real problem the user should see, not a score
        that looks like a genuine low-quality verdict."""
        # If caching is enabled, try to get a cached verdict first
        if self.cache is not None:
            # Compute cache key from the judge inputs and identity
            key = self._cache_key(system_prompt, user_prompt)
            cached_verdict = await self.cache.get(key)
            if cached_verdict is not None:
                return cached_verdict

        # Cache miss or caching disabled: get a fresh verdict
        raw = await self._complete_with_retry(system_prompt, user_prompt)
        verdict = _parse_verdict(raw)

        # Store the verdict in cache if caching is enabled and the verdict is valid
        if self.cache is not None:
            await self.cache.set(key, verdict)

        return verdict

    def _cache_key(self, system_prompt: str, user_prompt: str) -> str:
        """Create a cache key for the given prompts and this judge's identity.

        Subclasses must set `provider` and `model_name` attributes.
        """
        # Import locally to avoid circular imports
        from rag_score.cache import _make_cache_key

        return _make_cache_key(
            metric_name="",  # Not needed because prompts are metric-specific
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            provider=self.provider,  # type: ignore[attr-defined]
            model_name=self.model_name,  # type: ignore[attr-defined]
        )

    async def _complete_with_retry(self, system_prompt: str, user_prompt: str) -> str:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return await self.complete(system_prompt, user_prompt)
            except Exception as exc:  # noqa: BLE001 - deliberately broad, see class docstring
                last_exc = exc
                if attempt < self.max_retries:
                    delay = self.retry_base_delay * (2**attempt)
                    await asyncio.sleep(delay)
        assert last_exc is not None  # loop always sets this before falling through
        raise last_exc


# Judge/synthesis prompts ask for JSON but models often wrap it in
# ```json fences or add a little prose around it - this strips both
# before parsing. Shared by _parse_verdict() here and the synthesis
# module's question/answer parsing, since both face the exact same
# "model didn't return clean JSON" problem.
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_BRACE_SPAN_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_object(raw: str) -> dict:
    """Best-effort extraction of a JSON object from raw LLM text:
    strips markdown code fences if present, then falls back to
    grabbing the first {...} span if the response has extra prose
    around the JSON. Raises ValueError if no parseable object is found.
    """
    text = raw.strip()

    fence_match = _JSON_FENCE_RE.search(text)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        brace_match = _BRACE_SPAN_RE.search(text)
        if not brace_match:
            raise ValueError(
                f"Response was not valid JSON and contained no parseable object: {raw!r}"
            ) from None
        return json.loads(brace_match.group(0))


def _parse_verdict(raw: str) -> JudgeVerdict:
    data = extract_json_object(raw)

    if "score" not in data or "reasoning" not in data:
        raise ValueError(f"Judge response JSON is missing 'score' or 'reasoning': {data!r}")

    # Clamp defensively - some models return 0-10 or 0-100 despite
    # instructions. If it's clearly out of 0-1 range, assume out of 10.
    score = float(data["score"])
    if score > 1.0:
        score = score / 10.0 if score <= 10.0 else score / 100.0
    score = max(0.0, min(1.0, score))

    return JudgeVerdict(score=score, reasoning=str(data["reasoning"]))
