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

import json
import re
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class JudgeVerdict(BaseModel):
    """A judge's verdict on one generation-quality question: a 0-1
    score plus the reasoning behind it, so users can audit *why* a
    faithfulness or relevance score came out the way it did rather
    than trusting an opaque number."""

    score: float = Field(ge=0.0, le=1.0)
    reasoning: str


class LLMJudge(ABC):
    """Base class every LLM-judge provider implements."""

    @abstractmethod
    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Send the prompts to the underlying model and return its raw
        text response. Implementations should NOT try to parse JSON
        here - that's judge()'s job, kept separate so every provider
        gets identical, well-tested parsing behavior."""
        raise NotImplementedError

    async def judge(self, system_prompt: str, user_prompt: str) -> JudgeVerdict:
        """Get a completion and parse it into a JudgeVerdict. On a
        malformed response (the model didn't return valid JSON), this
        raises rather than silently returning a fabricated 0.0 - a
        parsing failure is a real problem the user should see, not a
        score that looks like a genuine low-quality verdict."""
        raw = await self.complete(system_prompt, user_prompt)
        return _parse_verdict(raw)


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
        raise ValueError(
            f"Judge response JSON is missing 'score' or 'reasoning': {data!r}"
        )

    # Clamp defensively - some models return 0-10 or 0-100 despite
    # instructions. If it's clearly out of 0-1 range, assume out of 10.
    score = float(data["score"])
    if score > 1.0:
        score = score / 10.0 if score <= 10.0 else score / 100.0
    score = max(0.0, min(1.0, score))

    return JudgeVerdict(score=score, reasoning=str(data["reasoning"]))
