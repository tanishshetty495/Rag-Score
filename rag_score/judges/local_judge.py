"""
Local judge - talks to Ollama or any other OpenAI-compatible local
inference server (vLLM, LM Studio, llama.cpp's server mode, etc.)

This is what makes LLM-judge metrics usable with zero API cost and
zero data leaving the user's machine: point base_url at a local
server and everything else (prompts, parsing) is identical to
OpenAIJudge, since Ollama's /v1 endpoint speaks the OpenAI chat
completions API.

Reuses the openai SDK rather than writing a separate HTTP client,
since "OpenAI-compatible" specifically means it already knows how to
talk to these servers - just pointed at a different base_url with a
dummy API key.
"""

from __future__ import annotations

from rag_score.cache import JudgeCache
from rag_score.judges.base import LLMJudge


class LocalJudge(LLMJudge):
    provider = "local"

    def __init__(
        self,
        model: str = "llama3.1",
        base_url: str | None = None,
        api_key: str = "dummy",
        max_retries: int = 2,
        retry_base_delay: float = 1.0,
        cache: JudgeCache | None = None,
    ) -> None:
        super().__init__(cache=cache)
        self.model = model
        # Default to Ollama's base URL if none provided
        if base_url is None:
            base_url = "http://localhost:11434/v1"
        self.base_url = base_url
        self.api_key = api_key
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay

        # Initialize the OpenAI client (lazy import to avoid hard dependency)
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError(
                "The local judge requires the openai package. "
                "Install it with: pip install rag-score[local-ml]"
            ) from e

        self._client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        # Make the API call
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,  # For deterministic outputs
        )

        # Extract the text content
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("empty response")
        return content