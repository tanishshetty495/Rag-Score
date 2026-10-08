"""
OpenAI judge implementation.

Optional dependency (pip install rag-score[openai]) - only imported
when this class is actually instantiated, so the core package install
never requires the openai SDK.
"""

from __future__ import annotations

from rag_score.cache import JudgeCache
from rag_score.judges.base import LLMJudge


class OpenAIJudge(LLMJudge):
    provider = "openai"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        max_retries: int = 2,
        retry_base_delay: float = 1.0,
        temperature: float = 0.0,
        cache: JudgeCache | None = None,
    ) -> None:
        super().__init__(cache=cache)
        self.model = model
        self.api_key = api_key
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.temperature = temperature

        # Initialize the OpenAI client (lazy import to avoid hard dependency)
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError(
                "The openai judge requires the openai package. "
                "Install it with: pip install rag-score[openai]"
            ) from e

        self._client = AsyncOpenAI(api_key=api_key)

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        # Make the API call
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,  # For deterministic outputs
        )

        # Extract the text content
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("empty response")
        return content