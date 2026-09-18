"""
OpenAI judge implementation.

Optional dependency (pip install rag-score[openai]) - only imported
when this class is actually instantiated, so the core package install
never requires the openai SDK.
"""

from __future__ import annotations

from rag_score.judges.base import LLMJudge


class OpenAIJudge(LLMJudge):
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_retries: int = 2,
        retry_base_delay: float = 1.0,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError(
                "The openai package is required for OpenAIJudge. "
                "Install it with: pip install rag-score[openai]"
            ) from e

        # api_key=None lets the SDK fall back to the OPENAI_API_KEY
        # env var itself - no need to duplicate that lookup here.
        self._client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = await self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("OpenAI judge returned an empty response")
        return content
