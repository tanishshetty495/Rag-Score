"""
Anthropic judge implementation.

Optional dependency (pip install rag-score[anthropic]) - only imported
when this class is actually instantiated.
"""

from __future__ import annotations

from rag_score.judges.base import LLMJudge


class AnthropicJudge(LLMJudge):
    def __init__(
        self,
        model: str = "claude-haiku-4-5",
        api_key: str | None = None,
        max_tokens: int = 512,
        max_retries: int = 2,
        retry_base_delay: float = 1.0,
    ) -> None:
        try:
            from anthropic import AsyncAnthropic
        except ImportError as e:
            raise ImportError(
                "The anthropic package is required for AnthropicJudge. "
                "Install it with: pip install rag-score[anthropic]"
            ) from e

        # api_key=None lets the SDK fall back to ANTHROPIC_API_KEY itself.
        self._client = AsyncAnthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = await self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text_blocks = [block.text for block in response.content if block.type == "text"]
        if not text_blocks:
            raise ValueError("Anthropic judge returned no text content")
        return "".join(text_blocks)
