"""
Anthropic judge implementation.

Optional dependency (pip install rag-score[anthropic]) - only imported
when this class is actually instantiated.
"""

from __future__ import annotations

from rag_score.judges.base import LLMJudge


class AnthropicJudge(LLMJudge):
    provider = "anthropic"

    def __init__(
        self,
        model: str = "claude-haiku-4-5",
        api_key: str | None = None,
        max_retries: int = 2,
        retry_base_delay: float = 1.0,
        max_tokens: int = 512,
        cache: "JudgeCache" | None = None,
    ) -> None:
        super().__init__(cache=cache)
        self.model = model
        self.api_key = api_key
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.max_tokens = max_tokens

        # Initialize the Anthropic client (lazy import to avoid hard dependency)
        try:
            from anthropic import AsyncAnthropic
        except ImportError as e:
            raise ImportError(
                "The anthropic judge requires the anthropic package. "
                "Install it with: pip install rag-score[anthropic]"
            ) from e

        self._client = AsyncAnthropic(api_key=api_key)

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        # Prepare the messages
        messages = [{"role": "user", "content": user_prompt}]

        # Make the API call with system prompt
        response = await self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=messages,
        )

        # Extract text content from all text blocks
        text_blocks = [block.text for block in response.content if hasattr(block, 'text')]
        if not text_blocks:
            raise ValueError("no text")
        content = "".join(text_blocks)
        if content is None:
            raise ValueError("empty response")
        return content