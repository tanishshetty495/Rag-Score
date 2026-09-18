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

from rag_score.judges.base import LLMJudge

# Ollama's default local server address - the vast majority of users
# running this locally won't need to override it.
_DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"


class LocalJudge(LLMJudge):
    def __init__(
        self,
        model: str,
        base_url: str = _DEFAULT_OLLAMA_BASE_URL,
        api_key: str = "not-needed",
        temperature: float = 0.0,
        max_retries: int = 2,
        retry_base_delay: float = 1.0,
    ) -> None:
        """
        model: the model name as your local server knows it, e.g.
               "llama3.1" for Ollama or a served model id for vLLM.
        base_url: the server's OpenAI-compatible endpoint. Defaults to
                  Ollama's standard local address; override for vLLM,
                  LM Studio, etc.
        api_key: most local servers don't check this, but the OpenAI
                 SDK requires a non-empty string to construct a client.
        max_retries/retry_base_delay: useful even locally - a server
                 that's still loading a model on first request can
                 return a transient error that clears up within a
                 couple seconds.
        """
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError(
                "The openai package is required for LocalJudge (it's used as "
                "a client for any OpenAI-compatible endpoint, not to call "
                "OpenAI's own API). Install it with: pip install rag-score[openai]"
            ) from e

        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
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
            raise ValueError(
                f"Local judge at {self._client.base_url} returned an empty response. "
                f"Check that the server is running and '{self.model}' is available."
            )
        return content
