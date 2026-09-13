"""Tests for judges/local_judge.py. No real local server is spun up -
we verify construction (correct base_url, model) and error handling
by monkeypatching the underlying OpenAI-compatible client call, since
that's the actual integration point this class adds over OpenAIJudge."""

from __future__ import annotations

import pytest

openai = pytest.importorskip("openai")

from rag_score.judges.local_judge import LocalJudge


class TestLocalJudgeConstruction:
    def test_defaults_to_ollama_base_url(self):
        judge = LocalJudge(model="llama3.1")
        assert "localhost:11434" in str(judge._client.base_url)
        assert judge.model == "llama3.1"

    def test_custom_base_url_for_other_servers(self):
        judge = LocalJudge(model="my-model", base_url="http://localhost:8000/v1")
        assert "8000" in str(judge._client.base_url)

    def test_does_not_require_a_real_api_key(self):
        # Should construct without raising even with the placeholder default
        judge = LocalJudge(model="llama3.1")
        assert judge is not None


class _FakeResponse:
    def __init__(self, content: str | None):
        message = type("Message", (), {"content": content})()
        choice = type("Choice", (), {"message": message})()
        self.choices = [choice]


class TestLocalJudgeCompletion:
    async def test_empty_content_raises_clear_error(self):
        judge = LocalJudge(model="llama3.1")

        async def fake_create(**kwargs):
            return _FakeResponse(content=None)

        judge._client.chat.completions.create = fake_create

        with pytest.raises(ValueError, match="empty response"):
            await judge.complete("system", "user")

    async def test_successful_completion_returns_content(self):
        judge = LocalJudge(model="llama3.1")

        async def fake_create(**kwargs):
            return _FakeResponse(content="the raw response text")

        judge._client.chat.completions.create = fake_create

        result = await judge.complete("system", "user")
        assert result == "the raw response text"

    async def test_full_judge_flow_parses_verdict(self):
        judge = LocalJudge(model="llama3.1")

        async def fake_create(**kwargs):
            return _FakeResponse(content='{"score": 0.9, "reasoning": "Solid answer."}')

        judge._client.chat.completions.create = fake_create

        verdict = await judge.judge("system prompt", "user prompt")
        assert verdict.score == 0.9
        assert verdict.reasoning == "Solid answer."
