"""Tests for judges/openai_judge.py and judges/anthropic_judge.py.

No real API calls are made - these verify construction (correct
defaults, custom model/api_key handling) and the complete() ->
response parsing logic by monkeypatching the underlying SDK client
call, the same pattern used for LocalJudge."""

from __future__ import annotations

import pytest

openai = pytest.importorskip("openai")
anthropic = pytest.importorskip("anthropic")

from rag_score.judges.anthropic_judge import AnthropicJudge
from rag_score.judges.openai_judge import OpenAIJudge


class TestOpenAIJudgeConstruction:
    def test_default_model(self):
        judge = OpenAIJudge(api_key="sk-fake-test-key")
        assert judge.model == "gpt-4o-mini"
        assert judge.temperature == 0.0

    def test_custom_model_and_temperature(self):
        judge = OpenAIJudge(model="gpt-4o", temperature=0.5, api_key="sk-fake-test-key")
        assert judge.model == "gpt-4o"
        assert judge.temperature == 0.5

    def test_explicit_api_key_accepted(self):
        # Shouldn't raise - just confirms the client constructs with an
        # explicit key rather than only relying on the env var.
        judge = OpenAIJudge(api_key="sk-fake-test-key")
        assert judge is not None


class _FakeOpenAIResponse:
    def __init__(self, content: str | None):
        message = type("Message", (), {"content": content})()
        choice = type("Choice", (), {"message": message})()
        self.choices = [choice]


class TestOpenAIJudgeCompletion:
    async def test_successful_completion(self):
        judge = OpenAIJudge(api_key="sk-fake-test-key")

        async def fake_create(**kwargs):
            return _FakeOpenAIResponse(content="the response")

        judge._client.chat.completions.create = fake_create
        result = await judge.complete("system", "user")
        assert result == "the response"

    async def test_empty_content_raises(self):
        judge = OpenAIJudge(api_key="sk-fake-test-key")

        async def fake_create(**kwargs):
            return _FakeOpenAIResponse(content=None)

        judge._client.chat.completions.create = fake_create
        with pytest.raises(ValueError, match="empty"):
            await judge.complete("system", "user")

    async def test_full_judge_flow(self):
        judge = OpenAIJudge(api_key="sk-fake-test-key")

        async def fake_create(**kwargs):
            return _FakeOpenAIResponse(content='{"score": 0.7, "reasoning": "decent"}')

        judge._client.chat.completions.create = fake_create
        verdict = await judge.judge("system", "user")
        assert verdict.score == 0.7
        assert verdict.reasoning == "decent"


class TestAnthropicJudgeConstruction:
    def test_default_model(self):
        judge = AnthropicJudge()
        assert judge.model == "claude-haiku-4-5"
        assert judge.max_tokens == 512

    def test_custom_model_and_max_tokens(self):
        judge = AnthropicJudge(model="claude-opus-5", max_tokens=1024)
        assert judge.model == "claude-opus-5"
        assert judge.max_tokens == 1024

    def test_explicit_api_key_accepted(self):
        judge = AnthropicJudge(api_key="sk-ant-fake-test-key")
        assert judge is not None


class _FakeAnthropicTextBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class _FakeAnthropicResponse:
    def __init__(self, content: list):
        self.content = content


class TestAnthropicJudgeCompletion:
    async def test_successful_completion(self):
        judge = AnthropicJudge(api_key="sk-ant-fake-test-key")

        async def fake_create(**kwargs):
            return _FakeAnthropicResponse(content=[_FakeAnthropicTextBlock("the response")])

        judge._client.messages.create = fake_create
        result = await judge.complete("system", "user")
        assert result == "the response"

    async def test_multiple_text_blocks_are_joined(self):
        judge = AnthropicJudge(api_key="sk-ant-fake-test-key")

        async def fake_create(**kwargs):
            return _FakeAnthropicResponse(
                content=[_FakeAnthropicTextBlock("part one "), _FakeAnthropicTextBlock("part two")]
            )

        judge._client.messages.create = fake_create
        result = await judge.complete("system", "user")
        assert result == "part one part two"

    async def test_no_text_blocks_raises(self):
        judge = AnthropicJudge(api_key="sk-ant-fake-test-key")

        async def fake_create(**kwargs):
            return _FakeAnthropicResponse(content=[])

        judge._client.messages.create = fake_create
        with pytest.raises(ValueError, match="no text"):
            await judge.complete("system", "user")

    async def test_full_judge_flow(self):
        judge = AnthropicJudge(api_key="sk-ant-fake-test-key")

        async def fake_create(**kwargs):
            return _FakeAnthropicResponse(
                content=[_FakeAnthropicTextBlock('{"score": 0.6, "reasoning": "okay"}')]
            )

        judge._client.messages.create = fake_create
        verdict = await judge.judge("system", "user")
        assert verdict.score == 0.6
        assert verdict.reasoning == "okay"
