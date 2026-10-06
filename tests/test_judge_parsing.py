"""Tests for judges/base.py - JudgeVerdict parsing from raw LLM text."""

from __future__ import annotations

import pytest

from rag_score.judges.base import _parse_verdict


class TestParseVerdict:
    def test_clean_json(self):
        v = _parse_verdict('{"score": 0.7, "reasoning": "ok"}')
        assert v.score == 0.7
        assert v.reasoning == "ok"

    def test_markdown_fenced_json(self):
        v = _parse_verdict('```json\n{"score": 0.9, "reasoning": "great"}\n```')
        assert v.score == 0.9
        assert v.reasoning == "great"

    def test_fenced_without_json_language_tag(self):
        v = _parse_verdict('```\n{"score": 0.6, "reasoning": "fine"}\n```')
        assert v.score == 0.6

    def test_prose_wrapped_json(self):
        v = _parse_verdict(
            'Here is my verdict: {"score": 0.4, "reasoning": "meh"} Hope that helps!'
        )
        assert v.score == 0.4
        assert v.reasoning == "meh"

    def test_score_out_of_ten_is_rescaled(self):
        v = _parse_verdict('{"score": 8, "reasoning": "good"}')
        assert v.score == pytest.approx(0.8)

    def test_score_out_of_hundred_is_rescaled(self):
        v = _parse_verdict('{"score": 85, "reasoning": "good"}')
        assert v.score == pytest.approx(0.85)

    def test_score_clamped_to_valid_range(self):
        # Already <= 1.0 but let's confirm boundary values pass through untouched
        v = _parse_verdict('{"score": 1.0, "reasoning": "perfect"}')
        assert v.score == 1.0
        v = _parse_verdict('{"score": 0.0, "reasoning": "terrible"}')
        assert v.score == 0.0

    def test_missing_score_field_raises(self):
        with pytest.raises(ValueError, match="score"):
            _parse_verdict('{"reasoning": "no score given"}')

    def test_missing_reasoning_field_raises(self):
        with pytest.raises(ValueError, match="reasoning"):
            _parse_verdict('{"score": 0.5}')

    def test_non_json_garbage_raises(self):
        with pytest.raises(ValueError):
            _parse_verdict("I refuse to answer in JSON format.")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            _parse_verdict("")
