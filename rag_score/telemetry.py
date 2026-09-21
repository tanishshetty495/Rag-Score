"""Tests for telemetry.py - token counting and cost estimation.

The fallback token-count path (word-based estimate) is tested for
real here, since this sandbox genuinely can't reach tiktoken's
encoding CDN - that's not a mock, it's the actual code path most
offline/firewalled users will also hit. The tiktoken-success path is
tested via mocking since exercising the real thing needs network
access this environment doesn't have.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from rag_score.telemetry import (
    DEFAULT_PRICING,
    TelemetryConfig,
    count_tokens,
    estimate_cost,
)


class TestCountTokens:
    def test_empty_text_returns_zero(self):
        assert count_tokens("", "gpt-4o-mini") == 0

    def test_fallback_path_returns_positive_count(self):
        # Genuinely exercises the fallback (no network to tiktoken's
        # CDN in this environment) rather than mocking it.
        tokens = count_tokens("This is a test sentence with several words.", "gpt-4o-mini")
        assert tokens > 0

    def test_fallback_scales_with_text_length(self):
        short = count_tokens("hello", "gpt-4o-mini")
        long = count_tokens("hello " * 100, "gpt-4o-mini")
        assert long > short

    def test_mocked_tiktoken_success_path(self):
        fake_encoding = MagicMock()
        fake_encoding.encode.return_value = [1, 2, 3, 4, 5]
        with patch("tiktoken.encoding_for_model", return_value=fake_encoding):
            assert count_tokens("some text", "gpt-4o-mini") == 5

    def test_unknown_model_falls_back_to_known_encoding(self):
        """Anthropic models aren't in tiktoken's registry at all -
        confirm the KeyError path falls back rather than raising."""
        fake_encoding = MagicMock()
        fake_encoding.encode.return_value = [1, 2, 3]

        def fake_lookup(model):
            if model == "claude-haiku-4-5":
                raise KeyError("not found")
            return fake_encoding

        with patch("tiktoken.encoding_for_model", side_effect=fake_lookup):
            assert count_tokens("text", "claude-haiku-4-5") == 3

    def test_missing_tiktoken_package_falls_back(self):
        real_module = sys.modules.pop("tiktoken", None)
        sys.modules["tiktoken"] = None
        try:
            tokens = count_tokens("some reasonably long test sentence here", "gpt-4o-mini")
            assert tokens > 0
        finally:
            del sys.modules["tiktoken"]
            if real_module is not None:
                sys.modules["tiktoken"] = real_module


class TestEstimateCost:
    def test_known_model_returns_correct_cost(self):
        cost = estimate_cost(1000, 500, "gpt-4o-mini")
        expected = (1000 / 1000) * 0.00015 + (500 / 1000) * 0.0006
        assert cost == pytest.approx(expected)

    def test_unknown_model_returns_none_not_zero(self):
        """A missing price must not silently look like a real $0.00 -
        that's a meaningfully different claim from 'we don't know'."""
        assert estimate_cost(1000, 500, "totally-unpriced-model") is None

    def test_zero_tokens_returns_zero_cost(self):
        assert estimate_cost(0, 0, "gpt-4o-mini") == 0.0

    def test_custom_pricing_table_overrides_default(self):
        custom = {"my-model": (0.01, 0.02)}
        cost = estimate_cost(1000, 1000, "my-model", pricing=custom)
        assert cost == pytest.approx(0.03)

    def test_custom_pricing_table_does_not_fall_back_to_default(self):
        # A model priced in DEFAULT_PRICING but not in a custom table
        # passed explicitly should still return None, not silently
        # use the default table behind the caller's back.
        custom = {"only-this-model": (0.01, 0.02)}
        assert estimate_cost(1000, 500, "gpt-4o-mini", pricing=custom) is None


class TestTelemetryConfig:
    def test_defaults(self):
        config = TelemetryConfig()
        assert config.model_name == "gpt-4o-mini"
        assert config.pricing == DEFAULT_PRICING

    def test_pricing_dicts_are_independent_across_instances(self):
        """Regression test for the classic mutable-default-dataclass
        bug - each TelemetryConfig() must get its own pricing dict,
        not a shared reference to one module-level dict."""
        config_a = TelemetryConfig()
        config_b = TelemetryConfig()
        config_a.pricing["new-model"] = (1.0, 2.0)
        assert "new-model" not in config_b.pricing
        assert "new-model" not in DEFAULT_PRICING

    def test_custom_model_and_pricing(self):
        config = TelemetryConfig(model_name="my-model", pricing={"my-model": (0.1, 0.2)})
        assert config.model_name == "my-model"
        assert config.pricing == {"my-model": (0.1, 0.2)}
