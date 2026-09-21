"""
Telemetry: token counting and cost estimation for evaluation runs.

Token counts are estimated from the actual prompt/completion TEXT via
tiktoken, not pulled from provider-reported usage - that's a
deliberate choice, not a shortcut. Making this accurate would mean
changing GeneratorAdapter.generate() to return a richer object
(answer + usage metadata) instead of a plain string, which would
break every existing adapter (Callable/LangChain/LlamaIndex) and every
example/test written against the current interface. Text-based
estimation costs nothing in compatibility and is accurate enough for
tracking relative cost/usage trends across a run.

tiktoken itself downloads its BPE encoding file from a CDN
(openaipublic.blob.core.windows.net) on first use per model - a
genuine problem for exactly the kind of offline/corporate-firewall
environments this package's local-first philosophy targets (see
judges/local_judge.py, judges/local_ml.py). count_tokens() therefore
falls back to a dependency-free word-based estimate whenever tiktoken
can't be used, rather than letting telemetry collection break an
otherwise-successful evaluation run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Prices are USD per 1,000 tokens: (prompt_price, completion_price).
# Deliberately small and easy to override - this is a starting point,
# not an attempt to track every provider's pricing in perpetuity.
DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.0025, 0.01),
    "claude-haiku-4-5": (0.001, 0.005),
    "claude-opus-5": (0.015, 0.075),
    "claude-sonnet-5": (0.003, 0.015),
}

# tiktoken model names occasionally differ from a provider's public
# model name (e.g. Anthropic models aren't in tiktoken's registry at
# all) - map to the closest tiktoken-known encoding so estimation
# still works, rather than failing for every non-OpenAI model.
_TIKTOKEN_MODEL_FALLBACK = "gpt-4o-mini"

# Rough words-to-tokens ratio used only when tiktoken is unavailable
# or its encoding file can't be downloaded. English text averages
# ~0.75 words per token (tokens are sub-word), so tokens ~= words / 0.75.
_FALLBACK_WORDS_TO_TOKENS_RATIO = 1 / 0.75


@dataclass
class TelemetryConfig:
    """Enables token/cost tracking on a run. Passing this to RunConfig
    turns telemetry on; omitting it (the default) keeps evaluation
    runs exactly as they were before this feature existed - zero
    overhead, zero behavior change, for anyone who doesn't opt in."""

    model_name: str = "gpt-4o-mini"
    pricing: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_PRICING))


def count_tokens(text: str, model_name: str) -> int:
    """Estimate the number of tokens in `text` for `model_name`.

    Tries tiktoken first; falls back to a word-count-based estimate if
    tiktoken isn't installed, doesn't recognize the model, or can't
    reach its encoding CDN (common in offline/firewalled environments -
    this is the expected, non-error path there, not a bug).
    """
    if not text:
        return 0

    try:
        import tiktoken

        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Model not in tiktoken's registry (e.g. any Anthropic
            # model) - use a known-good encoding as a stand-in. Token
            # counts across model families are close enough for cost
            # *estimation* purposes even though they're not identical.
            encoding = tiktoken.encoding_for_model(_TIKTOKEN_MODEL_FALLBACK)
        return len(encoding.encode(text))
    except Exception:  # noqa: BLE001 - any tiktoken failure (missing package,
        # network-blocked CDN download, etc.) falls back rather than
        # breaking telemetry collection for an otherwise-successful run.
        return max(1, round(len(text.split()) * _FALLBACK_WORDS_TO_TOKENS_RATIO))


def estimate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    model_name: str,
    pricing: dict[str, tuple[float, float]] | None = None,
) -> float | None:
    """Estimate cost in USD. Returns None (not 0.0) when the model
    isn't in the pricing table - silently reporting $0.00 for an
    unpriced model would look like a real "this is free" answer rather
    than "we don't know", which is a meaningfully different claim."""
    table = pricing if pricing is not None else DEFAULT_PRICING
    if model_name not in table:
        return None

    prompt_price, completion_price = table[model_name]
    return (prompt_tokens / 1000) * prompt_price + (completion_tokens / 1000) * completion_price
