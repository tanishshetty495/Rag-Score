"""
Local ML embedding infrastructure - shared by the local semantic
metrics (faithfulness, answer relevance).

This is deliberately NOT an LLMJudge subclass. LLMJudge's contract is
"send a prompt, parse a score+reasoning back out of the response" -
that's the wrong shape for an embedding model, which computes a
similarity score directly from two texts with no prompt involved.
Forcing embeddings through the judge() interface would mean packing
raw text into a "prompt" just to unpack it again on the other side -
more indirection for no benefit. So local semantic metrics implement
Metric directly and use this module's helpers, rather than going
through judges/base.py at all.

sentence-transformers is an optional dependency (pip install
rag-score[local-ml]) - only imported when a model is actually loaded,
consistent with every other optional integration in this package.
"""

from __future__ import annotations

import asyncio
import math
from typing import Protocol


class EmbeddingEncoder(Protocol):
    """The only method local metrics actually need from an encoder -
    matches sentence_transformers.SentenceTransformer.encode()'s
    signature closely enough that a real SentenceTransformer instance
    satisfies this protocol with no wrapping required. Also makes it
    trivial to inject a fake encoder in tests with zero network access
    or model download."""

    def encode(self, texts: list[str]) -> list[list[float]]: ...


# Loaded models are cached by name at module level - each metric
# instance doesn't need its own copy, and loading a sentence-transformer
# model is slow enough (disk read + GPU/CPU placement) that reusing one
# across every metric/test-case in a run matters for real usage.
_MODEL_CACHE: dict[str, EmbeddingEncoder] = {}


def load_embedding_model(model_name: str) -> EmbeddingEncoder:
    """Load (or return the cached) sentence-transformers model by name.
    Raises a clear ImportError if the optional dependency isn't
    installed, matching the pattern every other optional integration
    in this package follows."""
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required for local ML-based metrics. "
            "Install it with: pip install rag-score[local-ml]"
        ) from e

    model = SentenceTransformer(model_name)
    _MODEL_CACHE[model_name] = model
    return model


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Plain cosine similarity, clamped to [0.0, 1.0] - negative
    similarity (semantically opposed texts) is rare in practice for
    faithfulness/relevance comparisons and isn't meaningful as a
    "how good is this" score below zero, so it's floored rather than
    passed through as a negative number that would look like a bug
    everywhere else in this package's 0-1 score convention."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    similarity = dot / (norm_a * norm_b)
    return max(0.0, min(1.0, similarity))


async def encode_async(encoder: EmbeddingEncoder, texts: list[str]) -> list[list[float]]:
    """Run the (blocking, CPU/GPU-bound) encode() call in a thread so
    it doesn't stall the event loop - the runner fans out many
    test cases concurrently via asyncio.gather, and a synchronous
    encode() call would serialize all of them behind the GIL/blocking
    I/O instead of actually overlapping."""
    return await asyncio.to_thread(encoder.encode, texts)
