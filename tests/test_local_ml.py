"""Tests for judges/local_ml.py - the embedding infrastructure shared
by local semantic metrics."""

from __future__ import annotations

import sys

import pytest

from rag_score.judges.local_ml import (
    _MODEL_CACHE,
    cosine_similarity,
    load_embedding_model,
)


class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert cosine_similarity([1, 2, 3], [1, 2, 3]) == 1.0

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1, 0], [0, 1]) == 0.0

    def test_opposite_vectors_clamped_to_zero(self):
        assert cosine_similarity([1, 0], [-1, 0]) == 0.0

    def test_scale_invariance(self):
        assert cosine_similarity([1, 2, 3], [2, 4, 6]) == 1.0

    def test_zero_vector_does_not_crash(self):
        assert cosine_similarity([0, 0, 0], [1, 2, 3]) == 0.0
        assert cosine_similarity([1, 2, 3], [0, 0, 0]) == 0.0

    def test_partial_similarity(self):
        import math

        sim = cosine_similarity([1, 1, 0], [1, 0, 0])
        assert sim == pytest.approx(1 / math.sqrt(2))


class TestLoadEmbeddingModel:
    def test_raises_import_error_when_dependency_missing(self):
        real_module = sys.modules.pop("sentence_transformers", None)
        sys.modules["sentence_transformers"] = None
        try:
            with pytest.raises(ImportError, match=r"pip install rag-score\[local-ml\]"):
                load_embedding_model("some-model")
        finally:
            del sys.modules["sentence_transformers"]
            if real_module is not None:
                sys.modules["sentence_transformers"] = real_module

    def test_failed_load_is_not_cached(self):
        # A load that fails (network error, bad model name, or the
        # ImportError case above) must not poison the cache with a
        # broken/missing entry - the next attempt should try again,
        # not silently return something invalid.
        assert "definitely-not-a-cached-model-name" not in _MODEL_CACHE
