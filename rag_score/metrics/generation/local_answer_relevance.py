"""
Local semantic answer relevance - embedding-based approximation of
"does the answer address the question", with zero API calls.

Same trade-off as LocalSemanticFaithfulness: this catches "the answer
is topically about something completely different from the question"
cheaply and for free, but won't catch subtler failures (a confident,
on-topic non-answer, e.g. "That's a great question!" with no actual
content) the way an LLM judge reading the text for real would.
"""

from __future__ import annotations

from rag_score.core.types import EvalResult, TestCase
from rag_score.judges.local_ml import (
    EmbeddingEncoder,
    cosine_similarity,
    encode_async,
    load_embedding_model,
)
from rag_score.metrics.base import Metric

_DEFAULT_MODEL = "all-MiniLM-L6-v2"


class LocalSemanticAnswerRelevance(Metric):
    name = "local_answer_relevance"
    requires_api_key = False

    def __init__(
        self, model_name: str = _DEFAULT_MODEL, encoder: EmbeddingEncoder | None = None
    ) -> None:
        self._model_name = model_name
        self._injected_encoder = encoder

    def _get_encoder(self) -> EmbeddingEncoder:
        if self._injected_encoder is not None:
            return self._injected_encoder
        return load_embedding_model(self._model_name)

    async def score(self, test_case: TestCase, result: EvalResult) -> float:
        if not result.generated_answer:
            return 0.0

        encoder = self._get_encoder()
        embeddings = await encode_async(encoder, [test_case.question, result.generated_answer])
        return cosine_similarity(embeddings[0], embeddings[1])
