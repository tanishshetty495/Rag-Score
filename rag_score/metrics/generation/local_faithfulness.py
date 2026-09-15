"""
Local semantic faithfulness - embedding-based approximation of
"is the answer grounded in the retrieved context", with zero API
calls and zero network access after the model's first download.

This is a genuinely different signal than LLM-judge Faithfulness: it
measures semantic similarity between the answer and context, not
whether specific claims are logically entailed. A paraphrase scores
well here even with no LLM judging it; a subtly wrong number embedded
in an otherwise similar-sounding sentence might not be caught, since
cosine similarity doesn't reason about factual correctness the way an
LLM judge's chain of thought can. Use this as a fast, free first pass
or a complement to LLM-judge Faithfulness, not a strict replacement
for it - the two catch different failure modes.
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


class LocalSemanticFaithfulness(Metric):
    name = "local_faithfulness"
    requires_api_key = False

    def __init__(
        self, model_name: str = _DEFAULT_MODEL, encoder: EmbeddingEncoder | None = None
    ) -> None:
        """
        model_name: a sentence-transformers model name/path. Ignored
                    if `encoder` is provided.
        encoder: inject a pre-built encoder directly - mainly for
                 testing without a network call/model download, but
                 also useful if you're already loading this model
                 elsewhere and want to share the instance.
        """
        self._model_name = model_name
        self._injected_encoder = encoder

    def _get_encoder(self) -> EmbeddingEncoder:
        if self._injected_encoder is not None:
            return self._injected_encoder
        return load_embedding_model(self._model_name)

    async def score(self, test_case: TestCase, result: EvalResult) -> float:
        if not result.generated_answer or not result.retrieved_context:
            return 0.0

        context_text = " ".join(chunk.text for chunk in result.retrieved_context)
        encoder = self._get_encoder()
        embeddings = await encode_async(encoder, [result.generated_answer, context_text])
        return cosine_similarity(embeddings[0], embeddings[1])
