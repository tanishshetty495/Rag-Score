"""
LlamaIndex adapter.

Wraps a LlamaIndex BaseRetriever (via .aretrieve) and a query engine /
LLM-like object (via .aquery or .acomplete) so an existing LlamaIndex
RAG pipeline can be evaluated with zero rewriting.

llama-index-core is an optional dependency (pip install
rag-score[llamaindex]) - only imported when these classes are
actually instantiated.
"""

from __future__ import annotations

from typing import Any

from rag_score.adapters.base import GeneratorAdapter, RetrieverAdapter
from rag_score.core.types import RetrievedChunk


class LlamaIndexRetrieverAdapter(RetrieverAdapter):
    """Wraps a llama_index.core.base.base_retriever.BaseRetriever
    (what VectorStoreIndex.as_retriever() returns).

    Example:
        retriever = my_index.as_retriever(similarity_top_k=10)
        adapter = LlamaIndexRetrieverAdapter(retriever)
    """

    def __init__(self, retriever: Any) -> None:
        try:
            from llama_index.core.base.base_retriever import BaseRetriever
        except ImportError as e:
            raise ImportError(
                "llama-index-core is required for LlamaIndexRetrieverAdapter. "
                "Install it with: pip install rag-score[llamaindex]"
            ) from e

        if not isinstance(retriever, BaseRetriever):
            raise TypeError(f"Expected a llama_index BaseRetriever, got {type(retriever).__name__}")
        self._retriever = retriever

    async def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        # Most LlamaIndex retrievers apply their own top_k via
        # similarity_top_k set at construction time; we still truncate
        # here as a safety net in case the retriever returns more.
        nodes_with_scores = await self._retriever.aretrieve(query)
        chunks = [
            RetrievedChunk(
                doc_id=nws.node.node_id,
                text=nws.node.get_content(),
                score=nws.score,
            )
            for nws in nodes_with_scores
        ]
        return chunks[:top_k]


class LlamaIndexGeneratorAdapter(GeneratorAdapter):
    """Wraps a LlamaIndex query engine (has .aquery) or a bare LLM
    (has .acomplete). Query engines do their own retrieval internally,
    so when pairing this with LlamaIndexRetrieverAdapter in the same
    eval run, retrieval effectively happens twice - that's expected
    and lets each adapter be scored/timed independently.
    """

    def __init__(self, engine_or_llm: Any) -> None:
        self._target = engine_or_llm
        self._mode = self._detect_mode(engine_or_llm)

    @staticmethod
    def _detect_mode(obj: Any) -> str:
        if hasattr(obj, "aquery"):
            return "query_engine"
        if hasattr(obj, "acomplete"):
            return "llm"
        raise TypeError(
            f"{type(obj).__name__} has neither .aquery nor .acomplete - "
            "pass a LlamaIndex query engine or LLM."
        )

    async def generate(self, query: str, context: list[RetrievedChunk]) -> str:
        if self._mode == "query_engine":
            # Query engines retrieve internally, so we pass the
            # question straight through and ignore the runner's
            # separately-retrieved context.
            response = await self._target.aquery(query)
            return str(response)

        # Bare LLM: build the prompt manually from the runner's context.
        context_text = "\n\n".join(chunk.text for chunk in context)
        prompt = f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer:"
        completion = await self._target.acomplete(prompt)
        return str(completion)
