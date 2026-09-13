"""Tests for adapters/llamaindex_adapter.py - uses real llama_index.core
base classes so the adapter is verified against the actual interface
it wraps, not a guess at its shape."""

from __future__ import annotations

import pytest

llama_index_core = pytest.importorskip("llama_index.core")

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode

from rag_score.adapters.llamaindex_adapter import (
    LlamaIndexGeneratorAdapter,
    LlamaIndexRetrieverAdapter,
)


class _FakeLIRetriever(BaseRetriever):
    def _retrieve(self, query_bundle):
        return [
            NodeWithScore(node=TextNode(text="Refunds within 30 days.", id_="doc_1"), score=0.95),
            NodeWithScore(node=TextNode(text="Unrelated.", id_="doc_9"), score=0.40),
        ]


class TestLlamaIndexRetrieverAdapter:
    async def test_wraps_real_base_retriever(self):
        adapter = LlamaIndexRetrieverAdapter(_FakeLIRetriever())
        chunks = await adapter.retrieve("What is the refund policy?", top_k=5)
        assert len(chunks) == 2
        assert chunks[0].doc_id == "doc_1"
        assert chunks[0].text == "Refunds within 30 days."
        assert chunks[0].score == 0.95

    async def test_respects_top_k_truncation(self):
        adapter = LlamaIndexRetrieverAdapter(_FakeLIRetriever())
        chunks = await adapter.retrieve("q", top_k=1)
        assert len(chunks) == 1

    def test_rejects_non_retriever(self):
        with pytest.raises(TypeError):
            LlamaIndexRetrieverAdapter("not a retriever")


class TestLlamaIndexGeneratorAdapter:
    async def test_llm_mode(self):
        class FakeLLM:
            async def acomplete(self, prompt):
                return f"Completion for: {prompt[:20]}"

        from rag_score.core.types import RetrievedChunk

        adapter = LlamaIndexGeneratorAdapter(FakeLLM())
        answer = await adapter.generate("q", [RetrievedChunk(doc_id="d", text="context text")])
        assert "Completion for" in answer

    async def test_query_engine_mode(self):
        class FakeQueryEngine:
            async def aquery(self, query):
                return f"Query engine answer to: {query}"

        adapter = LlamaIndexGeneratorAdapter(FakeQueryEngine())
        answer = await adapter.generate("my question", [])
        assert answer == "Query engine answer to: my question"

    def test_rejects_object_with_neither_method(self):
        with pytest.raises(TypeError):
            LlamaIndexGeneratorAdapter(object())

    async def test_query_engine_takes_precedence_when_both_present(self):
        # If something has both aquery and acomplete, query engine mode
        # wins since that's the more complete/typical use case.
        class Both:
            async def aquery(self, query):
                return "from query engine"

            async def acomplete(self, prompt):
                return "from llm"

        adapter = LlamaIndexGeneratorAdapter(Both())
        answer = await adapter.generate("q", [])
        assert answer == "from query engine"
