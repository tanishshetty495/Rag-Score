"""
Regression/golden tests for framework adapters:
- Callable adapters
- LangChain adapter (mocked)
- LlamaIndex adapter (mocked)
"""

from __future__ import annotations

import asyncio
import pytest

from rag_score.adapters.base import (
    CallableGeneratorAdapter,
    CallableRetrieverAdapter,
    GeneratorAdapter,
    RetrieverAdapter,
)
from rag_score.core.types import RetrievedChunk


class TestCallableRetrieverAdapterRegression:
    """Tests for CallableRetrieverAdapter."""

    @pytest.mark.asyncio
    async def test_async_function(self):
        """Async function is awaited correctly."""

        async def async_retrieve(query: str, top_k: int = 5):
            await asyncio.sleep(0.01)
            return [RetrievedChunk(doc_id="d1", text="result")]

        adapter = CallableRetrieverAdapter(async_retrieve)
        result = await adapter.retrieve("test", top_k=3)
        assert len(result) == 1
        assert result[0].doc_id == "d1"

    @pytest.mark.asyncio
    async def test_sync_function(self):
        """Sync function is called correctly."""

        def sync_retrieve(query: str, top_k: int = 5):
            return [RetrievedChunk(doc_id="d1", text="result")]

        adapter = CallableRetrieverAdapter(sync_retrieve)
        result = await adapter.retrieve("test", top_k=3)
        assert len(result) == 1
        assert result[0].doc_id == "d1"

    @pytest.mark.asyncio
    async def test_top_k_passed_through(self):
        """top_k parameter is passed to the function."""
        received_k = None

        def retrieve(query: str, top_k: int = 5):
            nonlocal received_k
            received_k = top_k
            return [RetrievedChunk(doc_id="d1", text="result")]

        adapter = CallableRetrieverAdapter(retrieve)
        await adapter.retrieve("test", top_k=7)
        assert received_k == 7

    @pytest.mark.asyncio
    async def test_default_top_k(self):
        """Default top_k=5 is used when not specified."""
        received_k = None

        def retrieve(query: str, top_k: int = 5):
            nonlocal received_k
            received_k = top_k
            return [RetrievedChunk(doc_id="d1", text="result")]

        adapter = CallableRetrieverAdapter(retrieve)
        await adapter.retrieve("test")
        assert received_k == 5


class TestCallableGeneratorAdapterRegression:
    """Tests for CallableGeneratorAdapter."""

    @pytest.mark.asyncio
    async def test_async_function(self):
        """Async function is awaited correctly."""

        async def async_generate(query: str, context):
            await asyncio.sleep(0.01)
            return "async answer"

        adapter = CallableGeneratorAdapter(async_generate)
        result = await adapter.generate("test", [RetrievedChunk(doc_id="d1", text="ctx")])
        assert result == "async answer"

    @pytest.mark.asyncio
    async def test_sync_function(self):
        """Sync function is called correctly."""

        def sync_generate(query: str, context):
            return "sync answer"

        adapter = CallableGeneratorAdapter(sync_generate)
        result = await adapter.generate("test", [RetrievedChunk(doc_id="d1", text="ctx")])
        assert result == "sync answer"

    @pytest.mark.asyncio
    async def test_context_passed_correctly(self):
        """Context chunks are passed to the function."""
        received_context = None

        def generate(query: str, context):
            nonlocal received_context
            received_context = context
            return "answer"

        adapter = CallableGeneratorAdapter(generate)
        ctx = [RetrievedChunk(doc_id="d1", text="chunk1"), RetrievedChunk(doc_id="d2", text="chunk2")]
        await adapter.generate("test", ctx)
        assert received_context == ctx
        assert len(received_context) == 2

    @pytest.mark.asyncio
    async def test_history_passed_when_function_accepts_it(self):
        """History parameter is passed when the wrapped function accepts it."""

        def generate_with_history(query: str, context, history=None):
            return f"answer with history: {history}"

        adapter = CallableGeneratorAdapter(generate_with_history)
        result = await adapter.generate("test", [RetrievedChunk(doc_id="d1", text="ctx")], history=[{"role": "user", "content": "hi"}])
        assert "history" in result

    @pytest.mark.asyncio
    async def test_history_not_passed_when_function_rejects_it(self):
        """History parameter is not passed when the wrapped function doesn't accept it."""

        def generate_no_history(query: str, context):
            return "answer"

        adapter = CallableGeneratorAdapter(generate_no_history)
        # Should not raise even though history is provided
        result = await adapter.generate("test", [RetrievedChunk(doc_id="d1", text="ctx")], history=[{"role": "user", "content": "hi"}])
        assert result == "answer"


class TestRetrieverAdapterInterface:
    """Tests that RetrieverAdapter interface is properly defined."""

    def test_abstract_methods(self):
        """RetrieverAdapter has retrieve as abstract method."""
        assert hasattr(RetrieverAdapter, "retrieve")
        assert RetrieverAdapter.retrieve.__isabstractmethod__ is True

    def test_cannot_instantiate_directly(self):
        """Cannot instantiate RetrieverAdapter directly."""
        with pytest.raises(TypeError):
            RetrieverAdapter()


class TestGeneratorAdapterInterface:
    """Tests that GeneratorAdapter interface is properly defined."""

    def test_abstract_methods(self):
        """GeneratorAdapter has generate as abstract method."""
        assert hasattr(GeneratorAdapter, "generate")
        assert GeneratorAdapter.generate.__isabstractmethod__ is True

    def test_cannot_instantiate_directly(self):
        """Cannot instantiate GeneratorAdapter directly."""
        with pytest.raises(TypeError):
            GeneratorAdapter()


# Mock LangChain/LlamaIndex adapter tests (using callables to simulate)
class TestMockLangChainAdapter:
    """Tests simulating LangChain adapter behavior."""

    @pytest.mark.asyncio
    async def test_langchain_retriever_interface(self):
        """LangChain-style retriever can be wrapped."""

        class MockLangChainRetriever:
            async def ainvoke(self, query: str, config=None):
                return [{"page_content": "chunk1", "metadata": {"doc_id": "d1"}}]

        retriever = MockLangChainRetriever()

        async def wrapped_retrieve(query: str, top_k: int = 5):
            docs = await retriever.ainvoke(query)
            return [
                RetrievedChunk(doc_id=doc["metadata"].get("doc_id"), text=doc["page_content"])
                for doc in docs
            ][:top_k]

        adapter = CallableRetrieverAdapter(wrapped_retrieve)
        result = await adapter.retrieve("test", top_k=3)
        assert len(result) == 1
        assert result[0].doc_id == "d1"
        assert result[0].text == "chunk1"

    @pytest.mark.asyncio
    async def test_langchain_chain_interface(self):
        """LangChain-style chain can be wrapped."""

        class MockLangChainChain:
            async def ainvoke(self, input_dict: dict):
                return {"text": "generated answer"}

        chain = MockLangChainChain()

        async def wrapped_generate(query: str, context):
            result = await chain.ainvoke({"question": query, "context": context})
            return result["text"]

        adapter = CallableGeneratorAdapter(wrapped_generate)
        result = await adapter.generate("test", [RetrievedChunk(doc_id="d1", text="ctx")])
        assert result == "generated answer"


class TestMockLlamaIndexAdapter:
    """Tests simulating LlamaIndex adapter behavior."""

    @pytest.mark.asyncio
    async def test_llamaindex_retriever_interface(self):
        """LlamaIndex-style retriever can be wrapped."""

        class MockLlamaIndexRetriever:
            async def aretrieve(self, query: str):
                from llama_index.core.schema import NodeWithScore, TextNode
                return [
                    NodeWithScore(
                        node=TextNode(text="chunk1", id_="d1"),
                        score=0.9
                    )
                ]

        retriever = MockLlamaIndexRetriever()

        async def wrapped_retrieve(query: str, top_k: int = 5):
            nodes = await retriever.aretrieve(query)
            return [
                RetrievedChunk(doc_id=node.node.id_, text=node.node.text)
                for node in nodes
            ][:top_k]

        adapter = CallableRetrieverAdapter(wrapped_retrieve)
        result = await adapter.retrieve("test", top_k=3)
        assert len(result) == 1
        assert result[0].doc_id == "d1"
        assert result[0].text == "chunk1"

    @pytest.mark.asyncio
    async def test_llamaindex_query_engine_interface(self):
        """LlamaIndex-style query engine can be wrapped."""

        class MockLlamaIndexQueryEngine:
            async def aquery(self, query: str):
                return "generated answer"

        engine = MockLlamaIndexQueryEngine()

        async def wrapped_generate(query: str, context):
            return await engine.aquery(query)

        adapter = CallableGeneratorAdapter(wrapped_generate)
        result = await adapter.generate("test", [RetrievedChunk(doc_id="d1", text="ctx")])
        assert result == "generated answer"