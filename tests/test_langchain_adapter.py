"""Tests for adapters/langchain_adapter.py - uses real langchain_core
base classes (not mocks of them) so the adapter is verified against
the actual interface it wraps."""

from __future__ import annotations

import pytest

langchain_core = pytest.importorskip("langchain_core")

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda

from rag_score.adapters.langchain_adapter import (
    LangChainGeneratorAdapter,
    LangChainRetrieverAdapter,
    _extract_text,
)


class _FakeLCRetriever(BaseRetriever):
    def _get_relevant_documents(self, query, *, run_manager=None):
        return [
            Document(page_content="Refunds within 30 days.", metadata={"doc_id": "doc_1"}),
            Document(page_content="Unrelated content.", metadata={"doc_id": "doc_9"}),
        ]


class TestLangChainRetrieverAdapter:
    async def test_wraps_real_base_retriever(self):
        adapter = LangChainRetrieverAdapter(_FakeLCRetriever())
        chunks = await adapter.retrieve("What is the refund policy?", top_k=5)
        assert len(chunks) == 2
        assert chunks[0].doc_id == "doc_1"
        assert chunks[0].text == "Refunds within 30 days."

    async def test_respects_top_k_truncation(self):
        adapter = LangChainRetrieverAdapter(_FakeLCRetriever())
        chunks = await adapter.retrieve("q", top_k=1)
        assert len(chunks) == 1

    def test_rejects_non_retriever(self):
        with pytest.raises(TypeError):
            LangChainRetrieverAdapter("not a retriever")

    async def test_falls_back_through_metadata_keys(self):
        class _Retriever(BaseRetriever):
            def _get_relevant_documents(self, query, *, run_manager=None):
                return [Document(page_content="x", metadata={"source": "fallback_id"})]

        adapter = LangChainRetrieverAdapter(_Retriever())
        chunks = await adapter.retrieve("q")
        assert chunks[0].doc_id == "fallback_id"


class TestLangChainGeneratorAdapter:
    async def test_wraps_runnable_returning_string(self):
        runnable = RunnableLambda(lambda inputs: f"Answer using {len(inputs['context'])} chars")
        adapter = LangChainGeneratorAdapter(runnable)
        from rag_score.core.types import RetrievedChunk

        answer = await adapter.generate("q", [RetrievedChunk(doc_id="d", text="some context")])
        assert "Answer using" in answer

    async def test_custom_input_formatter(self):
        received = {}

        def formatter(query, context):
            received["query"] = query
            return {"custom_key": query}

        runnable = RunnableLambda(lambda inputs: inputs["custom_key"])
        adapter = LangChainGeneratorAdapter(runnable, input_formatter=formatter)
        answer = await adapter.generate("my question", [])
        assert answer == "my question"
        assert received["query"] == "my question"

    def test_rejects_non_runnable(self):
        with pytest.raises(TypeError):
            LangChainGeneratorAdapter("not a runnable")


class TestExtractText:
    def test_plain_string(self):
        assert _extract_text("hello") == "hello"

    def test_ai_message(self):
        assert _extract_text(AIMessage(content="hi there")) == "hi there"

    def test_dict_with_answer_key(self):
        assert _extract_text({"answer": "the answer"}) == "the answer"

    def test_dict_with_result_key(self):
        assert _extract_text({"result": "old style"}) == "old style"

    def test_unknown_shape_falls_back_to_str(self):
        assert _extract_text(42) == "42"
