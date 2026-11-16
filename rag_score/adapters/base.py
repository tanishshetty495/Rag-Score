"""
Adapter interfaces — the "zero lock-in" boundary.

Users implement one small async method to plug their existing retriever
or generator (LangChain, LlamaIndex, raw FAISS, an HTTP call, whatever)
into the evaluation runner. Everything downstream only ever talks to
these interfaces, never to a specific framework.

Async is required (not optional) because the runner fans out many
TestCases concurrently, gated by a semaphore, to keep multi-hundred-row
evaluation runs from taking hours against a rate-limited API.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag_score.core.types import RetrievedChunk


class RetrieverAdapter(ABC):
    """Wraps any retrieval system. Implementers just need to return
    plain RetrievedChunk objects — doc_id is optional but strongly
    recommended, since it's required for precision/recall/MRR/nDCG."""

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Return up to top_k retrieved chunks for the given query."""
        raise NotImplementedError


class GeneratorAdapter(ABC):
    """Wraps any generation system (a raw LLM call, a LangChain chain,
    an agent, etc.). Only needs the final answer string."""

    @abstractmethod
    async def generate(self, query: str, context: list[RetrievedChunk]) -> str:
        """Return the generated answer for the query given the retrieved context."""
        raise NotImplementedError


class CallableRetrieverAdapter(RetrieverAdapter):
    """Thin wrapper so users can pass a plain function instead of
    subclassing. Accepts either a sync or async callable.

    Example:
        async def my_retrieve(query: str, top_k: int) -> list[RetrievedChunk]:
            ...
        adapter = CallableRetrieverAdapter(my_retrieve)
    """

    def __init__(self, fn) -> None:
        self._fn = fn

    async def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        import inspect

        result = self._fn(query, top_k)
        if inspect.isawaitable(result):
            result = await result
        return result


class CallableGeneratorAdapter(GeneratorAdapter):
    """Same idea as CallableRetrieverAdapter, for generators."""

    def __init__(self, fn) -> None:
        self._fn = fn
        # Check if the wrapped function accepts a 'history' parameter
        import inspect
        self._accepts_history = "history" in inspect.signature(fn).parameters

    async def generate(self, query: str, context: list[RetrievedChunk], history=None) -> str:
        import inspect

        # Pass history to the wrapped function if it accepts it
        if self._accepts_history and history is not None:
            result = self._fn(query, context, history=history)
        else:
            result = self._fn(query, context)
        if inspect.isawaitable(result):
            result = await result
        return result
