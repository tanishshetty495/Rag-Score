"""
Agent adapter interface.

A single method: run(query) -> (tool_calls, final_answer). This is
deliberately one call rather than the retrieve/generate split used
for single-shot RAG, since an agent's tool-calling loop is internal
to the agent itself - the adapter's job is just to capture the trace
of what it did, not to orchestrate the steps.

Same zero-lock-in philosophy as adapters/base.py: implement this
against your actual agent framework (LangGraph, a custom loop,
whatever), and everything downstream only ever talks to this
interface.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod

from rag_score.agentic.types import ToolCall


class AgentAdapter(ABC):
    """Wraps an agent that may call zero or more tools before
    producing a final answer."""

    @abstractmethod
    async def run(self, query: str) -> tuple[list[ToolCall], str]:
        """Execute the agent on the query. Return the full ordered
        list of tool calls it made, plus its final answer."""
        raise NotImplementedError


class CallableAgentAdapter(AgentAdapter):
    """Thin wrapper so users can pass a plain function instead of
    subclassing - same pattern as CallableRetrieverAdapter. Accepts
    either a sync or async callable returning (tool_calls, answer).

    Example:
        async def my_agent(query: str) -> tuple[list[ToolCall], str]:
            ...
        adapter = CallableAgentAdapter(my_agent)
    """

    def __init__(self, fn) -> None:
        self._fn = fn

    async def run(self, query: str) -> tuple[list[ToolCall], str]:
        result = self._fn(query)
        if inspect.isawaitable(result):
            result = await result
        return result
