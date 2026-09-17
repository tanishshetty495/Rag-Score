"""Tests for agentic/adapters.py."""

from __future__ import annotations

from rag_score.agentic.adapters import CallableAgentAdapter
from rag_score.agentic.types import ToolCall


class TestCallableAgentAdapter:
    async def test_wraps_async_function(self):
        async def my_agent(query: str):
            return [ToolCall(tool_name="search")], f"answer to {query}"

        adapter = CallableAgentAdapter(my_agent)
        tool_calls, answer = await adapter.run("what is x?")
        assert tool_calls == [ToolCall(tool_name="search")]
        assert answer == "answer to what is x?"

    async def test_wraps_sync_function(self):
        def my_agent(query: str):
            return [], "sync answer"

        adapter = CallableAgentAdapter(my_agent)
        tool_calls, answer = await adapter.run("q")
        assert tool_calls == []
        assert answer == "sync answer"

    async def test_passes_query_through(self):
        received = {}

        async def my_agent(query: str):
            received["query"] = query
            return [], "x"

        adapter = CallableAgentAdapter(my_agent)
        await adapter.run("the exact query text")
        assert received["query"] == "the exact query text"
