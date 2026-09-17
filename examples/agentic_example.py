"""
Example: evaluating an agent's tool-calling trajectory.

Unlike the RAG examples (raw_example.py, langchain_example.py,
llamaindex_example.py), this evaluates a multi-step agent - one that
decides which tools to call, in what order, before producing a final
answer - rather than a single retrieve-then-generate pass.

The demo agent below is a simple rule-based stand-in; in a real
project, swap it for your actual agent framework's entrypoint
(LangGraph, a custom ReAct loop, whatever) - the only contract is:
take a query, return (list[ToolCall], final_answer_string).

Run it:
    python examples/agentic_example.py
"""

from __future__ import annotations

import asyncio

from rag_score.agentic.adapters import CallableAgentAdapter
from rag_score.agentic.metrics.tool_call_order import ToolCallOrderCorrectness
from rag_score.agentic.metrics.tool_selection_precision import ToolSelectionPrecision
from rag_score.agentic.metrics.tool_selection_recall import ToolSelectionRecall
from rag_score.agentic.runner import TrajectoryRunConfig, run_trajectory_evaluation
from rag_score.agentic.types import ToolCall, TrajectoryTestCase


async def my_agent(query: str) -> tuple[list[ToolCall], str]:
    """Stands in for a real agent - swap this for your actual
    LangGraph/custom-loop agent entrypoint in a real project. Decides
    which tools to call based on keywords in the query (a real agent
    would use an LLM to decide this instead)."""
    query_lower = query.lower()
    tool_calls: list[ToolCall] = []

    if "weather" in query_lower:
        tool_calls.append(
            ToolCall(tool_name="search_weather", tool_input={"city": "Paris"}, tool_output="Sunny, 22C")
        )
    if any(op in query_lower for op in ("+", "-", "*", "times", "percent", "%")):
        tool_calls.append(
            ToolCall(tool_name="calculator", tool_input={"expr": query}, tool_output="42")
        )
    if not tool_calls:
        tool_calls.append(
            ToolCall(tool_name="general_search", tool_input={"q": query}, tool_output="some result")
        )

    answer = f"Based on {len(tool_calls)} tool call(s): here's the answer to '{query}'."
    return tool_calls, answer


async def main() -> None:
    test_cases = [
        TrajectoryTestCase(
            question="What's the weather in Paris?",
            expected_tool_sequence=["search_weather"],
        ),
        TrajectoryTestCase(
            question="What is 15 times 3?",
            expected_tool_sequence=["calculator"],
        ),
        TrajectoryTestCase(
            question="Tell me about black holes",
            expected_tool_sequence=["general_search"],
        ),
    ]

    agent = CallableAgentAdapter(my_agent)
    metrics = [ToolSelectionRecall(), ToolSelectionPrecision(), ToolCallOrderCorrectness()]

    report = await run_trajectory_evaluation(
        test_cases, agent, metrics, TrajectoryRunConfig(run_id="agentic-example")
    )

    print(f"Evaluated {len(report.results)} test cases\n")
    for result in report.results:
        question = next(
            tc.question for tc in test_cases if tc.test_case_id == result.test_case_id
        )
        tools_used = [c.tool_name for c in result.tool_calls]
        print(f"Q: {question}")
        print(f"  Tools called: {tools_used}")
        print(f"  A: {result.final_answer}\n")

    by_metric: dict[str, list[float]] = {}
    for s in report.scores:
        by_metric.setdefault(s.metric_name, []).append(s.score_value)
    print("Average scores:")
    for name, values in by_metric.items():
        print(f"  {name}: {sum(values) / len(values):.3f}")


if __name__ == "__main__":
    asyncio.run(main())
