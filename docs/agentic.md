# Agentic trajectory evaluation

For evaluating a multi-step agent (one that decides which tools to call, in what order, before producing a final answer) rather than a single retrieve-then-generate pass. This is a separate evaluation mode from everything else in this doc site - a different CLI command, different config shape, different metrics - since a tool-calling trajectory is a genuinely different kind of thing to score than a retrieved-context-plus-answer pair.

## 1. Describe your agent as one function

```python
# my_agent.py
from rag_score.agentic.types import ToolCall

async def my_agent(query: str) -> tuple[list[ToolCall], str]:
    # call your actual agent here - LangGraph, a custom ReAct loop, whatever
    # a real agent would decide which tools to call using an LLM; return
    # every tool call it actually made, in order, plus its final answer
    return [ToolCall(tool_name="search", tool_input={"q": query}, tool_output="...")], "the final answer"
```

Unlike the retriever/generator split used for single-shot RAG, there's only one function here - the tool-calling loop happens inside your agent, and this adapter's only job is to report what it did.

## 2. Write a test set

```json title="trajectory_test_set.json"
[
  {
    "question": "What's the weather in Paris and what's 15% of 200?",
    "expected_tool_sequence": ["search_weather", "calculator"]
  }
]
```

`expected_tool_sequence` is optional - metrics that need it return `0.0` without it, same convention as `expected_doc_ids` on the single-shot `TestCase`.

## 3. Write a config and run it

```json title="trajectory_config.json"
{
  "dataset": "trajectory_test_set.json",
  "agent": "my_agent:my_agent",
  "metrics": ["tool_selection_recall", "tool_selection_precision", "tool_call_order_correctness"]
}
```

```bash
rageval run-trajectory trajectory_config.json
```

## Metrics

| Metric | Class | What it measures |
|---|---|---|
| `tool_selection_recall` | [`ToolSelectionRecall`][rag_score.agentic.metrics.tool_selection_recall.ToolSelectionRecall] | Of the expected tools, what fraction did the agent actually call (anywhere, order not considered)? |
| `tool_selection_precision` | [`ToolSelectionPrecision`][rag_score.agentic.metrics.tool_selection_precision.ToolSelectionPrecision] | Of the tools the agent called, what fraction were actually expected? |
| `tool_call_order_correctness` | [`ToolCallOrderCorrectness`][rag_score.agentic.metrics.tool_call_order.ToolCallOrderCorrectness] | Were the expected tools called in the right relative order? Extra/unrelated calls in between are fine - this is a subsequence check, not an exact-match check. |

### Why three separate metrics

An agent that calls every available tool "just in case" would score perfectly on recall (it got all the expected ones) while scoring poorly on precision (most of its calls were wasted). An agent that calls the right tools in the wrong order scores perfectly on recall and precision but zero on order correctness - which matters when a downstream tool depends on an earlier one's output (e.g. looking up a location before checking its weather). Reading all three together tells you more than any one alone.

## Config reference

| Field | Type | Description |
|---|---|---|
| `dataset` | string | *(required)* Path to a JSON or YAML trajectory test set. |
| `agent` | string | *(required)* `"module.path:attribute"` reference to your agent function or an [`AgentAdapter`][rag_score.agentic.adapters.AgentAdapter] instance. |
| `metrics` | list[string] | *(required)* Which trajectory metrics to run. |
| `project_name` | string | Recorded for your own reference; not yet exported to SQLite (see note below). |
| `max_concurrency` | int | Default `8`. |
| `output` | string | Path to write a full JSON dump of results + scores. |

Note: `sqlite_output` and `html_output` aren't wired up for trajectory runs yet - use `output` for now and inspect the JSON directly, or export it yourself with [`TrajectoryEvalResult`][rag_score.agentic.types.TrajectoryEvalResult]'s `model_dump()`.

## Python API

```python
import asyncio
from rag_score.agentic.types import load_trajectory_dataset
from rag_score.agentic.adapters import CallableAgentAdapter
from rag_score.agentic.runner import TrajectoryRunConfig, run_trajectory_evaluation
from rag_score.agentic.metrics.tool_selection_recall import ToolSelectionRecall

test_cases = load_trajectory_dataset("trajectory_test_set.json")
agent = CallableAgentAdapter(my_agent)
metrics = [ToolSelectionRecall()]

report = asyncio.run(
    run_trajectory_evaluation(test_cases, agent, metrics, TrajectoryRunConfig(run_id="run-1"))
)
```

See `examples/agentic_example.py` for a full runnable version.
