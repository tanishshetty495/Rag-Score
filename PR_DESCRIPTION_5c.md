## What does this PR do?

Adds agentic trajectory evaluation — a genuinely separate evaluation mode for multi-step agents that call tools before answering, rather than the single retrieve-then-generate pass everything else in this library handles. New `rag_score.agentic` subpackage, new `rageval run-trajectory` CLI command, three new offline metrics.

## Changes

- **New subpackage:** `rag_score/agentic/` — kept separate from `core/`/`metrics/` rather than bolted onto the existing types, since a tool-call trajectory is a different shape of thing to evaluate than retrieved-context-plus-answer
  - `types.py` — `ToolCall`, `TrajectoryTestCase`, `TrajectoryEvalResult`, `load_trajectory_dataset()`
  - `adapters.py` — `AgentAdapter`, `CallableAgentAdapter` (one method: `run(query) -> (tool_calls, answer)`, since the tool-calling loop is internal to the agent)
  - `runner.py` — `run_trajectory_evaluation()`, `TrajectoryRunConfig`, `TrajectoryRunReport` (same concurrency-bounded, failure-isolated two-pass structure as the existing runner)
  - `metrics_base.py` — `TrajectoryMetric` (parallel to `Metric`, not shared — different enough input shapes that sharing a base class wasn't worth the generic-typing complexity)
  - `metrics/` — `ToolSelectionRecall`, `ToolSelectionPrecision`, `ToolCallOrderCorrectness` (subsequence-based, not exact-match — extra/unrelated tool calls don't unfairly penalize order correctness)
- **CLI:** new `rageval run-trajectory <config>` command, config shape: `dataset`, `agent`, `metrics`, `output`
- **New example:** `examples/agentic_example.py`
- **Docs:** new `docs/agentic.md` page, added to nav, new API reference entries
- **README:** new "Agentic trajectory evaluation" section; roadmap cleared (all four original items now shipped)
- **pyproject.toml:** version bump to 0.4.0, no new dependencies

## A real bug caught and fixed during this work

While wiring the new CLI command, `def main()` and the `if __name__ == "__main__"` guard ended up positioned *before* `run-trajectory`'s `@cli.command()` decorator in the file. Since Python executes a module top-to-bottom, running `python cli.py` directly (rather than through the installed `rageval` console script) would have called `main()` before `run-trajectory` was registered on the `cli` group — silently missing the new command in that invocation path. Caught by explicitly checking `rageval --help` output before considering the feature done; fixed by moving `main()`/the guard to the true end of the file, after every command definition.

## Testing

- 57 new tests across `test_agentic_types.py`, `test_agentic_adapters.py`, `test_agentic_metrics.py`, `test_agentic_runner.py`, plus 4 new cases in `test_cli.py`
- The subsequence-matching algorithm behind `tool_call_order_correctness` is tested directly and exhaustively (exact match, interspersed extra calls, wrong order, missing elements, duplicate tool names) since iterator-consumption logic is easy to get subtly wrong
- A full runner-level regression test locks in the exact "correct tools, wrong order" scenario verified by hand during development: recall and precision both score 1.0, order correctness correctly scores 0.0
- Real CLI smoke-tested end-to-end from an actual terminal invocation (not just `CliRunner`), including error paths (unknown metric, missing required config field)
- Verified the built wheel actually includes the new `agentic/` subpackage (9 files) and passes `twine check`
- **202/202 tests passing** (up from 154)
- `ruff check` — all clean
- `mkdocs build --strict` — clean, no broken cross-references

## Checklist

- [x] Tests pass locally (`pytest tests/ -v`)
- [x] Lint passes locally (`ruff check rag_score/ tests/ examples/`)
- [x] Added tests for new functionality (no real API calls)
- [x] Updated `README.md`
- [x] No new dependencies

## Related issue

N/A