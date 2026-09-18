# Changelog

All notable changes to this project are documented here.

## [0.5.0]

### Added
- Automatic retry with exponential backoff for LLM-judge API calls (`max_retries`, `retry_base_delay`, configurable per-judge or via the `judge` config's `max_retries`/`retry_base_delay` fields). A transient network blip or rate limit no longer kills that test case's score outright.
- Progress bar for `rageval run` and `rageval run-trajectory`, via `click.progressbar` (no new dependency).
- `on_progress` callback parameter on `run_evaluation()` and `run_trajectory_evaluation()` for library users who want their own progress reporting.

## [0.4.0]

### Added
- Agentic trajectory evaluation: new `rag_score.agentic` subpackage for evaluating multi-step agents (tool-calling sequences) rather than single retrieve-then-generate pipelines.
  - `ToolCall`, `TrajectoryTestCase`, `TrajectoryEvalResult`, `AgentAdapter`/`CallableAgentAdapter`
  - Metrics: `tool_selection_recall`, `tool_selection_precision`, `tool_call_order_correctness` (subsequence-based, not exact-match)
  - New CLI command: `rageval run-trajectory`
  - New example: `examples/agentic_example.py`

## [0.3.0]

### Added
- Local ML-based metrics: `local_faithfulness`, `local_answer_relevance` — semantic similarity via a local embedding model (`sentence-transformers`), zero LLM calls, zero API keys, zero network access after the model's first download.
- New optional extra: `rag-score[local-ml]`.
- `encoder_model` config field to override the default embedding model.

### Fixed
- A metric raising during scoring (e.g. failing to download an embedding model) now surfaces a clean CLI error instead of a raw traceback.

## [0.2.0]

### Added
- LangChain and LlamaIndex framework adapters (`LangChainRetrieverAdapter`/`LangChainGeneratorAdapter`, `LlamaIndexRetrieverAdapter`/`LlamaIndexGeneratorAdapter`).
- `LocalJudge` for Ollama or any OpenAI-compatible local inference server — zero API cost, zero data leaving the user's machine.
- Score bar chart (inline SVG) in the HTML report.
- `examples/` folder with runnable end-to-end scripts (raw, LangChain, LlamaIndex).
- Synthetic test-set generation: `rageval synthesize` chunks raw `.txt`/`.md` documents and generates a real `test_set.json` via an LLM judge.
- LLM-judge generation metrics: `faithfulness`, `answer_relevance`, `context_precision`, with OpenAI and Anthropic judge implementations.
- CI: GitHub Actions running the test suite across Python 3.10-3.12, plus lint (`ruff`).
- Docs site (`mkdocs-material`), deployed to GitHub Pages.

## [0.1.0] - Initial release

### Added
- Core evaluation engine: async, concurrency-bounded, per-test-case failure isolation.
- Offline retrieval metrics (zero API keys required): `precision_at_k`, `recall_at_k`, `mrr`, `ndcg_at_k`.
- `RetrieverAdapter`/`GeneratorAdapter` interfaces with callable-function wrappers — zero framework lock-in.
- CLI: `rageval run <config>`.
- SQLite star-schema export (`dim_runs`, `dim_test_cases`, `fact_evaluations`, `fact_metric_scores`) for BI tools.
- Pandas DataFrame export for notebook users.
- Self-contained HTML report generator.
- `$GITHUB_STEP_SUMMARY` output for CI runs.
