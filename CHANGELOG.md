# Changelog

All notable changes to this project are documented here.

## [0.12.0] - 2026-10-06
### Added
- Statistical significance testing for regression detection (Feature 8):
  - New `rag_score.stats` module with Welch's t-test and bootstrap confidence interval methods
  - `--max-regression-significant` flag for `rageval gate`: only fails if regression exceeds threshold AND is statistically significant (p < alpha)
  - `--significance-level` flag for `rageval gate`: configurable significance threshold (default: 0.05)
  - `--with-significance` flag for `rageval compare`: adds p-value/significance columns to show whether differences are likely real or noise
  - Falls back to bootstrap method when scipy is not installed (installed via `rag-score[stats]` extra)
  - Handles degenerate cases (empty samples, single samples, zero variance) gracefully
- Updated optional dependencies:
  - Added `stats` extra containing `scipy>=1.10.0`
  - Updated `all` and `dev` extras to include the stats dependency
- Updated documentation:
  - Added statistical significance testing section to README.md
  - Expanded CI/CD integration guide with quality gates subsection covering statistical significance
  - Updated CHANGELOG.md

### Updated
- Bumped version to 0.12.0.

## [0.11.0] - 2026-10-05
### Added
- New CLI command: `rageval gate <results.json> --baseline <baseline.json>` to enforce quality gates in CI/CD pipelines.
  - Supports inline flags: `--min-score <metric>=<value>` and `--max-regression <metric>=<value>` (repeatable).
  - Supports `--gate-config <path>` for JSON/YAML config files containing min_score and max_regression dictionaries.
  - Inline flags override the config file if both are provided for the same metric.
  - Outputs a markdown table showing each rule, current value, threshold, and PASS/FAIL status.
  - Exit code 0 if all gates pass, 1 if any gate fails.
  - A metric referenced in a gate rule but absent from the results file fails that rule with a clear message.
  - No rules configured results in exit 0 with an explicit message.
- Updated the Feature 3 CI workflow template (`.github/workflows/ragmark-eval-template.yml`) to add an optional `rageval gate` step after the existing compare step, clearly marked as commented-out/opt-in with inline instructions.
- Added documentation for quality gates in README.md and expanded the CI/CD integration guide (docs/ci-cd-integration.md) with a quality gates subsection.

### Updated
- Bumped version to 0.11.0.

## [0.9.0]

### Added
- New metric: ContextCarryOver for evaluating multi-turn conversational RAG systems. Measures whether generated answers correctly resolve references (pronouns, ellipsis, "the second one", "that plan") to earlier turns in the conversation.
  - Implemented in `rag_score/metrics/generation/context_carry_over.py`.
  - Requires an LLM judge; returns 0.0 without calling the judge when conversation history is missing or malformed.
  - Runner integration: passes conversation history to generators that opt in via a `history` keyword argument in their `generate` method.
  - CLI registration: available as `context_carry_over` in config metrics arrays.
  - Documentation: added to `docs/metrics.md` and `docs/api-reference.md`.
  - Example: `examples/multi_turn_example.py` demonstrates usage with fake retriever, generator, and judge.
- Judge response caching: avoid re-paying for identical LLM judge calls when re-running evaluations on unchanged test cases.
  - New `rag_score.cache` module with `JudgeCache` protocol, `InMemoryCacheBackend`, `FileCacheBackend`, and `CacheConfig`.
  - Integrated into `LLMJudge.judge()` in `rag_score/judges/base.py`; optional `cache` parameter added to judge constructors.
  - CLI wiring: add a `cache` section to the judge config, e.g. `{"provider": "openai", "cache": {"enabled": true, "path": ".ragmark_cache/judge_cache.db", "max_age_seconds": 86400}}`.
  - Cache effectiveness printed after each run: "Judge cache: 34 hits, 12 misses (74% hit rate)".
  - Updated CI workflow template (`.github/workflows/ragmark-eval-template.yml`) to persist the cache directory between runs using `actions/cache`.
- Updated telemetry token counting to include conversation history text when passed to the generator.

## [0.8.0]

### Added
- New CLI command: `rageval compare <results_a.json> <results_b.json>` to compare two evaluation result files and output a Markdown table of metric differences.
  - Computes per-metric average scores from the `summary` field in each result file (no recomputation from raw scores).
  - Outputs a Markdown table by default, with optional `--output` flag to write to a file.
  - Uses ✅ for improvements (delta > threshold, default 0.02), ⚠️ for regressions (delta < -threshold), and no emoji for negligible changes.
  - Handles missing metrics gracefully (shows "N/A" for missing side).
  - Handles malformed JSON input with clean error messages (no traceback).
- New GitHub Actions workflow template: `.github/workflows/ragmark-eval-template.yml` for CI/CD integration.
  - Triggers on pull_request.
  - Evaluates the PR branch and the main branch separately using `rageval run`.
  - Compares results with `rageval compare` and posts the Markdown table as a PR comment using `marocchino/sticky-pull-request-comment`.
  - Includes clear comments indicating where users must customize (config path, dataset path, Python version, API key secrets).
- New documentation page: `docs/ci-cd-integration.md` explaining how to use the workflow template, customize it, set required secrets, and interpret the resulting PR comment.

## [0.7.0]

### Added
- Advanced synthetic test-set generation: `rageval synthesize` now supports `--query-types` parameter to generate adversarial, multi-hop, and unanswerable questions in addition to standard questions
  - Adversarial questions: include typos, vague/colloquial phrasing, or ambiguous pronouns to test robustness to messy real-world input
  - Multi-hop questions: require information from two adjacent text chunks to test multi-step reasoning capabilities
  - Unanswerable questions: sound plausible but cannot be answered from the context to test hallucination resistance
  - Each generated test case is tagged with its query type in the `metadata` field for later analysis
- Backward compatibility: when `--query-types` is not specified, defaults to "standard" only, preserving existing behavior

## [0.6.0]

### Added
- Telemetry: token counting and cost estimation, opt-in via `RunConfig(telemetry=TelemetryConfig(...))` or the CLI's `telemetry` config field. Populates `total_tokens`/`estimated_cost_usd`, fields that existed in the schema since v0.1.0 but were never wired up.
  - Token counts estimated from actual prompt/completion text via `tiktoken`, with a dependency-free word-based fallback when `tiktoken` isn't installed or can't reach its encoding CDN (offline/firewalled environments)
  - Configurable pricing table (`DEFAULT_PRICING` covers a handful of common models); unpriced models report `None` cost, never a misleading `$0.00`
  - New `retrieval_latency_sec`/`generation_latency_sec` columns in the SQLite export (as native generated columns, always in sync with the existing `_ms` values) and the Pandas export
  - New token/cost summary cards and a per-row column in the HTML report, shown only when telemetry is enabled
  - New optional extra: `ragmark[telemetry]`
- Closed a pre-existing test coverage gap in `dataframe_export.py` (had zero tests since it was first written)

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