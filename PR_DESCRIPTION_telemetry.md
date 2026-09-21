## What does this PR do?

Adds Feature 1 from the enterprise-MLOps roadmap: telemetry (token counting + cost estimation). Off by default; opt in via `RunConfig(telemetry=TelemetryConfig(...))` or the CLI's `telemetry` config field.

## Changes

- **New:** `rag_score/telemetry.py` — `TelemetryConfig`, `count_tokens()`, `estimate_cost()`, `DEFAULT_PRICING`
  - Token counts are estimated from the actual prompt/completion **text** via `tiktoken`, not pulled from provider-reported usage — that's deliberate, not a shortcut. Getting real provider usage would mean changing `GeneratorAdapter.generate() -> str` to return a richer object, breaking every existing adapter (Callable/LangChain/LlamaIndex) and every example/test written against the current interface.
  - `count_tokens()` falls back to a dependency-free word-based estimate whenever `tiktoken` isn't installed *or can't reach its encoding CDN* — a real problem in offline/firewalled environments, not just a sandbox quirk (I confirmed this by hitting it for real: `tiktoken.encoding_for_model()` tries to download from `openaipublic.blob.core.windows.net` on first use per model). This keeps telemetry consistent with the project's existing local-first philosophy (`LocalJudge`, local ML metrics).
  - `estimate_cost()` returns `None` (not `0.0`) for unpriced models — silently reporting free would be a meaningfully different, misleading claim.
- **`core/runner.py`:** `RunConfig` gets an optional `telemetry: TelemetryConfig | None = None` field. `EvalResult.total_tokens`/`.estimated_cost_usd` — fields that have existed in the schema since v0.1.0 but were never populated — now get real values when telemetry is enabled.
- **`export/sqlite_export.py`:** `retrieval_latency_sec`/`generation_latency_sec` added as SQLite **generated columns** (`GENERATED ALWAYS AS (... / 1000.0) STORED`) rather than duplicated insert logic — always exactly in sync with the existing `_ms` columns, zero extra write code.
- **`export/dataframe_export.py`:** same `_sec` columns added to `results_to_dataframe()`.
- **`report/html_report.py` + template:** token/cost summary cards and a per-row column, shown only when telemetry was enabled for that run (checked via `total_tokens is not None`, the same "not tracked" convention used everywhere else in this package).
- **CLI:** new `telemetry` config field on `rageval run`; JSON pricing tables (`{"model": [prompt, completion]}`) converted to the tuple shape `TelemetryConfig` expects, since JSON has no tuple type.
- **New extra:** `ragmark[telemetry]` (`tiktoken>=0.7.0`)
- **README/docs:** new sections, `CHANGELOG.md` entry, version bump to 0.6.0

## Testing

- 14 new tests in `test_telemetry.py` — including a **genuine** (not mocked) test of the fallback path, since this sandbox can't reach tiktoken's CDN either; the mocked-success and unknown-model-KeyError paths are covered separately via mocking
- 5 new runner integration tests confirming: telemetry off leaves fields `None` (the critical backward-compat check), telemetry on populates real values, unpriced models count tokens but report `None` cost, latency tracking is unaffected, and a failed test case correctly has no telemetry (nothing was generated to count)
- 3 new SQLite export tests confirming the generated `_sec` columns are always exactly `ms / 1000`, and telemetry fields are populated/null correctly
- **Closed a pre-existing gap:** `dataframe_export.py` had zero test coverage before this PR (not related to telemetry — just never written). Added 12 tests covering both the existing functions and the new telemetry columns.
- 2 new HTML report tests confirming the telemetry cards appear only when enabled
- Real end-to-end CLI verification: a full `rageval run` with `telemetry` configured, confirming correct values landed in both the JSON output and the HTML report
- **251/251 tests passing** (up from 215) — critically, all 215 pre-existing tests pass unchanged, confirming zero breaking changes
- `ruff check` — all clean
- `mkdocs build --strict` — clean
- Full PyPI-readiness check repeated at v0.6.0: `python -m build`, `twine check` (passed), confirmed `telemetry.py` is present in the built wheel

## Checklist

- [x] Tests pass locally (`pytest tests/ -v`)
- [x] Lint passes locally (`ruff check rag_score/ tests/ examples/`)
- [x] Added tests for new functionality
- [x] Updated `README.md`
- [x] New optional dependency (`tiktoken`) imported lazily, added to its own extra

## Related issue

N/A
