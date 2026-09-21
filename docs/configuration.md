# Configuration reference

Every field accepted by `eval_config.json` / `eval_config.yaml`, passed to `rageval run <config>`.

## Required fields

| Field | Type | Description |
|---|---|---|
| `dataset` | string | Path to a JSON or YAML test-set file (see [Quickstart](quickstart.md#3-write-a-small-test-set)). |
| `retriever` | string | `"module.path:attribute"` reference to your retriever function or a [`RetrieverAdapter`][rag_score.adapters.base.RetrieverAdapter] instance. Resolved relative to your current working directory. |
| `generator` | string | Same format, for your generator function or [`GeneratorAdapter`][rag_score.adapters.base.GeneratorAdapter] instance. |
| `metrics` | list[string] | Which metrics to run - see the [Metrics reference](metrics.md) for every valid name. |

## Optional fields

| Field | Type | Default | Description |
|---|---|---|---|
| `project_name` | string | `"default"` | Recorded in the SQLite export's `dim_runs` table. |
| `top_k` | int | `5` | Passed to your retriever as its `top_k` argument. |
| `max_concurrency` | int | `8` | How many test cases run concurrently. Raise for offline metrics on a fast machine; lower if you're hitting a rate-limited judge API. |
| `output` | string | none | Path to write a full JSON dump of results + scores. |
| `sqlite_output` | string | none | Path to a `.db` file - see [Export options](#export-options). |
| `html_output` | string | none | Path to write a self-contained `report.html`. |
| `environment` | string | `"local"` | Recorded in the SQLite export; useful for distinguishing local runs from CI runs. |
| `encoder_model` | string | `"all-MiniLM-L6-v2"` | Sentence-transformers model name used by `local_faithfulness`/`local_answer_relevance` (requires `pip install ragmark[local-ml]`). Ignored if those metrics aren't requested. |
| `telemetry` | object | off | Enables token/cost tracking. `{"model": "gpt-4o-mini"}` at minimum; add `"pricing": {"model-name": [prompt_price_per_1k, completion_price_per_1k]}` to override the built-in pricing table. Requires `pip install ragmark[telemetry]` for accurate counts (falls back to a rough estimate otherwise). |

## `judge`

Required only if any metric in `metrics` needs an LLM judge (`faithfulness`, `answer_relevance`, `context_precision`).

```json
{ "judge": { "provider": "anthropic", "model": "claude-haiku-4-5" } }
```

| Field | Type | Description |
|---|---|---|
| `provider` | string | One of `"openai"`, `"anthropic"`, `"local"`. |
| `model` | string | Model name. Required for `"local"`; optional for the others (each has a default). |
| `api_key` | string | Optional - if omitted, falls back to the provider's standard env var (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`). Not used by `"local"`. |
| `base_url` | string | `"local"` only - defaults to Ollama's `http://localhost:11434/v1`. Override for vLLM, LM Studio, etc. |

## Export options

**`output`** writes everything - raw results and every metric score - as one JSON file. Good for one-off inspection or feeding into your own tooling.

**`sqlite_output`** appends the run to a star-schema SQLite database (`dim_runs`, `dim_test_cases`, `fact_evaluations`, `fact_metric_scores`) - safe to point at the same file across many runs, since results accumulate rather than overwrite. Point Power BI, Superset, or a plain SQL query at it to track scores over time.

**`html_output`** writes a self-contained report - summary cards, a score-by-metric bar chart, and a per-question breakdown with judge reasoning shown as tooltips. No server needed to view it.

All three can be set at once; each is independent.

## Full example

```json
{
  "dataset": "test_set.json",
  "retriever": "my_pipeline:my_retriever",
  "generator": "my_pipeline:my_generator",
  "metrics": ["precision_at_5", "recall_at_5", "mrr", "faithfulness", "answer_relevance"],
  "judge": { "provider": "openai", "model": "gpt-4o-mini" },
  "top_k": 5,
  "max_concurrency": 10,
  "output": "results.json",
  "sqlite_output": "eval_history.db",
  "html_output": "report.html",
  "project_name": "my-rag-app"
}
```

## `rageval synthesize`

Generates a test set from raw documents instead of writing one by hand. This is a separate CLI command, not a config field:

```bash
rageval synthesize <docs_dir> --judge '{"provider": "anthropic"}' [options]
```

| Flag | Default | Description |
|---|---|---|
| `--output` / `-o` | `test_set.json` | Where to write the generated test set. |
| `--judge` | *(required)* | Same JSON shape as the `judge` config field above. |
| `--chunk-size` | `500` | Words per chunk. |
| `--chunk-overlap` | `50` | Overlapping words between consecutive chunks. Must be smaller than `--chunk-size`. |
| `--questions-per-chunk` | `1` | How many question/answer pairs to generate per chunk. |
| `--max-concurrency` | `5` | Concurrent judge calls. |

Only `.txt` and `.md` files in `<docs_dir>` are read (non-recursive). See [`synthesize_test_set`][rag_score.synthesize.synthesize_test_set] for the Python API if you want to call this from code instead of the CLI - e.g. to pass in documents loaded from somewhere other than a local directory.
