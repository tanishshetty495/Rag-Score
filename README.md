# rag-score

[![Tests](https://github.com/tanishshetty495/Rag-Score/actions/workflows/test.yml/badge.svg)](https://github.com/tanishshetty495/Rag-Score/actions/workflows/test.yml)

Know if your RAG pipeline actually works — in 3 lines of code, with zero framework lock-in.

`rag-score` is a lightweight, framework-agnostic Python library for evaluating Retrieval-Augmented Generation pipelines. Bring your own retriever (LangChain, LlamaIndex, raw FAISS, an HTTP call — anything), get retrieval and generation quality metrics back.

**Core promise: zero lock-in, zero forced API cost.**
- Works with any pipeline via a two-method adapter interface
- Retrieval metrics (Precision@k, Recall@k, MRR, nDCG) run **completely offline** — no API keys needed
- Async-first execution engine so a few hundred test cases don't take hours
- Exports to SQLite (star schema, BI-ready) or Pandas DataFrames for notebooks
- CI/CD ready: writes `$GITHUB_STEP_SUMMARY` automatically in GitHub Actions

## Install

```bash
pip install ragmark
```

> The PyPI listing name is `ragmark` (the name `rag-score` was already too similar to several existing packages) — everything else, including `import rag_score` and the `rageval` CLI command, is unaffected.

## 60-second quickstart

**1. Describe your pipeline as two functions:**

```python
# my_pipeline.py
from rag_score.core.types import RetrievedChunk

async def my_retriever(query: str, top_k: int):
    # call your actual retriever here — FAISS, Pinecone, whatever
    return [RetrievedChunk(doc_id="doc_1", text="...")]

async def my_generator(query: str, context: list[RetrievedChunk]):
    # call your actual LLM here
    return "the generated answer"
```

**2. Write a small test set** (`test_set.json`):

```json
[
  {
    "question": "What is the refund policy?",
    "expected_doc_ids": ["doc_1", "doc_2"]
  }
]
```

**3. Write a config** (`eval_config.json`):

```json
{
  "dataset": "test_set.json",
  "retriever": "my_pipeline:my_retriever",
  "generator": "my_pipeline:my_generator",
  "metrics": ["precision_at_5", "recall_at_5", "mrr", "ndcg_at_5"],
  "html_output": "report.html"
}
```

**4. Run it:**

```bash
rageval run eval_config.json
```

```
Running 1 test cases with 4 metrics...

Evaluated 1 test cases (0 failed)

Metric          Score
------------------------
precision_at_5  1.000
recall_at_5     0.500
mrr             1.000
ndcg_at_5       0.613

HTML report written to report.html
```

Open `report.html` — no server required, it's a single self-contained file.

## LLM-judge metrics (v0.2, needs an API key)

Add a `judge` section to your config and reference judge-based metrics by name:

```json
{
  "dataset": "test_set.json",
  "retriever": "my_pipeline:my_retriever",
  "generator": "my_pipeline:my_generator",
  "metrics": ["faithfulness", "answer_relevance", "context_precision"],
  "judge": { "provider": "anthropic", "model": "claude-haiku-4-5" },
  "html_output": "report.html"
}
```

The judge reads its API key from the provider's usual environment variable (`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`) — no need to put it in the config. Every judge verdict includes a one-sentence reasoning string, visible as a tooltip on each score in the HTML report.

| Metric | What it measures |
|---|---|
| `faithfulness` | Is every claim in the answer supported by the retrieved context? |
| `answer_relevance` | Does the answer actually address the question asked? |
| `context_precision` | Of the retrieved chunks, what fraction are actually relevant? |

Mix and match freely with the offline metrics in the same run — `["precision_at_5", "faithfulness"]` works fine.

Every judge automatically retries transient failures (rate limits, connection blips) with exponential backoff — 2 retries, 1 second base delay by default. Override per judge:

```json
{ "...": "...", "judge": { "provider": "anthropic", "max_retries": 5, "retry_base_delay": 2.0 } }
```

## Using it as a library (Jupyter/notebooks)

```python
import asyncio
from rag_score.core.dataset import load_dataset
from rag_score.core.runner import RunConfig, run_evaluation
from rag_score.adapters.base import CallableRetrieverAdapter, CallableGeneratorAdapter
from rag_score.metrics.retrieval.precision_at_k import PrecisionAtK
from rag_score.metrics.retrieval.mrr import MRR
from rag_score.export.dataframe_export import full_report_dataframe

test_cases = load_dataset("test_set.json")
retriever = CallableRetrieverAdapter(my_retriever)
generator = CallableGeneratorAdapter(my_generator)
metrics = [PrecisionAtK(k=5), MRR()]

report = asyncio.run(
    run_evaluation(test_cases, retriever, generator, metrics, RunConfig(run_id="run-1"))
)

df = full_report_dataframe(report, test_cases)
df.describe()
```

## Offline metrics (zero API keys required)

| Metric | What it measures |
|---|---|
| `precision_at_k` | Of the top-k retrieved chunks, what fraction are relevant? |
| `recall_at_k` | Of all relevant chunks, what fraction did top-k retrieval surface? |
| `mrr` | How high up the ranking was the first relevant hit? |
| `ndcg_at_k` | Ranking quality, rewarding relevant results appearing earlier |

## BI export

```json
{ "..." : "...", "sqlite_output": "eval_history.db" }
```

Every run appends to the same SQLite file using a star schema (`dim_runs`, `dim_test_cases`, `fact_evaluations`, `fact_metric_scores`), so you can point Power BI, Superset, or a plain SQL query at your evaluation history over time.

## Generating a test set from your own docs

Don't have a test set yet? Point `rageval synthesize` at a folder of `.txt`/`.md` files and it'll chunk them and generate questions + ground-truth answers via an LLM judge:

```bash
rageval synthesize ./docs --judge '{"provider": "anthropic"}' --output test_set.json
```

```
Loaded 4 document(s) from ./docs
Chunking at 500 words (overlap 50) and generating questions...

Generated 12 test case(s), 0 chunk(s) failed
Written to test_set.json
```

The output is a real `test_set.json` - load it with `load_dataset()` or point `rageval run` straight at it, same as a hand-written one. `expected_doc_ids` in the generated set reference chunk-level IDs (`mydoc::chunk_0`), so precision/recall/MRR/nDCG only work against it if your retriever returns matching chunk IDs; the LLM-judge metrics (`faithfulness`, `answer_relevance`) work regardless since they don't need `expected_doc_ids` at all.

Useful flags: `--chunk-size` (words per chunk, default 500), `--chunk-overlap` (default 50), `--questions-per-chunk` (default 1), `--max-concurrency` (default 5). A chunk that fails to generate valid JSON doesn't abort the run - it's listed at the end so you can see exactly what didn't make it in.

## Framework adapters

Already using LangChain or LlamaIndex? Wrap your existing retriever/chain instead of rewriting it:

```python
# LangChain
from rag_score.adapters.langchain_adapter import LangChainRetrieverAdapter, LangChainGeneratorAdapter
retriever = LangChainRetrieverAdapter(my_vectorstore.as_retriever())
generator = LangChainGeneratorAdapter(my_lcel_chain)

# LlamaIndex
from rag_score.adapters.llamaindex_adapter import LlamaIndexRetrieverAdapter, LlamaIndexGeneratorAdapter
retriever = LlamaIndexRetrieverAdapter(my_index.as_retriever())
generator = LlamaIndexGeneratorAdapter(my_index.as_query_engine())  # or a bare LLM
```

See `examples/langchain_example.py` and `examples/llamaindex_example.py` for full runnable versions.

## Local judges (zero API cost, zero data leaving your machine)

Point a judge at Ollama or any other OpenAI-compatible local server instead of a paid API:

```json
{ "...": "...", "judge": { "provider": "local", "model": "llama3.1" } }
```

Or in Python:

```python
from rag_score.judges.local_judge import LocalJudge
judge = LocalJudge(model="llama3.1")  # defaults to http://localhost:11434/v1 (Ollama)
```

## Local ML metrics (no LLM at all)

`local_faithfulness` and `local_answer_relevance` go further than the Ollama judge above - no LLM, no inference server, just a small embedding model computing semantic similarity locally:

```bash
pip install ragmark[local-ml]
```

```json
{ "...": "...", "metrics": ["local_faithfulness", "local_answer_relevance"] }
```

The model (`all-MiniLM-L6-v2` by default, [sentence-transformers](https://www.sbert.net/)) downloads once on first use, then runs fully offline. This trades some accuracy for speed and zero cost - cosine similarity catches "the answer is about something completely different" reliably, but won't reason about factual correctness the way an LLM judge can. Use it as a fast free first pass, or alongside `faithfulness`/`answer_relevance` rather than as a strict replacement for them.

## Agentic trajectory evaluation

Evaluating a multi-step agent (one that calls tools before answering) instead of a single retrieve-then-generate pass? `rageval run-trajectory` is a separate command for that:

```python
# my_agent.py
from rag_score.agentic.types import ToolCall

async def my_agent(query: str) -> tuple[list[ToolCall], str]:
    # call your actual agent here - LangGraph, a custom loop, whatever
    return [ToolCall(tool_name="search", tool_output="...")], "the final answer"
```

```json
{
  "dataset": "trajectory_test_set.json",
  "agent": "my_agent:my_agent",
  "metrics": ["tool_selection_recall", "tool_selection_precision", "tool_call_order_correctness"]
}
```

```bash
rageval run-trajectory eval_config.json
```

| Metric | What it measures |
|---|---|
| `tool_selection_recall` | Of the expected tools, what fraction did the agent actually call? |
| `tool_selection_precision` | Of the tools the agent called, what fraction were actually expected? |
| `tool_call_order_correctness` | Were the expected tools called in the right relative order (extra calls in between are fine)? |

All three are offline (zero API keys) and mirror the naming/semantics of the single-shot retrieval metrics deliberately - same questions, applied to a sequence of tool names instead of a set of retrieved doc IDs. See `examples/agentic_example.py` for a full runnable version.

## Roadmap

Nothing left from the original blueprint - the current focus is polish, real-world hardening, and the first PyPI release. Ideas and PRs welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).

## Why not Ragas / TruLens / DeepEval?

Those are excellent, more full-featured tools. `rag-score` exists for the case where you want something smaller: a library you can read end-to-end in an afternoon, with an adapter interface that doesn't assume you're using any particular framework, and a set of metrics that work with zero API keys before you ever reach for an LLM judge.

## License

MIT
