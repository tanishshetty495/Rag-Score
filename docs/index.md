# rag-score

Know if your RAG pipeline actually works — in 3 lines of code, with zero framework lock-in.

`rag-score` is a lightweight, framework-agnostic Python library for evaluating Retrieval-Augmented Generation pipelines. Bring your own retriever (LangChain, LlamaIndex, raw FAISS, an HTTP call — anything), get retrieval and generation quality metrics back.

## Core promise

- **Zero lock-in** — works with any pipeline via a two-method adapter interface
- **Zero forced API cost** — retrieval metrics ([Precision@k](metrics.md), Recall@k, MRR, nDCG) run completely offline
- **Async-first** — a few hundred test cases don't take hours
- **BI-ready** — exports to SQLite (star schema) or Pandas DataFrames
- **CI/CD ready** — writes `$GITHUB_STEP_SUMMARY` automatically in GitHub Actions

## Install

```bash
pip install rag-score
```

## 60-second example

```python
import asyncio
from rag_score.core.dataset import load_dataset
from rag_score.core.runner import RunConfig, run_evaluation
from rag_score.adapters.base import CallableRetrieverAdapter, CallableGeneratorAdapter
from rag_score.metrics.retrieval.precision_at_k import PrecisionAtK
from rag_score.metrics.retrieval.mrr import MRR

test_cases = load_dataset("test_set.json")
retriever = CallableRetrieverAdapter(my_retriever)
generator = CallableGeneratorAdapter(my_generator)
metrics = [PrecisionAtK(k=5), MRR()]

report = asyncio.run(
    run_evaluation(test_cases, retriever, generator, metrics, RunConfig(run_id="run-1"))
)
```

See the [Quickstart](quickstart.md) for the CLI-based version, or the [API reference](api-reference.md) for full class/function docs.

## Why not Ragas / TruLens / DeepEval?

Those are excellent, more full-featured tools. `rag-score` exists for the case where you want something smaller: a library you can read end-to-end in an afternoon, with an adapter interface that doesn't assume you're using any particular framework, and a set of metrics that work with zero API keys before you ever reach for an LLM judge.
