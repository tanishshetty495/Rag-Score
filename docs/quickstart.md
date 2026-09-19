# Quickstart

This walks through the CLI-based workflow, the fastest way to get a first result.

## 1. Install

```bash
pip install ragmark
```

## 2. Describe your pipeline as two functions

```python
# my_pipeline.py
from rag_score.core.types import RetrievedChunk

async def my_retriever(query: str, top_k: int):
    # call your actual retriever here - FAISS, Pinecone, whatever
    return [RetrievedChunk(doc_id="doc_1", text="...")]

async def my_generator(query: str, context: list[RetrievedChunk]):
    # call your actual LLM here
    return "the generated answer"
```

The only contract: `RetrieverAdapter`-shaped functions take `(query, top_k)` and return a list of [`RetrievedChunk`][rag_score.core.types.RetrievedChunk]; generator functions take `(query, context)` and return a string.

## 3. Write a small test set

```json title="test_set.json"
[
  {
    "question": "What is the refund policy?",
    "expected_doc_ids": ["doc_1", "doc_2"]
  }
]
```

`expected_doc_ids` is only needed for the offline retrieval metrics (Precision@k, Recall@k, MRR, nDCG) - omit it if you're only running LLM-judge metrics.

## 4. Write a config

```json title="eval_config.json"
{
  "dataset": "test_set.json",
  "retriever": "my_pipeline:my_retriever",
  "generator": "my_pipeline:my_generator",
  "metrics": ["precision_at_5", "recall_at_5", "mrr", "ndcg_at_5"],
  "html_output": "report.html"
}
```

See the [Configuration reference](configuration.md) for every available field.

## 5. Run it

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

Open `report.html` - it's a single self-contained file, no server required.

## Next steps

- Don't have a test set yet? [Generate one from your own docs](configuration.md#rageval-synthesize) with `rageval synthesize`
- Add [LLM-judge metrics](metrics.md#llm-judge-metrics) for faithfulness and relevance scoring
- Plug in your [LangChain or LlamaIndex](adapters.md) pipeline directly instead of writing wrapper functions
- Export to [SQLite or Pandas](configuration.md#export-options) for tracking scores across runs
