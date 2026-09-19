# Metrics reference

## Offline metrics (zero API keys required)

These run entirely locally - no network access, no cost. Every one is disabled by default in any given run; list the ones you want by name in your config's `metrics` array or pass instances directly in Python.

| Metric name | Class | What it measures |
|---|---|---|
| `precision_at_<k>` | [`PrecisionAtK`][rag_score.metrics.retrieval.precision_at_k.PrecisionAtK] | Of the top-k retrieved chunks, what fraction are relevant (their `doc_id` is in `expected_doc_ids`)? |
| `recall_at_<k>` | [`RecallAtK`][rag_score.metrics.retrieval.recall_at_k.RecallAtK] | Of all relevant chunks, what fraction did top-k retrieval surface? |
| `mrr` | [`MRR`][rag_score.metrics.retrieval.mrr.MRR] | Reciprocal rank of the first relevant hit (1.0 if it's first, 0.5 if second, etc.) |
| `ndcg_at_<k>` | [`NDCG`][rag_score.metrics.retrieval.ndcg.NDCG] | Ranking quality - rewards relevant results appearing earlier, normalized against the ideal ordering |

`<k>` is any positive integer, e.g. `precision_at_5`, `recall_at_10`.

All four require `expected_doc_ids` on your `TestCase` - without it, they return `0.0` (not an error), since there's nothing to compare against.

## LLM-judge metrics

These call an LLM to assess quality that can't be reduced to set membership - whether an answer is actually grounded in its context, for instance. They need a `judge` configured (see [Configuration reference](configuration.md#judge)).

| Metric name | Class | What it measures | Needs context? | Needs answer? |
|---|---|---|---|---|
| `faithfulness` | [`Faithfulness`][rag_score.metrics.generation.faithfulness.Faithfulness] | Is every claim in the answer supported by the retrieved context? | Yes | Yes |
| `answer_relevance` | [`AnswerRelevance`][rag_score.metrics.generation.answer_relevance.AnswerRelevance] | Does the answer actually address the question asked? | No | Yes |
| `context_precision` | [`ContextPrecision`][rag_score.metrics.generation.context_precision.ContextPrecision] | Of the retrieved chunks, what fraction are relevant? | Yes | No |

A metric returns `0.0` without calling the judge when its required inputs are missing (e.g. `faithfulness` on an empty answer) - this avoids wasting an API call on input that can't be meaningfully judged.

Every LLM-judge verdict includes a one-sentence `reasoning` string alongside its score, visible as a tooltip on each score in the HTML report and stored in the `judge_reasoning` column of the SQLite export.

### Faithfulness vs. Answer Relevance

These two metrics are deliberately independent, and reading them together tells you more than either alone:

- **High faithfulness, low relevance**: the answer is grounded in the retrieved context, but that context was the wrong context - a retrieval problem.
- **Low faithfulness, high relevance**: the answer addresses the question but invents details not in the context - a generation/hallucination problem.

## Local ML metrics (no LLM, no API, no server)

```bash
pip install ragmark[local-ml]
```

| Metric name | Class | What it measures |
|---|---|---|
| `local_faithfulness` | [`LocalSemanticFaithfulness`][rag_score.metrics.generation.local_faithfulness.LocalSemanticFaithfulness] | Cosine similarity between the answer and retrieved context |
| `local_answer_relevance` | [`LocalSemanticAnswerRelevance`][rag_score.metrics.generation.local_answer_relevance.LocalSemanticAnswerRelevance] | Cosine similarity between the question and answer |

These use a local embedding model ([sentence-transformers](https://www.sbert.net/), `all-MiniLM-L6-v2` by default) instead of an LLM judge - no API key, no inference server, and no network access after the model's first download. They're a weaker signal than the LLM-judge equivalents (semantic similarity isn't the same as logical entailment or factual correctness) but are free and fast enough to run on every single test case as a first pass.

Override the model with `encoder_model` in your config, or pass `model_name=` directly in Python:

```python
from rag_score.metrics.generation.local_faithfulness import LocalSemanticFaithfulness
metric = LocalSemanticFaithfulness(model_name="all-mpnet-base-v2")  # larger, more accurate, slower
```

## Writing a custom metric

See [Contributing](https://github.com/tanishshetty495/Rag-Score/blob/main/CONTRIBUTING.md#adding-a-new-metric) for the full guide. In short: subclass [`Metric`][rag_score.metrics.base.Metric], implement `async def score(...)`, and register it in the CLI's metric-name resolution if you want it usable from a config file.
