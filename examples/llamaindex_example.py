"""
Example: evaluating a LlamaIndex RAG pipeline.

Uses real llama_index.core objects (a BaseRetriever subclass and a
bare LLM-like object with .acomplete) to show how
LlamaIndexRetrieverAdapter and LlamaIndexGeneratorAdapter plug into an
existing LlamaIndex setup.

In a real project, swap _DemoRetriever for
`your_index.as_retriever(similarity_top_k=5)` and the demo LLM for a
real one (e.g. `OpenAI()` from llama_index.llms.openai). If you're
using a full query engine instead of a bare LLM, pass
`your_index.as_query_engine()` to LlamaIndexGeneratorAdapter instead -
it auto-detects which mode to use.

Requires: pip install ragmark[llamaindex]

Run it:
    python examples/llamaindex_example.py
"""

from __future__ import annotations

import asyncio

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode

from rag_score.adapters.llamaindex_adapter import (
    LlamaIndexGeneratorAdapter,
    LlamaIndexRetrieverAdapter,
)
from rag_score.core.runner import RunConfig, run_evaluation
from rag_score.core.types import TestCase
from rag_score.metrics.retrieval.mrr import MRR
from rag_score.metrics.retrieval.precision_at_k import PrecisionAtK

_DOCS = {
    "doc_1": "Our refund policy allows refunds within 30 days of purchase.",
    "doc_2": "Shipping typically takes 3-5 business days within the continental US.",
}


class _DemoRetriever(BaseRetriever):
    """Stands in for `your_index.as_retriever()` - the only thing
    LlamaIndexRetrieverAdapter cares about is that this is a genuine
    llama_index.core BaseRetriever."""

    def _retrieve(self, query_bundle):
        query_lower = query_bundle.query_str.lower()
        if "refund" in query_lower:
            return [NodeWithScore(node=TextNode(text=_DOCS["doc_1"], id_="doc_1"), score=0.9)]
        if "shipping" in query_lower:
            return [NodeWithScore(node=TextNode(text=_DOCS["doc_2"], id_="doc_2"), score=0.9)]
        return []


class _DemoLLM:
    """Stands in for a real LlamaIndex LLM (e.g. llama_index.llms.openai.OpenAI) -
    swap this for a real one in a real project. Only needs .acomplete."""

    async def acomplete(self, prompt: str) -> str:
        return f"Answer derived from: {prompt.splitlines()[1] if len(prompt.splitlines()) > 1 else prompt}"


async def main() -> None:
    test_cases = [
        TestCase(question="What is the refund policy?", expected_doc_ids=["doc_1"]),
        TestCase(question="How long does shipping take?", expected_doc_ids=["doc_2"]),
    ]

    retriever_adapter = LlamaIndexRetrieverAdapter(_DemoRetriever())
    generator_adapter = LlamaIndexGeneratorAdapter(_DemoLLM())  # bare-LLM mode
    metrics = [PrecisionAtK(k=3), MRR()]

    report = await run_evaluation(
        test_cases, retriever_adapter, generator_adapter, metrics,
        RunConfig(run_id="llamaindex-example"),
    )

    print(f"Evaluated {len(report.results)} test cases\n")
    for result in report.results:
        question = next(
            tc.question for tc in test_cases if tc.test_case_id == result.test_case_id
        )
        print(f"Q: {question}")
        print(f"A: {result.generated_answer}\n")

    by_metric: dict[str, list[float]] = {}
    for s in report.scores:
        by_metric.setdefault(s.metric_name, []).append(s.score_value)
    print("Average scores:")
    for name, values in by_metric.items():
        print(f"  {name}: {sum(values) / len(values):.3f}")


if __name__ == "__main__":
    asyncio.run(main())
