"""
Example: evaluating a LangChain RAG pipeline.

Uses real langchain_core objects (a BaseRetriever subclass and an LCEL
RunnableLambda standing in for a chain) to show exactly how
LangChainRetrieverAdapter and LangChainGeneratorAdapter plug into an
existing LangChain setup with zero rewriting of your retriever/chain.

In a real project, swap _DemoRetriever for your actual vectorstore
retriever (e.g. `vectorstore.as_retriever()`) and the RunnableLambda
for your actual chain (e.g. a full LCEL RAG chain or a ChatOpenAI call).

Requires: pip install ragmark[langchain]

Run it:
    python examples/langchain_example.py
"""

from __future__ import annotations

import asyncio

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda

from rag_score.adapters.langchain_adapter import (
    LangChainGeneratorAdapter,
    LangChainRetrieverAdapter,
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
    """Stands in for a real vectorstore retriever - swap this class
    for `your_vectorstore.as_retriever()` in a real project. The only
    thing LangChainRetrieverAdapter cares about is that this is a
    genuine langchain_core BaseRetriever."""

    def _get_relevant_documents(self, query, *, run_manager=None):
        query_lower = query.lower()
        if "refund" in query_lower:
            return [Document(page_content=_DOCS["doc_1"], metadata={"doc_id": "doc_1"})]
        if "shipping" in query_lower:
            return [Document(page_content=_DOCS["doc_2"], metadata={"doc_id": "doc_2"})]
        return []


def _demo_chain(inputs: dict) -> str:
    """Stands in for a real LLM chain - swap this for an actual
    ChatOpenAI/ChatAnthropic call or a full LCEL chain in a real
    project. Receives {"question": ..., "context": ...} by default."""
    return f"Based on the context: {inputs['context']}"


async def main() -> None:
    test_cases = [
        TestCase(question="What is the refund policy?", expected_doc_ids=["doc_1"]),
        TestCase(question="How long does shipping take?", expected_doc_ids=["doc_2"]),
    ]

    retriever_adapter = LangChainRetrieverAdapter(_DemoRetriever())
    generator_adapter = LangChainGeneratorAdapter(RunnableLambda(_demo_chain))
    metrics = [PrecisionAtK(k=3), MRR()]

    report = await run_evaluation(
        test_cases, retriever_adapter, generator_adapter, metrics,
        RunConfig(run_id="langchain-example"),
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
