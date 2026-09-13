"""
Example: evaluating a "raw" RAG pipeline - no framework at all.

This is the simplest possible integration: two plain async functions,
no adapters to subclass, no framework objects to construct. If you've
built your own retrieval/generation code from scratch (e.g. following
along with a "RAG from scratch" tutorial), this is exactly the shape
rag-score expects.

Run it:
    python examples/raw_example.py
"""

from __future__ import annotations

import asyncio

from rag_score.adapters.base import CallableGeneratorAdapter, CallableRetrieverAdapter
from rag_score.core.runner import RunConfig, run_evaluation
from rag_score.core.types import RetrievedChunk, TestCase
from rag_score.metrics.retrieval.mrr import MRR
from rag_score.metrics.retrieval.precision_at_k import PrecisionAtK
from rag_score.metrics.retrieval.recall_at_k import RecallAtK

# A tiny in-memory "document store" standing in for a real vector DB.
_DOCS = {
    "doc_1": "Our refund policy allows refunds within 30 days of purchase with a valid receipt.",
    "doc_2": "Shipping typically takes 3-5 business days within the continental US.",
    "doc_3": "Our support team is available Monday-Friday, 9am-6pm EST.",
    "doc_4": "Gift cards do not expire and can be used for any purchase.",
}


# Naive stopword list - just enough to keep the demo retriever's
# keyword overlap meaningful instead of tying on "is"/"the"/"a".
_STOPWORDS = {"what", "is", "the", "a", "an", "how", "do", "does", "for", "of", "in", "on"}


async def my_retriever(query: str, top_k: int = 5) -> list[RetrievedChunk]:
    """A deliberately naive keyword-overlap 'retriever' - swap this out
    for your real FAISS/Pinecone/Chroma/etc. lookup. The only contract
    is: take a query and top_k, return a list of RetrievedChunk."""
    query_words = set(query.lower().rstrip("?").split()) - _STOPWORDS
    scored = []
    for doc_id, text in _DOCS.items():
        doc_words = set(text.lower().replace(".", "").replace(",", "").split())
        overlap = len(query_words & doc_words)
        if overlap > 0:
            scored.append((overlap, doc_id, text))
    scored.sort(reverse=True)
    return [
        RetrievedChunk(doc_id=doc_id, text=text)
        for _, doc_id, text in scored[:top_k]
    ]


async def my_generator(query: str, context: list[RetrievedChunk]) -> str:
    """Stand-in for a real LLM call - swap this for an actual API call
    to OpenAI/Anthropic/your local model."""
    if not context:
        return "I don't have enough information to answer that."
    return f"Based on the available information: {context[0].text}"


async def main() -> None:
    test_cases = [
        TestCase(
            question="What is the refund policy?",
            expected_doc_ids=["doc_1"],
        ),
        TestCase(
            question="How long does shipping take?",
            expected_doc_ids=["doc_2"],
        ),
        TestCase(
            question="Do gift cards expire?",
            expected_doc_ids=["doc_4"],
        ),
    ]

    retriever = CallableRetrieverAdapter(my_retriever)
    generator = CallableGeneratorAdapter(my_generator)
    metrics = [PrecisionAtK(k=3), RecallAtK(k=3), MRR()]

    report = await run_evaluation(
        test_cases, retriever, generator, metrics, RunConfig(run_id="raw-example")
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
