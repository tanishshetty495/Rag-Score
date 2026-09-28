"""
Example: Multi-turn conversational evaluation with ContextCarryOver metric.

This example shows how to evaluate a RAG system on multi-turn conversations
where the generator needs to use conversation history to resolve references
like "the second one" or "that plan".

It includes:
- A fake retriever that returns fixed chunks
- A generator that opts in to history (by accepting a `history` kwarg)
- A fake judge that returns deterministic scores for demonstration
- Three multi-turn test cases, including the classic "second one" case
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from rag_score.adapters.base import GeneratorAdapter, RetrieverAdapter
from rag_score.core.runner import RunConfig, run_evaluation
from rag_score.core.types import RetrievedChunk, TestCase
from rag_score.judges.base import LLMJudge
from rag_score.metrics.generation.context_carry_over import ContextCarryOver

# ---------------------------------------------------------------------------
# Fake components
# ---------------------------------------------------------------------------


class FakeRetriever(RetrieverAdapter):
    """Returns the same two chunks regardless of query."""

    async def retrieve(
        self, query: str, top_k: int = 5
    ) -> list[RetrievedChunk]:
        # Ignore query and top_k for simplicity
        return [
            RetrievedChunk(
                text="The iPhone 15 features a 6.1-inch display, A16 Bionic chip, and 48MP main camera.",
                score=0.9,
            ),
            RetrievedChunk(
                text="The Pixel 8 offers a 6.2-inch Actua display, Tensor G3 chip, and 50MP main camera.",
                score=0.85,
            ),
        ]


@dataclass
class FakeGeneratorOptions:
    """Options to control the fake generator's behavior for demonstration."""
    # If True, the generator will use the history to resolve references.
    use_history: bool = True
    # If True, the generator will produce an answer that correctly resolves references.
    correct: bool = True


class FakeGenerator(GeneratorAdapter):
    """
    A generator that demonstrates opting in to conversation history.

    It accepts a `history` keyword argument in its generate method.
    For demonstration, it returns a fixed answer that either correctly
    or incorrectly resolves references based on the options.
    """

    def __init__(self, options: FakeGeneratorOptions | None = None):
        self.options = options or FakeGeneratorOptions()

    async def generate(
        self, query: str, context: list[RetrievedChunk], *, history: list[dict] | None = None
    ) -> str:
        """
        Generate an answer that may or may not use history.

        In a real system, the history would be used to resolve references.
        Here we hardcode answers for clarity.
        """
        if not self.options.use_history or history is None:
            # Ignore history - treat as single-turn
            return "I cannot compare phones without knowing which ones you mean."

        # For this fake, we look at the query to see if it contains references.
        query_lower = query.lower()
        if self.options.correct:
            # Produce an answer that correctly resolves references
            if "second one" in query_lower:
                return "The Pixel 8 has a 6.2-inch display, Tensor G3 chip, and 50MP main camera."
            elif "first one" in query_lower:
                return "The iPhone 15 features a 6.1-inch display, A16 Bionic chip, and 48MP main camera."
            else:
                return "The iPhone 15 and Pixel 8 are both great phones with different strengths."
        else:
            # Produce an answer that ignores or misresolves references
            if "second one" in query_lower:
                return "The iPhone 15 has a 6.1-inch display and A16 Bionic chip."
            else:
                return "I don't have enough information to answer that."


class FakeJudge(LLMJudge):
    """A fake judge that returns deterministic scores for demonstration."""

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        # For demonstration, we look for keywords in the user_prompt to decide the score.
        # In a real judge, this would be an LLM call.
        if "Pixel 8 has a 6.2-inch display" in user_prompt:
            return '{"score": 1.0, "reasoning": "Correctly resolved \'the second one\' to Pixel 8."}'
        if "iPhone 15 features a 6.1-inch display" in user_prompt:
            return '{"score": 1.0, "reasoning": "Correctly resolved \'the first one\' to iPhone 15."}'
        if "I cannot compare phones" in user_prompt or "don't have enough information" in user_prompt:
            return '{"score": 0.0, "reasoning": "Failed to resolve references from conversation history."}'
        # Default partial credit
        return '{"score": 0.5, "reasoning": "Partially resolved references."}'


# ---------------------------------------------------------------------------
# Test cases with conversation history
# ---------------------------------------------------------------------------

def get_multi_turn_test_cases() -> list[TestCase]:
    """Return a list of multi-turn test cases."""
    return [
        TestCase(
            question="What about the second one?",
            ground_truth_answer="The Pixel 8 has a 6.2-inch display, Tensor G3 chip, and 50MP main camera.",
            metadata={
                "conversation_history": [
                    {"role": "user", "content": "Compare the iPhone 15 and the Pixel 8"},
                    {
                        "role": "assistant",
                        "content": "The iPhone 15 features a 6.1-inch display, A16 Bionic chip, and 48MP main camera. The Pixel 8 offers a 6.2-inch Actua display, Tensor G3 chip, and 50MP main camera.",
                    },
                ]
            },
        ),
        TestCase(
            question="What about the first one?",
            ground_truth_answer="The iPhone 15 features a 6.1-inch display, A16 Bionic chip, and 48MP main camera.",
            metadata={
                "conversation_history": [
                    {"role": "user", "content": "Compare the iPhone 15 and the Pixel 8"},
                    {
                        "role": "assistant",
                        "content": "The iPhone 15 features a 6.1-inch display, A16 Bionic chip, and 48MP main camera. The Pixel 8 offers a 6.2-inch Actua display, Tensor G3 chip, and 50MP main camera.",
                    },
                ]
            },
        ),
        TestCase(
            question="What are the key differences?",
            ground_truth_answer="The iPhone 15 has a 6.1-inch display and A16 Bionic chip, while the Pixel 8 has a 6.2-inch display and Tensor G3 chip.",
            metadata={
                "conversation_history": [
                    {"role": "user", "content": "Compare the iPhone 15 and the Pixel 8"},
                    {
                        "role": "assistant",
                        "content": "The iPhone 15 features a 6.1-inch display, A16 Bionic chip, and 48MP main camera. The Pixel 8 offers a 6.2-inch Actua display, Tensor G3 chip, and 50MP main camera.",
                    },
                ]
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Run the evaluation
# ---------------------------------------------------------------------------

async def main() -> None:
    test_cases = get_multi_turn_test_cases()
    retriever = FakeRetriever()
    generator = FakeGenerator(FakeGeneratorOptions(use_history=True, correct=True))
    judge = FakeJudge()

    metrics = [ContextCarryOver(judge=judge)]

    config = RunConfig(
        run_id="multi-turn-demo",
        project_name="rag-score-demo",
        top_k=2,
        max_concurrency=2,
    )

    print("Running multi-turn evaluation with ContextCarryOver metric...")
    report = await run_evaluation(
        test_cases, retriever, generator, metrics, config
    )

    print(f"\nEvaluated {len(test_cases)} test cases.")
    for i, result in enumerate(report.results):
        print(f"\nTest case {i+1}: {result.test_case_id}")
        print(f"  Question: {test_cases[i].question}")
        print(f"  Generated answer: {result.generated_answer}")
        if result.error:
            print(f"  Error: {result.error}")

    print("\nMetric scores:")
    for score in report.scores:
        print(f"  {score.metric_name}: {score.score_value:.3f}")
        if score.judge_reasoning:
            print(f"    Reasoning: {score.judge_reasoning}")


if __name__ == "__main__":
    asyncio.run(main())