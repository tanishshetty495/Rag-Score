"""
Tests for the ContextCarryOver metric.
"""

from __future__ import annotations

import asyncio
import json

from rag_score.adapters.base import GeneratorAdapter, RetrieverAdapter
from rag_score.core.dataset import load_dataset
from rag_score.core.runner import RunConfig, _run_single
from rag_score.core.types import EvalResult, TestCase
from rag_score.judges.base import LLMJudge
from rag_score.metrics.generation.context_carry_over import ContextCarryOver


class FakeJudge(LLMJudge):
    """A fake judge that records calls and returns a deterministic score."""

    def __init__(self, score: float = 0.8, reasoning: str = "Fake reasoning", cache=None):
        self.score = score
        self.reasoning = reasoning
        self.called = False
        self.system_prompt: str | None = None
        self.user_prompt: str | None = None
        super().__init__(cache=cache)

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.called = True
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        return f'{{"score": {self.score}, "reasoning": "{self.reasoning}"}}'


def make_test_case(question: str, history: list[dict] | None = None) -> TestCase:
    """Helper to create a TestCase with required fields."""
    return TestCase(
        question=question,
        metadata={"conversation_history": history} if history is not None else {},
    )


def make_eval_result(answer: str) -> EvalResult:
    """Helper to create an EvalResult with required fields."""
    return EvalResult(
        run_id="test-run",
        test_case_id="test-tc",
        generated_answer=answer,
    )


async def test_context_carry_over_with_history_scores_from_judge():
    """When history is present and valid, the metric calls the judge and returns its score."""
    judge = FakeJudge(score=0.75, reasoning="Good context carry over")
    metric = ContextCarryOver(judge=judge)

    test_case = make_test_case(
        "What about the second one?",
        [
            {"role": "user", "content": "Compare the iPhone 15 and the Pixel 8"},
            {"role": "assistant", "content": "The iPhone 15 has ... The Pixel 8 has ..."},
        ],
    )
    result = make_eval_result("The Pixel 8 has a great camera.")

    # Run the metric
    score = await metric.score(test_case, result)

    # Assertions
    assert judge.called
    assert score == 0.75
    # Verify the judge saw the history, question, and answer in the user prompt
    assert judge.user_prompt is not None
    assert "Compare the iPhone 15 and the Pixel 8" in judge.user_prompt
    assert "The iPhone 15 has ... The Pixel 8 has ..." in judge.user_prompt
    assert "What about the second one?" in judge.user_prompt
    assert "The Pixel 8 has a great camera." in judge.user_prompt


async def test_context_carry_over_without_history_returns_zero_and_does_not_call_judge():
    """When history is missing, the metric returns 0.0 and does not call the judge."""
    judge = FakeJudge()
    metric = ContextCarryOver(judge=judge)

    test_case = make_test_case("What is the answer?")
    result = make_eval_result("Some answer")

    score = await metric.score(test_case, result)

    assert not judge.called
    assert score == 0.0
    score, reasoning = await metric.score_with_reasoning(test_case, result)
    assert score == 0.0
    assert reasoning == "No usable conversation history in metadata."


async def test_context_carry_over_with_empty_history_returns_zero_and_does_not_call_judge():
    """When history is an empty list, the metric returns 0.0 and does not call the judge."""
    judge = FakeJudge()
    metric = ContextCarryOver(judge=judge)

    test_case = make_test_case("What is the answer?", [])
    result = make_eval_result("Some answer")

    score = await metric.score(test_case, result)

    assert not judge.called
    assert score == 0.0
    score, reasoning = await metric.score_with_reasoning(test_case, result)
    assert score == 0.0
    assert reasoning == "No usable conversation history in metadata."


async def test_context_carry_over_with_wrong_type_history_returns_zero_and_does_not_call_judge():
    """When history is not a list, the metric returns 0.0 and does not call the judge."""
    judge = FakeJudge()
    metric = ContextCarryOver(judge=judge)

    test_case = TestCase(
        question="What is the answer?",
        metadata={"conversation_history": "not a list"},
    )
    result = make_eval_result("Some answer")

    score = await metric.score(test_case, result)

    assert not judge.called
    assert score == 0.0
    score, reasoning = await metric.score_with_reasoning(test_case, result)
    assert score == 0.0
    assert reasoning == "No usable conversation history in metadata."


async def test_context_carry_over_with_malformed_history_entries_returns_zero_and_does_not_call_judge():
    """When history entries are not dicts or missing required fields, the metric returns 0.0 and does not call the judge."""
    judge = FakeJudge()
    metric = ContextCarryOver(judge=judge)

    # Entry is not a dict
    test_case = TestCase(
        question="What is the answer?",
        metadata={"conversation_history": ["not a dict"]},
    )
    result = make_eval_result("Some answer")
    score = await metric.score(test_case, result)
    assert not judge.called
    assert score == 0.0
    score, reasoning = await metric.score_with_reasoning(test_case, result)
    assert score == 0.0
    assert reasoning == "History entry 0 is not a dictionary."

    # Entry missing content
    test_case = TestCase(
        question="What is the answer?",
        metadata={"conversation_history": [{"role": "user"}]},  # missing content
    )
    score = await metric.score(test_case, result)
    assert not judge.called
    assert score == 0.0
    score, reasoning = await metric.score_with_reasoning(test_case, result)
    assert score == 0.0
    assert reasoning == "History entry 0 missing or invalid 'role' or 'content'."

    # Entry missing role
    test_case = TestCase(
        question="What is the answer?",
        metadata={"conversation_history": [{"content": "hello"}]},  # missing role
    )
    score = await metric.score(test_case, result)
    assert not judge.called
    assert score == 0.0
    score, reasoning = await metric.score_with_reasoning(test_case, result)
    assert score == 0.0
    assert reasoning == "History entry 0 missing or invalid 'role' or 'content'."


async def test_context_carry_over_reasoning_is_passed_through():
    """The reasoning string from the judge is returned by score_with_reasoning."""
    judge = FakeJudge(reasoning="This is the reasoning")
    metric = ContextCarryOver(judge=judge)

    test_case = make_test_case(
        "What about the second one?",
        [
            {"role": "user", "content": "Compare X and Y"},
            {"role": "assistant", "content": "X is ... Y is ..."},
        ],
    )
    result = make_eval_result("Y is better.")

    score, reasoning = await metric.score_with_reasoning(test_case, result)

    assert judge.called
    assert score == 0.8
    assert reasoning == "This is the reasoning"


async def test_context_carry_over_returns_zero_when_no_generated_answer():
    """If there is no generated answer, the metric returns 0.0 without calling the judge."""
    judge = FakeJudge()
    metric = ContextCarryOver(judge=judge)

    test_case = make_test_case(
        "What is the answer?",
        [
            {"role": "user", "content": "Compare X and Y"},
            {"role": "assistant", "content": "X is ... Y is ..."},
        ],
    )
    result = EvalResult(
        run_id="test-run",
        test_case_id="test-tc",
        # generated_answer is None by default
    )

    score = await metric.score(test_case, result)

    assert not judge.called
    assert score == 0.0
    score, reasoning = await metric.score_with_reasoning(test_case, result)
    assert score == 0.0
    assert reasoning == "No generated answer to evaluate."


# ---------------------------------------------------------------------------
# Runner integration tests
# ---------------------------------------------------------------------------


async def test_runner_passes_history_to_opt_in_generator():
    """The runner passes history to a generator that declares a `history` kwarg."""
    class OptInGenerator(GeneratorAdapter):
        def __init__(self):
            self.received_history = None

        async def generate(self, query: str, context: list, *, history: list | None = None) -> str:
            self.received_history = history
            return "generated answer"

    class DummyRetriever(RetrieverAdapter):
        async def retrieve(self, query: str, top_k: int = 5) -> list:
            return []

    test_case = make_test_case(
        "What about the second one?",
        [
            {"role": "user", "content": "Compare A and B"},
            {"role": "assistant", "content": "A is ... B is ..."},
        ],
    )
    config = RunConfig(
        run_id="test-run",
        project_name="test",
        top_k=2,
        max_concurrency=1,
        continue_on_error=True,
        telemetry=None,
    )
    semaphore = asyncio.Semaphore(1)

    generator = OptInGenerator()
    retriever = DummyRetriever()

    result = await _run_single(test_case, retriever, generator, config, semaphore, on_progress=None)

    assert generator.received_history == [
        {"role": "user", "content": "Compare A and B"},
        {"role": "assistant", "content": "A is ... B is ..."},
    ]
    assert result.generated_answer == "generated answer"


async def test_runner_does_not_pass_history_to_legacy_generator():
    """The runner does not pass history to a generator that lacks a `history` kwarg."""
    class LegacyGenerator(GeneratorAdapter):
        def __init__(self):
            self.call_count = 0

        async def generate(self, query: str, context: list) -> str:  # No history parameter
            self.call_count += 1
            return "generated answer"

    class DummyRetriever(RetrieverAdapter):
        async def retrieve(self, query: str, top_k: int = 5) -> list:
            return []

    test_case = make_test_case(
        "What about the second one?",
        [
            {"role": "user", "content": "Compare A and B"},
            {"role": "assistant", "content": "A is ... B is ..."},
        ],
    )
    config = RunConfig(
        run_id="test-run",
        project_name="test",
        top_k=2,
        max_concurrency=1,
        continue_on_error=True,
        telemetry=None,
    )
    semaphore = asyncio.Semaphore(1)

    generator = LegacyGenerator()
    retriever = DummyRetriever()

    result = await _run_single(test_case, retriever, generator, config, semaphore, on_progress=None)

    assert generator.call_count == 1
    # The generator should not have received a history keyword argument
    # Since we didn't capture kwargs, we can only assert it was called with two positional args.
    # The test passes if no TypeError is raised (i.e., the generator was called correctly).
    assert result.generated_answer == "generated answer"


async def test_runner_does_not_pass_history_when_test_case_lacks_history():
    """The runner does not pass history to an opt-in generator when the test case has no history."""
    class OptInGenerator(GeneratorAdapter):
        def __init__(self):
            self.received_history = None

        async def generate(self, query: str, context: list, *, history: list | None = None) -> str:
            self.received_history = history
            return "generated answer"

    class DummyRetriever(RetrieverAdapter):
        async def retrieve(self, query: str, top_k: int = 5) -> list:
            return []

    test_case = TestCase(question="What is the answer?")  # No metadata, hence no history
    config = RunConfig(
        run_id="test-run",
        project_name="test",
        top_k=2,
        max_concurrency=1,
        continue_on_error=True,
        telemetry=None,
    )
    semaphore = asyncio.Semaphore(1)

    generator = OptInGenerator()
    retriever = DummyRetriever()

    result = await _run_single(test_case, retriever, generator, config, semaphore, on_progress=None)

    # When there is no history, the runner should pass None for history (or not pass it at all).
    # Our generator's signature makes history a kwarg that defaults to None, so we expect None.
    assert generator.received_history is None
    assert result.generated_answer == "generated answer"


# ---------------------------------------------------------------------------
# load_dataset round-trip test
# ---------------------------------------------------------------------------


def test_load_dataset_roundtrips_conversation_history_metadata(tmp_path):
    """load_dataset preserves the conversation_history field in metadata."""
    data = [
        {
            "question": "What about the second one?",
            "metadata": {
                "conversation_history": [
                    {"role": "user", "content": "Compare X and Y"},
                    {"role": "assistant", "content": "X is ... Y is ..."},
                ]
            },
        },
        {
            "question": "What is the answer?",
            "metadata": {},  # no history
        },
    ]
    path = tmp_path / "test_set.json"
    path.write_text(json.dumps(data))

    cases = load_dataset(path)

    assert len(cases) == 2
    assert cases[0].question == "What about the second one?"
    assert cases[0].metadata["conversation_history"] == [
        {"role": "user", "content": "Compare X and Y"},
        {"role": "assistant", "content": "X is ... Y is ..."},
    ]
    assert cases[1].question == "What is the answer?"
    assert cases[1].metadata == {}  # history key should not be present if not in original