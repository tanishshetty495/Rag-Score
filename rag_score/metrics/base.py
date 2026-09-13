"""
Metric base class.

score() is async even for pure-math metrics like precision@k, so that
the runner can treat every metric uniformly (gather() across all of
them) regardless of whether a given metric internally calls an LLM
judge or a local embedding model. Cheap metrics just return
immediately — there's no meaningful overhead from being a coroutine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag_score.core.types import EvalResult, TestCase


class Metric(ABC):
    """Base class for every retrieval or generation metric.

    Subclasses set `name` and implement `score()`. `requires_api_key`
    lets the CLI/report warn users up front if a chosen metric needs an
    LLM judge or network access, keeping the "offline-first" promise
    explicit rather than a surprise at runtime.
    """

    name: str = "base_metric"
    requires_api_key: bool = False

    @abstractmethod
    async def score(self, test_case: TestCase, result: EvalResult) -> float:
        """Return a float score for this one TestCase/EvalResult pair.

        Implementations should return 0.0 (not raise) when the inputs
        are insufficient to compute the metric (e.g. no expected_doc_ids),
        so that a single malformed row can't crash a whole run. Reserve
        exceptions for genuine bugs.
        """
        raise NotImplementedError

    async def score_with_reasoning(
        self, test_case: TestCase, result: EvalResult
    ) -> tuple[float, str | None]:
        """Like score(), but also returns a human-readable justification
        when one is available (LLM-judge metrics populate this from the
        judge's own explanation; pure-math metrics like precision_at_k
        have nothing to add and just return None).

        The runner calls this instead of score() directly so that
        reasoning flows into MetricScore.judge_reasoning without every
        metric needing to implement it - most never will.
        """
        score = await self.score(test_case, result)
        return score, None

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<Metric name={self.name!r} requires_api_key={self.requires_api_key}>"
