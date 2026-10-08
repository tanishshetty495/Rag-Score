"""
No-answer abstention — checks if the model correctly abstains when
the metadata tag "no_answer" is present.

When a test case has metadata["no_answer"] = True, the model should
abstain from answering (return an empty string or a specific abstention
phrase like "I don't know.").

Pure logic, no dependencies, no network calls — this is
one of the "offline-first" metrics that works with zero API keys.
"""

from __future__ import annotations

from rag_score.core.metric import Metric
from rag_score.core.models import EvaluationCase


class NoAnswerAbstention(Metric):
    def __init__(self) -> None:
        self.name = "no_answer_abstention"

    def evaluate(self, case: EvaluationCase) -> MetricResult:
        # Check if this is a no_answer case
        is_no_answer_case = case.metadata.get("no_answer", False)
        if not is_no_answer_case:
            # Not a no_answer case -> this metric doesn't apply, return 0.0
            # (or we could return 1.0? But the task says it checks correct abstention
            # for no_answer cases, so for non-no_answer cases we return 0.0 to indicate
            # the metric is not applicable)
            score_value = 0.0

        # This is a no_answer case, check if the model abstained
        answer = case.answer or ""

        # Define what counts as abstention:
        # 1. Empty string
        # 2. Common abstention phrases (case-insensitive)
        abstention_phrases = [
            "i don't know",
            "i do not know",
            "unknown",
            "i cannot answer",
            "i can't answer",
            "not enough information",
            "cannot be determined",
        ]

        answer_lower = answer.lower().strip()

        # Check for exact empty string
        if answer == "":
            score_value = 1.0

        # Check for abstention phrases
        else:
            for phrase in abstention_phrases:
                if phrase in answer_lower:
                    score_value = 1.0
                    break
            else:
                # If we get here, the model did not abstain
                score_value = 0.0

        # Return a MetricResult
        return MetricResult(
            score_id="",  # We don't have an ID from the legacy metric
            evaluation_id="",  # We don't store this
            metric_name=self.name,
            score_value=score_value,
            judge_reasoning=None,  # These metrics don't provide reasoning by default
        )
