"""
Context coverage — fraction of reference key facts found in contexts.

This metric computes what fraction of key facts from the reference answer
are present in the retrieved contexts. For simplicity, we treat key facts
as individual words (lowercased, alphanumeric only).

Pure set-membership math, no dependencies, no network calls — this is
one of the "offline-first" metrics that works with zero API keys.
"""

from __future__ import annotations

import re

from rag_score.core.metric import Metric
from rag_score.core.models import EvaluationCase


def _extract_key_facts(text: str) -> set[str]:
    """
    Extract key facts from text by splitting into alphanumeric words.
    """
    if not text:
        return set()
    # Convert to lowercase and extract alphanumeric words
    words = re.findall(r"\b[a-z0-9]+\b", text.lower())
    return set(words)


class ContextCoverage(Metric):
    def __init__(self) -> None:
        self.name = "context_coverage"

    def evaluate(self, case: EvaluationCase) -> MetricResult:
        reference = case.reference or ""
        if not reference:
            # No reference -> undefined, return 0.0
            score_value = 0.0

        # Get contexts from the case
        contexts = case.contexts
        if not contexts:
            score_value = 0.0

        # Extract key facts from reference
        reference_facts = _extract_key_facts(reference)
        if not reference_facts:
            # No key facts in reference -> undefined, return 0.0
            score_value = 0.0

        # Extract key facts from all contexts combined
        context_text = " ".join(contexts)
        context_facts = _extract_key_facts(context_text)

        # Compute coverage: fraction of reference facts found in contexts
        if not reference_facts:
            score_value = 0.0
        else:
            covered_facts = reference_facts.intersection(context_facts)
            score_value = len(covered_facts) / len(reference_facts)

        # Return a MetricResult
        return MetricResult(
            score_id="",  # We don't have an ID from the legacy metric
            evaluation_id="",  # We don't store this
            metric_name=self.name,
            score_value=score_value,
            judge_reasoning=None,  # These metrics don't provide reasoning by default
        )
