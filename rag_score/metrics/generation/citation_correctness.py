"""
Citation correctness — fraction of citation markers in the answer that
correspond to actual retrieved chunks.

When answers contain citation markers like [1], [2], etc., this metric
checks what fraction of those markers refer to actually retrieved chunks
(based on their position in the retrieved_context list).

Pure logic, no dependencies, no network calls — this is
one of the "offline-first" metrics that works with zero API keys.
"""

from __future__ import annotations

import re

from rag_score.core.metric import Metric
from rag_score.core.models import EvaluationCase


class CitationCorrectness(Metric):
    def __init__(self) -> None:
        self.name = "citation_correctness"

    def evaluate(self, case: EvaluationCase) -> MetricResult:
        answer = case.answer or ""
        if not answer:
            # No answer -> no citations, correctness is 0.0 (or undefined?)
            score_value = 0.0

        # Find all citation markers like [1], [2], etc.
        citation_pattern = r"\[(\d+)\]"
        citation_matches = re.findall(citation_pattern, answer)
        if not citation_matches:
            # No citations -> correctness is 1.0 (vacuously true)
            score_value = 1.0

        # Convert to integers (1-based indexing)
        citation_indices = []
        for match in citation_matches:
            try:
                idx = int(match)
                citation_indices.append(idx)
            except ValueError:
                # Invalid citation marker, ignore it
                continue

        if not citation_indices:
            # No valid citation markers
            score_value = 1.0

        # Get the number of retrieved chunks
        num_chunks = len(case.contexts)

        # Check each citation index: it should be between 1 and num_chunks (inclusive)
        correct_citations = 0
        for idx in citation_indices:
            if 1 <= idx <= num_chunks:
                correct_citations += 1

        # Return fraction of correct citations
        if len(citation_indices) > 0:
            score_value = correct_citations / len(citation_indices)
        else:
            score_value = 1.0

        # Return a MetricResult
        return MetricResult(
            score_id="",  # We don't have an ID from the legacy metric
            evaluation_id="",  # We don't store this
            metric_name=self.name,
            score_value=score_value,
            judge_reasoning=None,  # These metrics don't provide reasoning by default
        )
