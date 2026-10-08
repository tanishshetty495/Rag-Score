"""
Citation completeness — fraction of retrieved chunks that are cited in the answer.

When answers contain citation markers like [1], [2], etc., this metric
checks what fraction of retrieved chunks are actually cited (based on their
position in the retrieved_context list).

Pure logic, no dependencies, no network calls — this is
one of the "offline-first" metrics that works with zero API keys.
"""

from __future__ import annotations

import re

from rag_score.core.metric import Metric
from rag_score.core.models import EvaluationCase


class CitationCompleteness(Metric):
    def __init__(self) -> None:
        self.name = "citation_completeness"

    def evaluate(self, case: EvaluationCase) -> MetricResult:
        answer = case.answer or ""
        if not answer:
            # No answer -> no citations, completeness is 0.0
            score_value = 0.0

        # Find all citation markers like [1], [2], etc.
        citation_pattern = r"\[(\d+)\]"
        citation_matches = re.findall(citation_pattern, answer)
        if not citation_matches:
            # No citations -> completeness is 0.0
            score_value = 0.0

        # Convert to integers (1-based indexing) and deduplicate
        cited_indices = set()
        for match in citation_matches:
            try:
                idx = int(match)
                cited_indices.add(idx)
            except ValueError:
                # Invalid citation marker, ignore it
                continue

        if not cited_indices:
            # No valid citation markers
            score_value = 0.0

        # Get the number of retrieved chunks
        num_chunks = len(case.contexts)
        if num_chunks == 0:
            # No chunks -> completeness is 0.0 (can't cite what doesn't exist)
            score_value = 0.0

        # Count how many chunks are cited (indices 1 through num_chunks)
        cited_chunks = 0
        for idx in range(1, num_chunks + 1):
            if idx in cited_indices:
                cited_chunks += 1

        # Return fraction of chunks that are cited
        if num_chunks > 0:
            score_value = cited_chunks / num_chunks
        else:
            score_value = 0.0

        # Return a MetricResult
        return MetricResult(
            score_id="",  # We don't have an ID from the legacy metric
            evaluation_id="",  # We don't store this
            metric_name=self.name,
            score_value=score_value,
            judge_reasoning=None,  # These metrics don't provide reasoning by default
        )
