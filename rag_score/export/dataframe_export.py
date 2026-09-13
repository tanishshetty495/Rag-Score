"""
Pandas DataFrame export - for Jupyter users who want to `df.describe()`
or plot results without touching SQLite at all.

pandas is an optional extra (pip install rag-score[bi]); this module
only raises ImportError at call time, not at package import time, so
installing rag-score core doesn't drag pandas in for people who never
call these functions.
"""

from __future__ import annotations

from rag_score.core.runner import RunReport
from rag_score.core.types import TestCase


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "pandas is required for DataFrame export. "
            "Install it with: pip install rag-score[bi]"
        ) from e
    return pd


def results_to_dataframe(report: RunReport):
    """One row per (test_case, evaluation) with raw I/O and latency -
    the fact_evaluations table, as a DataFrame."""
    pd = _require_pandas()
    rows = [r.model_dump(mode="json") for r in report.results]
    return pd.DataFrame(rows)


def scores_to_dataframe(report: RunReport):
    """One row per (evaluation, metric) - the fact_metric_scores table,
    as a DataFrame. Long/tidy format, easy to pivot or groupby."""
    pd = _require_pandas()
    rows = [s.model_dump(mode="json") for s in report.scores]
    return pd.DataFrame(rows)


def scores_to_wide_dataframe(report: RunReport):
    """One row per evaluation, one column per metric - the pivoted view
    people usually want for a quick df.describe() or a scatter plot of
    two metrics against each other."""
    # Calling for the side effect of raising early if pandas isn't
    # installed - scores_to_dataframe() below does its own import, but
    # failing fast here gives a clearer stack trace than deep inside it.
    _require_pandas()
    long_df = scores_to_dataframe(report)
    if long_df.empty:
        return long_df
    return long_df.pivot_table(
        index="evaluation_id", columns="metric_name", values="score_value"
    ).reset_index()


def full_report_dataframe(report: RunReport, test_cases: list[TestCase]):
    """Everything joined into one wide table: question, answer, latency,
    and every metric score side by side - the "just show me one table"
    view for quick exploration in a notebook."""
    pd = _require_pandas()

    results_df = results_to_dataframe(report)
    scores_wide_df = scores_to_wide_dataframe(report)
    tc_df = pd.DataFrame([tc.model_dump(mode="json") for tc in test_cases])

    merged = results_df.merge(
        tc_df[["test_case_id", "question", "ground_truth_answer", "expected_doc_ids"]],
        on="test_case_id",
        how="left",
    )
    if not scores_wide_df.empty:
        merged = merged.merge(scores_wide_df, on="evaluation_id", how="left")
    return merged
