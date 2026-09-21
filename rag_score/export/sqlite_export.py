"""
SQLite star-schema export.

Pure stdlib sqlite3 - no ORM, no pandas dependency required for this
module itself (dataframe_export.py is the pandas-dependent sibling).
Writing here means a single .db file that BI tools (Power BI, Superset,
Metabase) or a plain SQL query can point at directly, with runs kept
separate so historical comparisons across evaluation runs are just a
WHERE clause away.

Schema:
    dim_runs           - one row per evaluation run (config, timestamp)
    dim_test_cases     - one row per question in a dataset (deduped by test_case_id)
    fact_evaluations   - one row per (run, test_case) pairing - the actual I/O
    fact_metric_scores - one row per (evaluation, metric) - the scored numbers
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from rag_score.core.runner import RunReport
from rag_score.core.types import (
    DimRun,
    DimTestCase,
    EvalResult,
    FactEvaluation,
    FactMetricScore,
    MetricScore,
    TestCase,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS dim_runs (
    run_id            TEXT PRIMARY KEY,
    run_timestamp     TEXT NOT NULL,
    project_name      TEXT NOT NULL,
    dataset_name      TEXT NOT NULL,
    retriever_config  TEXT,
    generator_config  TEXT,
    environment       TEXT
);

CREATE TABLE IF NOT EXISTS dim_test_cases (
    test_case_id        TEXT PRIMARY KEY,
    dataset_name         TEXT NOT NULL,
    question              TEXT NOT NULL,
    ground_truth_answer  TEXT,
    expected_doc_ids     TEXT,
    difficulty_category  TEXT
);

CREATE TABLE IF NOT EXISTS fact_evaluations (
    evaluation_id         TEXT PRIMARY KEY,
    run_id                TEXT NOT NULL REFERENCES dim_runs(run_id),
    test_case_id          TEXT NOT NULL REFERENCES dim_test_cases(test_case_id),
    retrieved_context     TEXT,
    generated_answer      TEXT,
    retrieval_latency_ms  REAL,
    generation_latency_ms REAL,
    total_tokens          INTEGER,
    estimated_cost_usd    REAL
);

CREATE TABLE IF NOT EXISTS fact_metric_scores (
    score_id         TEXT PRIMARY KEY,
    evaluation_id    TEXT NOT NULL REFERENCES fact_evaluations(evaluation_id),
    metric_name      TEXT NOT NULL,
    score_value      REAL NOT NULL,
    judge_reasoning  TEXT
);

CREATE INDEX IF NOT EXISTS idx_fact_evaluations_run_id
    ON fact_evaluations(run_id);
CREATE INDEX IF NOT EXISTS idx_fact_metric_scores_evaluation_id
    ON fact_metric_scores(evaluation_id);
CREATE INDEX IF NOT EXISTS idx_fact_metric_scores_metric_name
    ON fact_metric_scores(metric_name);
"""


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)


def _to_dim_test_case(tc: TestCase) -> DimTestCase:
    return DimTestCase(
        test_case_id=tc.test_case_id,
        dataset_name=tc.dataset_name,
        question=tc.question,
        ground_truth_answer=tc.ground_truth_answer,
        expected_doc_ids=json.dumps(tc.expected_doc_ids),
        difficulty_category=tc.difficulty_category,
    )


def _to_fact_evaluation(result: EvalResult) -> FactEvaluation:
    return FactEvaluation(
        evaluation_id=result.evaluation_id,
        run_id=result.run_id,
        test_case_id=result.test_case_id,
        retrieved_context=json.dumps(
            [chunk.model_dump(mode="json") for chunk in result.retrieved_context]
        ),
        generated_answer=result.generated_answer,
        retrieval_latency_ms=result.retrieval_latency_ms,
        generation_latency_ms=result.generation_latency_ms,
        total_tokens=result.total_tokens,
        estimated_cost_usd=result.estimated_cost_usd,
    )


def _to_fact_metric_score(score: MetricScore) -> FactMetricScore:
    return FactMetricScore(
        score_id=score.score_id,
        evaluation_id=score.evaluation_id,
        metric_name=score.metric_name,
        score_value=score.score_value,
        judge_reasoning=score.judge_reasoning,
    )


def export_to_sqlite(
    db_path: str | Path,
    run: DimRun,
    test_cases: list[TestCase],
    report: RunReport,
) -> None:
    """Write one full evaluation run (run metadata, test cases, raw I/O,
    and metric scores) into the star schema at db_path. Safe to call
    repeatedly against the same file - runs accumulate rather than
    overwrite, since dim_runs/dim_test_cases use INSERT OR REPLACE on
    their natural keys and fact tables key off unique evaluation_id/score_id.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    try:
        _ensure_schema(conn)

        conn.execute(
            """
            INSERT OR REPLACE INTO dim_runs
                (run_id, run_timestamp, project_name, dataset_name,
                 retriever_config, generator_config, environment)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.run_id,
                run.run_timestamp.isoformat(),
                run.project_name,
                run.dataset_name,
                run.retriever_config,
                run.generator_config,
                run.environment,
            ),
        )

        dim_rows = [_to_dim_test_case(tc) for tc in test_cases]
        conn.executemany(
            """
            INSERT OR REPLACE INTO dim_test_cases
                (test_case_id, dataset_name, question, ground_truth_answer,
                 expected_doc_ids, difficulty_category)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    r.test_case_id,
                    r.dataset_name,
                    r.question,
                    r.ground_truth_answer,
                    r.expected_doc_ids,
                    r.difficulty_category,
                )
                for r in dim_rows
            ],
        )

        fact_eval_rows = [_to_fact_evaluation(r) for r in report.results]
        conn.executemany(
            """
            INSERT OR REPLACE INTO fact_evaluations
                (evaluation_id, run_id, test_case_id, retrieved_context,
                 generated_answer, retrieval_latency_ms, generation_latency_ms,
                 total_tokens, estimated_cost_usd)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    r.evaluation_id,
                    r.run_id,
                    r.test_case_id,
                    r.retrieved_context,
                    r.generated_answer,
                    r.retrieval_latency_ms,
                    r.generation_latency_ms,
                    r.total_tokens,
                    r.estimated_cost_usd,
                )
                for r in fact_eval_rows
            ],
        )

        fact_score_rows = [_to_fact_metric_score(s) for s in report.scores]
        conn.executemany(
            """
            INSERT OR REPLACE INTO fact_metric_scores
                (score_id, evaluation_id, metric_name, score_value, judge_reasoning)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (r.score_id, r.evaluation_id, r.metric_name, r.score_value, r.judge_reasoning)
                for r in fact_score_rows
            ],
        )

        conn.commit()
    finally:
        conn.close()
