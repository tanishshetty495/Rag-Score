"""
Types for agentic trajectory evaluation.

Deliberately separate from core/types.py rather than bolted onto
TestCase/EvalResult: a multi-step agent run (a sequence of tool calls
before a final answer) isn't a retrieve-then-generate pipeline with
extra fields tacked on, it's a different shape of thing to evaluate.
Keeping them apart means the existing single-shot types stay simple,
and trajectory evaluation can evolve its own conventions without
either side having to carry fields the other doesn't need.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


def _new_id() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ToolCall(BaseModel):
    """One step in an agent's trajectory - a single tool invocation."""

    tool_name: str
    tool_input: dict[str, Any] = Field(default_factory=dict)
    tool_output: str | None = None
    latency_ms: float | None = None


class TrajectoryTestCase(BaseModel):
    """A single question for agentic evaluation.

    expected_tool_sequence is optional (like expected_doc_ids on the
    single-shot TestCase) - metrics that need it return 0.0 without
    it rather than raising, so a dataset that only cares about the
    final answer doesn't need to specify tool expectations at all.
    """

    __test__ = False  # tell pytest this isn't a test class despite the name

    test_case_id: str = Field(default_factory=_new_id)
    dataset_name: str = "default"
    question: str
    expected_tool_sequence: list[str] = Field(default_factory=list)
    expected_final_answer: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrajectoryEvalResult(BaseModel):
    """The full recorded output of running one TrajectoryTestCase
    through an agent once - every tool call made, in order, plus the
    final answer."""

    evaluation_id: str = Field(default_factory=_new_id)
    run_id: str
    test_case_id: str

    tool_calls: list[ToolCall] = Field(default_factory=list)
    final_answer: str | None = None
    total_latency_ms: float | None = None

    error: str | None = None  # populated if the agent raised; metrics should skip


# ---------------------------------------------------------------------------
# Dataset loading - mirrors core/dataset.py's load_dataset but for the
# trajectory TestCase shape
# ---------------------------------------------------------------------------

def load_trajectory_dataset(
    path: str | Path, dataset_name: str | None = None
) -> list[TrajectoryTestCase]:
    """Load a list of TrajectoryTestCase from a .json or .yaml/.yml file.

    Expected JSON shape:
        [
          {
            "question": "...",
            "expected_tool_sequence": ["search", "calculator"],  # optional
            "expected_final_answer": "..."                       # optional
          },
          ...
        ]
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    inferred_name = dataset_name or path.stem

    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "PyYAML is required to load .yaml datasets. "
                "Install it with: pip install rag-score[yaml]"
            ) from e
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    else:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError(  # noqa: TRY004 - intentional, matches load_dataset's convention
            f"Dataset file {path} must contain a JSON/YAML list of test cases, "
            f"got {type(raw).__name__}"
        )

    test_cases: list[TrajectoryTestCase] = []
    for i, row in enumerate(raw):
        if "question" not in row:
            raise ValueError(f"Row {i} in {path} is missing required field 'question'")
        row.setdefault("dataset_name", inferred_name)
        test_cases.append(TrajectoryTestCase(**row))

    return test_cases
