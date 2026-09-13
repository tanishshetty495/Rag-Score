"""
Dataset loading — turns a user's test_set.json (or .yaml) into a list of
TestCase objects. Deliberately dumb and dependency-light: no schema
registry, no proprietary format. If it's JSON with the right keys, it works.
"""

from __future__ import annotations

import json
from pathlib import Path

from rag_score.core.types import TestCase


def load_dataset(path: str | Path, dataset_name: str | None = None) -> list[TestCase]:
    """Load a list of TestCase from a .json or .yaml/.yml file.

    Expected JSON shape:
        [
          {
            "question": "...",
            "ground_truth_answer": "...",   # optional
            "expected_doc_ids": ["doc_1"],  # optional
            "difficulty_category": "easy"   # optional
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
            import yaml  # optional dependency, only needed for YAML datasets
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
        # ValueError, not TypeError, is intentional here - it's part of
        # this function's documented and tested contract (see
        # tests/test_dataset.py::test_non_list_json_raises), consistent
        # with the other validation errors load_dataset raises.
        raise ValueError(  # noqa: TRY004
            f"Dataset file {path} must contain a JSON/YAML list of test cases, "
            f"got {type(raw).__name__}"
        )

    test_cases: list[TestCase] = []
    for i, row in enumerate(raw):
        if "question" not in row:
            raise ValueError(f"Row {i} in {path} is missing required field 'question'")
        row.setdefault("dataset_name", inferred_name)
        test_cases.append(TestCase(**row))

    return test_cases
