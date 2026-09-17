"""Tests for agentic/types.py."""

from __future__ import annotations

import json

import pytest

from rag_score.agentic.types import (
    ToolCall,
    TrajectoryTestCase,
    load_trajectory_dataset,
)


class TestToolCall:
    def test_minimal_construction(self):
        call = ToolCall(tool_name="search")
        assert call.tool_name == "search"
        assert call.tool_input == {}
        assert call.tool_output is None

    def test_full_construction(self):
        call = ToolCall(
            tool_name="calculator",
            tool_input={"expr": "2+2"},
            tool_output="4",
            latency_ms=12.5,
        )
        assert call.tool_input == {"expr": "2+2"}
        assert call.tool_output == "4"
        assert call.latency_ms == 12.5


class TestTrajectoryTestCase:
    def test_defaults(self):
        tc = TrajectoryTestCase(question="q")
        assert tc.expected_tool_sequence == []
        assert tc.expected_final_answer is None
        assert tc.dataset_name == "default"
        assert tc.test_case_id  # auto-generated

    def test_with_expectations(self):
        tc = TrajectoryTestCase(
            question="q", expected_tool_sequence=["search", "calc"], expected_final_answer="42"
        )
        assert tc.expected_tool_sequence == ["search", "calc"]
        assert tc.expected_final_answer == "42"


class TestLoadTrajectoryDataset:
    def test_loads_valid_json(self, tmp_path):
        data = [
            {"question": "Q1?", "expected_tool_sequence": ["search"]},
            {"question": "Q2?", "expected_tool_sequence": ["calc"]},
        ]
        path = tmp_path / "trajectory_set.json"
        path.write_text(json.dumps(data))

        cases = load_trajectory_dataset(path)
        assert len(cases) == 2
        assert cases[0].question == "Q1?"
        assert cases[0].expected_tool_sequence == ["search"]

    def test_infers_dataset_name_from_filename(self, tmp_path):
        path = tmp_path / "my_trajectories.json"
        path.write_text(json.dumps([{"question": "Q?"}]))
        cases = load_trajectory_dataset(path)
        assert cases[0].dataset_name == "my_trajectories"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_trajectory_dataset(tmp_path / "does_not_exist.json")

    def test_missing_question_field_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps([{"expected_tool_sequence": ["x"]}]))
        with pytest.raises(ValueError, match="question"):
            load_trajectory_dataset(path)

    def test_non_list_json_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"question": "not a list"}))
        with pytest.raises(ValueError, match="list"):
            load_trajectory_dataset(path)

    def test_optional_fields_default_correctly(self, tmp_path):
        path = tmp_path / "minimal.json"
        path.write_text(json.dumps([{"question": "Q?"}]))
        cases = load_trajectory_dataset(path)
        assert cases[0].expected_tool_sequence == []
        assert cases[0].expected_final_answer is None
