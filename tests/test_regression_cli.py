"""
Regression/golden tests for CLI commands via CliRunner.
Tests all 5 CLI commands: run, run-trajectory, synthesize, compare, gate.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from rag_score.cli import cli


class TestCLIRunCommand:
    """Tests for `rageval run` command."""

    def test_run_missing_config_fails(self):
        """Missing config file shows clear error."""
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "nonexistent.yaml"])
        assert result.exit_code != 0
        assert "does not exist" in result.output.lower()

    def test_run_missing_required_fields_fails(self):
        """Config missing required fields shows clear error."""
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("dataset: test.json\n")  # missing retriever, generator, metrics
            config_path = f.name

        try:
            result = runner.invoke(cli, ["run", config_path])
            assert result.exit_code != 0
            assert "missing required field" in result.output.lower()
        finally:
            Path(config_path).unlink()

    def test_run_invalid_metric_name_fails(self):
        """Unknown metric name shows clear error (after module resolution)."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a dummy dataset file (must be a list)
            dataset_path = Path(tmpdir) / "test.json"
            dataset_path.write_text('[]')

            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                f.write(f"""
dataset: {dataset_path}
retriever: rag_score.metrics.retrieval.precision_at_k:PrecisionAtK
generator: rag_score.metrics.retrieval.precision_at_k:PrecisionAtK
metrics: [unknown_metric]
""")
                config_path = f.name

            try:
                result = runner.invoke(cli, ["run", config_path])
                assert result.exit_code != 0
                # Either metric validation error or module error - both are valid failures
                output = result.output.lower()
                assert "unknown metric" in output or "no module" in output or "import" in output
            finally:
                Path(config_path).unlink()

    def test_run_judge_metric_without_judge_fails(self):
        """Judge-requiring metric without judge config shows clear error."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a dummy dataset file (must be a list)
            dataset_path = Path(tmpdir) / "test.json"
            dataset_path.write_text('[]')

            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                f.write(f"""
dataset: {dataset_path}
retriever: rag_score.metrics.retrieval.precision_at_k:PrecisionAtK
generator: rag_score.metrics.retrieval.precision_at_k:PrecisionAtK
metrics: [faithfulness]
""")
                config_path = f.name

            try:
                result = runner.invoke(cli, ["run", config_path])
                assert result.exit_code != 0
                output = result.output.lower()
                assert "requires an llm judge" in output or "judge" in output
            finally:
                Path(config_path).unlink()

    def test_run_json_config_works(self):
        """JSON config files work (no yaml dependency)."""
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "dataset": "test.json",
                "retriever": "dummy:retrieve",
                "generator": "dummy:generate",
                "metrics": ["precision_at_5"]
            }, f)
            config_path = f.name

        try:
            result = runner.invoke(cli, ["run", config_path])
            # Config file found, but dataset/test modules missing - that's expected
            # The important thing is it doesn't fail with "yaml not installed"
            assert "yaml" not in result.output.lower() or result.exit_code == 0
        finally:
            Path(config_path).unlink()


class TestCLIRunTrajectoryCommand:
    """Tests for `rageval run-trajectory` command."""

    def test_run_trajectory_missing_config_fails(self):
        """Missing config file shows clear error."""
        runner = CliRunner()
        result = runner.invoke(cli, ["run-trajectory", "nonexistent.yaml"])
        assert result.exit_code != 0
        assert "does not exist" in result.output.lower()

    def test_run_trajectory_missing_required_fields_fails(self):
        """Config missing required fields shows clear error."""
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("dataset: test.json\n")  # missing agent, metrics
            config_path = f.name

        try:
            result = runner.invoke(cli, ["run-trajectory", config_path])
            assert result.exit_code != 0
            assert "missing required field" in result.output.lower()
        finally:
            Path(config_path).unlink()


class TestCLISynthesizeCommand:
    """Tests for `rageval synthesize` command."""

    def test_synthesize_missing_args_fails(self):
        """Missing required args shows clear error."""
        runner = CliRunner()
        result = runner.invoke(cli, ["synthesize"])
        assert result.exit_code != 0
        assert "missing" in result.output.lower() or "required" in result.output.lower()

    def test_synthesize_missing_judge_fails(self):
        """Missing --judge option shows clear error."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(cli, [
                "synthesize",
                tmpdir,
                "--output", "out.json"
            ])
            assert result.exit_code != 0
            assert "missing" in result.output.lower() or "required" in result.output.lower()

    def test_synthesize_invalid_provider_fails(self):
        """Invalid provider shows clear error."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a dummy doc file
            doc_path = Path(tmpdir) / "test.txt"
            doc_path.write_text("Test content")

            result = runner.invoke(cli, [
                "synthesize",
                tmpdir,
                "--output", "out.json",
                "--judge", '{"provider": "invalid_provider", "model": "test"}',
                "--questions-per-chunk", "1"
            ])
            assert result.exit_code != 0
            assert "unknown provider" in result.output.lower() or "invalid" in result.output.lower()

    def test_synthesize_missing_openai_key_fails(self):
        """OpenAI provider without API key shows clear error."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a dummy doc file
            doc_path = Path(tmpdir) / "test.txt"
            doc_path.write_text("Test content")

            result = runner.invoke(cli, [
                "synthesize",
                tmpdir,
                "--output", "out.json",
                "--judge", '{"provider": "openai", "model": "gpt-4o-mini"}',
                "--num-questions", "1"
            ], env={"OPENAI_API_KEY": ""})  # Empty key
            assert result.exit_code != 0


class TestCLICompareCommand:
    """Tests for `rageval compare` command."""

    def test_compare_missing_args_fails(self):
        """Missing required args shows clear error."""
        runner = CliRunner()
        result = runner.invoke(cli, ["compare"])
        assert result.exit_code != 0
        assert "missing" in result.output.lower() or "required" in result.output.lower()

    def test_compare_invalid_files_fails(self):
        """Non-existent files show clear error."""
        runner = CliRunner()
        result = runner.invoke(cli, ["compare", "file1.json", "file2.json"])
        assert result.exit_code != 0
        assert "does not exist" in result.output.lower() or "not found" in result.output.lower()


class TestCLIGateCommand:
    """Tests for `rageval gate` command."""

    def test_gate_missing_args_fails(self):
        """Missing required args shows clear error."""
        runner = CliRunner()
        result = runner.invoke(cli, ["gate"])
        assert result.exit_code != 0
        assert "missing" in result.output.lower() or "required" in result.output.lower()

    def test_gate_invalid_files_fails(self):
        """Non-existent files show clear error."""
        runner = CliRunner()
        result = runner.invoke(cli, ["gate", "file1.json", "file2.json"])
        assert result.exit_code != 0
        assert "does not exist" in result.output.lower() or "not found" in result.output.lower()


class TestCLIVersionCommand:
    """Tests for version output."""

    def test_version_flag(self):
        """--version flag works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "version" in result.output.lower()


class TestCLIHelpCommands:
    """Tests for help output."""

    def test_main_help(self):
        """Main help works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "run" in result.output
        assert "run-trajectory" in result.output
        assert "synthesize" in result.output
        assert "compare" in result.output
        assert "gate" in result.output

    def test_run_help(self):
        """run subcommand help works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "config" in result.output.lower()

    def test_run_trajectory_help(self):
        """run-trajectory subcommand help works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["run-trajectory", "--help"])
        assert result.exit_code == 0
        assert "config" in result.output.lower()

    def test_synthesize_help(self):
        """synthesize subcommand help works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["synthesize", "--help"])
        assert result.exit_code == 0
        assert "docs-dir" in result.output.lower() or "docs_dir" in result.output.lower()
        assert "judge" in result.output.lower()

    def test_compare_help(self):
        """compare subcommand help works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["compare", "--help"])
        assert result.exit_code == 0
        assert "results" in result.output.lower()

    def test_gate_help(self):
        """gate subcommand help works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["gate", "--help"])
        assert result.exit_code == 0
        assert "baseline" in result.output.lower()
        assert "gate" in result.output.lower()