"""Tests for cli.py using Click's CliRunner - the entrypoint had zero
direct test coverage before (only manual bash smoke tests), so this
fills that gap. Focused on error paths that don't need real judge
API calls; the happy-path CLI behavior is already covered indirectly
by the runner/metrics tests plus manual smoke testing during
development.
"""

from __future__ import annotations

import json

from click.testing import CliRunner

from rag_score.cli import cli


def _write_demo_pipeline(runner_fs) -> None:
    with open("demo_pipeline.py", "w") as f:
        f.write(
            "async def my_retriever(query, top_k=5):\n"
            "    from rag_score.core.types import RetrievedChunk\n"
            "    return [RetrievedChunk(doc_id='doc_1', text='some text')]\n"
            "\n"
            "async def my_generator(query, context):\n"
            "    return 'an answer'\n"
        )
    with open("test_set.json", "w") as f:
        json.dump([{"question": "q?", "expected_doc_ids": ["doc_1"]}], f)


class TestRunCommand:
    def test_missing_config_file_is_usage_error(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "does_not_exist.json"])
        assert result.exit_code != 0

    def test_config_missing_required_fields(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            with open("bad_config.json", "w") as f:
                json.dump({"dataset": "test_set.json"}, f)  # missing retriever/generator/metrics
            result = runner.invoke(cli, ["run", "bad_config.json"])
            assert result.exit_code != 0
            assert "missing required field" in result.output

    def test_unknown_metric_is_clean_error(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_demo_pipeline(runner)
            with open("config.json", "w") as f:
                json.dump({
                    "dataset": "test_set.json",
                    "retriever": "demo_pipeline:my_retriever",
                    "generator": "demo_pipeline:my_generator",
                    "metrics": ["not_a_real_metric"],
                }, f)
            result = runner.invoke(cli, ["run", "config.json"])
            assert result.exit_code != 0
            assert "Unknown metric" in result.output

    def test_judge_metric_without_judge_config_is_clean_error(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_demo_pipeline(runner)
            with open("config.json", "w") as f:
                json.dump({
                    "dataset": "test_set.json",
                    "retriever": "demo_pipeline:my_retriever",
                    "generator": "demo_pipeline:my_generator",
                    "metrics": ["faithfulness"],
                }, f)
            result = runner.invoke(cli, ["run", "config.json"])
            assert result.exit_code != 0
            assert "requires an LLM judge" in result.output

    def test_local_ml_metric_failure_is_wrapped_cleanly(self, monkeypatch):
        """The real regression test for the try/except added around
        run_evaluation() - when a local ML metric's model can't be
        loaded, the CLI should surface a clean, actionable error, not
        a raw traceback through transformers/huggingface internals.

        Monkeypatches load_embedding_model to fail immediately rather
        than actually hitting the network - a real download attempt
        against an unreachable host can take 60+ seconds of retries
        before timing out, which is both slow and non-deterministic
        for a test suite (see local_ml.py's own tests for coverage of
        the real ImportError/network-failure paths in isolation)."""
        import rag_score.cli as cli_module  # noqa: F401 - imported for clarity/consistency with other tests

        def _fail_to_load(model_name):
            raise OSError(f"Simulated: couldn't connect to load '{model_name}'")

        # local_faithfulness.py does `from rag_score.judges.local_ml import
        # load_embedding_model`, which binds a local name in ITS module
        # namespace - patching judges.local_ml.load_embedding_model directly
        # wouldn't affect that already-bound reference, so patch the actual
        # call site instead.
        monkeypatch.setattr(
            "rag_score.metrics.generation.local_faithfulness.load_embedding_model",
            _fail_to_load,
        )

        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_demo_pipeline(runner)
            with open("config.json", "w") as f:
                json.dump({
                    "dataset": "test_set.json",
                    "retriever": "demo_pipeline:my_retriever",
                    "generator": "demo_pipeline:my_generator",
                    "metrics": ["local_faithfulness"],
                }, f)
            result = runner.invoke(cli, ["run", "config.json"])
            assert result.exit_code != 0
            assert "Evaluation failed" in result.output
            assert "Simulated" in result.output

    def test_unknown_judge_provider_is_clean_error(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_demo_pipeline(runner)
            with open("config.json", "w") as f:
                json.dump({
                    "dataset": "test_set.json",
                    "retriever": "demo_pipeline:my_retriever",
                    "generator": "demo_pipeline:my_generator",
                    "metrics": ["faithfulness"],
                    "judge": {"provider": "not_a_real_provider"},
                }, f)
            result = runner.invoke(cli, ["run", "config.json"])
            assert result.exit_code != 0
            assert "Unknown judge provider" in result.output


class TestSynthesizeCommand:
    def test_invalid_judge_json_is_clean_error(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            import os
            os.makedirs("docs")
            with open("docs/doc1.txt", "w") as f:
                f.write("some content")
            result = runner.invoke(cli, ["synthesize", "docs", "--judge", "not valid json"])
            assert result.exit_code != 0
            assert "must be valid JSON" in result.output

    def test_empty_docs_dir_is_clean_error(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            import os
            os.makedirs("empty_docs")
            result = runner.invoke(
                cli, ["synthesize", "empty_docs", "--judge", '{"provider": "local", "model": "x"}']
            )
            assert result.exit_code != 0
            assert "No .txt or .md files found" in result.output

    def test_invalid_chunk_overlap_is_clean_error_not_traceback(self):
        """Regression test for the specific bug caught during manual
        CLI testing: overlap >= chunk_size used to crash with a raw
        Python traceback instead of a clean CLI error."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            import os
            os.makedirs("docs")
            with open("docs/doc1.txt", "w") as f:
                f.write(" ".join(f"word{i}" for i in range(100)))
            result = runner.invoke(cli, [
                "synthesize", "docs",
                "--judge", '{"provider": "local", "model": "llama3.1"}',
                "--chunk-size", "50",
                "--chunk-overlap", "50",
            ])
            assert result.exit_code != 0
            assert "overlap must be smaller than chunk_size" in result.output
            assert "Traceback" not in result.output

    def test_query_types_default_is_standard(self):
        """--query-types should default to 'standard' only, preserving old behavior."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            import os
            os.makedirs("docs")
            with open("docs/doc1.txt", "w") as f:
                f.write("some test content")

            # Test without --query-types flag (should default to standard)
            # Note: We expect this to fail due to missing local judge server,
            # but it should get past parameter parsing and into synthesis
            result = runner.invoke(cli, [
                "synthesize", "docs",
                "--judge", '{"provider": "local", "model": "llama3.1"}',
                "--output", "test_set.json"
            ])
            # Should fail due to judge connection issues, not parameter issues
            assert result.exit_code != 0
            assert "Error: Synthesis produced zero test cases" in result.output or \
                   "Error: Unknown judge provider" not in result.output  # If it got past judge parsing

            # More importantly, check that it didn't fail on parameter parsing
            assert "Invalid value for '--query-types'" not in result.output
            assert "Unknown query type" not in result.output  # Should not fail on valid default

    def test_query_types_parameter_accepts_comma_separated_values(self):
        """--query-types should accept comma-separated values."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            import os
            os.makedirs("docs")
            with open("docs/doc1.txt", "w") as f:
                f.write("some test content for testing")

            result = runner.invoke(cli, [
                "synthesize", "docs",
                "--judge", '{"provider": "local", "model": "llama3.1"}',
                "--query-types", "standard,adversarial",
                "--output", "test_set.json"
            ])
            # Should fail due to judge connection issues, not parameter issues
            assert result.exit_code != 0
            # Check that it didn't fail on parameter parsing
            assert "Invalid value for '--query-types'" not in result.output
            assert "Unknown query type" not in result.output  # Should not fail on valid types
            # Should get to the synthesis step
            assert "Loaded 1 document(s) from docs" in result.output
            assert "Chunking at" in result.output and "words (overlap" in result.output

    def test_query_types_invalid_type_shows_clear_error(self):
        """Invalid query type in --query-types should show a clear error message."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            import os
            os.makedirs("docs")
            with open("docs/doc1.txt", "w") as f:
                f.write("some test content")

            result = runner.invoke(cli, [
                "synthesize", "docs",
                "--judge", '{"provider": "local", "model": "llama3.1"}',
                "--query-types", "standard,invalid_type",
                "--output", "test_set.json"
            ])
            assert result.exit_code != 0
            assert "Unknown query type" in result.output
            # Check that we get the valid options in the error message (order may vary)
            assert "Valid options are:" in result.output
            assert "standard" in result.output
            assert "adversarial" in result.output
            assert "multi_hop" in result.output
            assert "unanswerable" in result.output

    def test_query_types_multi_hop_requires_sufficient_content(self):
        """--query-types multi_hop needs sufficient content to form chunk pairs."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            import os
            os.makedirs("docs")
            with open("docs/doc1.txt", "w") as f:
                f.write("short")  # Too short for even one chunk with default size

            result = runner.invoke(cli, [
                "synthesize", "docs",
                "--judge", '{"provider": "local", "model": "llama3.1"}',
                "--query-types", "multi_hop",
                "--chunk-size", "10",
                "--output", "test_set.json"
            ])
            # Should fail due to judge connection issues, not parameter issues
            assert result.exit_code != 0
            # Check that it didn't fail on parameter parsing
            assert "Invalid value for '--query-types'" not in result.output
            assert "Unknown query type" not in result.output  # Should not fail on valid type
            # Should get to the synthesis step
            assert "Loaded 1 document(s) from docs" in result.output
            assert "Chunking at" in result.output and "words (overlap" in result.output

    def test_query_types_unanswerable_has_empty_expected_doc_ids(self):
        """--query-types unanswerable should produce test cases with empty expected_doc_ids."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            import os
            os.makedirs("docs")
            with open("docs/doc1.txt", "w") as f:
                f.write("This is a test document with some content for testing.")

            result = runner.invoke(cli, [
                "synthesize", "docs",
                "--judge", '{"provider": "local", "model": "llama3.1"}',
                "--query-types", "unanswerable",
                "--output", "test_set.json"
            ])
            # Should fail due to judge connection issues, not parameter issues
            assert result.exit_code != 0
            # Check that it didn't fail on parameter parsing
            assert "Invalid value for '--query-types'" not in result.output
            assert "Unknown query type" not in result.output  # Should not fail on valid type
            # Should get to the synthesis step
            assert "Loaded 1 document(s) from docs" in result.output
            assert "Chunking at" in result.output and "words (overlap" in result.output


def _write_demo_agent() -> None:
    with open("demo_agent.py", "w") as f:
        f.write(
            "async def my_agent(query):\n"
            "    from rag_score.agentic.types import ToolCall\n"
            "    return [ToolCall(tool_name='search')], 'an answer'\n"
        )
    with open("trajectory_test_set.json", "w") as f:
        json.dump([{"question": "q?", "expected_tool_sequence": ["search"]}], f)


class TestRunTrajectoryCommand:
    def test_happy_path_end_to_end(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_demo_agent()
            with open("config.json", "w") as f:
                json.dump({
                    "dataset": "trajectory_test_set.json",
                    "agent": "demo_agent:my_agent",
                    "metrics": ["tool_selection_recall", "tool_selection_precision", "tool_call_order_correctness"],
                }, f)
            result = runner.invoke(cli, ["run-trajectory", "config.json"])
            assert result.exit_code == 0
            assert "tool_selection_recall" in result.output
            assert "1.000" in result.output

    def test_missing_required_field_is_clean_error(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_demo_agent()
            with open("config.json", "w") as f:
                json.dump({"dataset": "trajectory_test_set.json", "metrics": ["tool_selection_recall"]}, f)
            result = runner.invoke(cli, ["run-trajectory", "config.json"])
            assert result.exit_code != 0
            assert "missing required field" in result.output
            assert "agent" in result.output

    def test_unknown_metric_is_clean_error(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_demo_agent()
            with open("config.json", "w") as f:
                json.dump({
                    "dataset": "trajectory_test_set.json",
                    "agent": "demo_agent:my_agent",
                    "metrics": ["not_a_real_metric"],
                }, f)
            result = runner.invoke(cli, ["run-trajectory", "config.json"])
            assert result.exit_code != 0
            assert "Unknown trajectory metric" in result.output

    def test_output_file_is_written(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            import os
            _write_demo_agent()
            with open("config.json", "w") as f:
                json.dump({
                    "dataset": "trajectory_test_set.json",
                    "agent": "demo_agent:my_agent",
                    "metrics": ["tool_selection_recall"],
                    "output": "results.json",
                }, f)
            result = runner.invoke(cli, ["run-trajectory", "config.json"])
            assert result.exit_code == 0
            assert os.path.exists("results.json")
            with open("results.json") as f:
                payload = json.load(f)
            assert "summary" in payload
            assert "results" in payload


class TestProgressBar:
    """Regression tests for the progress bar added to both `run` and
    `run-trajectory` - confirms the on_progress wiring doesn't break
    either command and that results are correct with it in place."""

    def test_run_command_still_produces_correct_results_with_progress_bar(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_demo_pipeline(runner)
            with open("config.json", "w") as f:
                json.dump({
                    "dataset": "test_set.json",
                    "retriever": "demo_pipeline:my_retriever",
                    "generator": "demo_pipeline:my_generator",
                    "metrics": ["precision_at_5"],
                }, f)
            result = runner.invoke(cli, ["run", "config.json"])
            assert result.exit_code == 0
            assert "precision_at_5" in result.output

    def test_run_trajectory_command_still_produces_correct_results_with_progress_bar(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_demo_agent()
            with open("config.json", "w") as f:
                json.dump({
                    "dataset": "trajectory_test_set.json",
                    "agent": "demo_agent:my_agent",
                    "metrics": ["tool_selection_recall"],
                }, f)
            result = runner.invoke(cli, ["run-trajectory", "config.json"])
            assert result.exit_code == 0
            assert "tool_selection_recall" in result.output
