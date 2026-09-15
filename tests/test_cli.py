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
