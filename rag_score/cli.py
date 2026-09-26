"""
CLI entrypoint: `rageval run config.yaml`

Deliberately keeps config loading dependency-light: JSON configs work
with zero extra installs; YAML configs need the `yaml` extra
(pip install rag-score[yaml]) since PyYAML isn't in core deps.

Config format:
    dataset: test_set.json
    retriever: my_module:my_retriever      # "module.path:attribute"
    generator: my_module:my_generator
    metrics: [precision_at_5, recall_at_5, mrr, ndcg_at_5]
    top_k: 5
    max_concurrency: 8
    output: results.json                   # optional, raw per-row dump
"""

from __future__ import annotations

import asyncio
import importlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import click

from rag_score.adapters.base import (
    CallableGeneratorAdapter,
    CallableRetrieverAdapter,
    GeneratorAdapter,
    RetrieverAdapter,
)
from rag_score.agentic.adapters import AgentAdapter, CallableAgentAdapter
from rag_score.agentic.metrics.tool_call_order import ToolCallOrderCorrectness
from rag_score.agentic.metrics.tool_selection_precision import ToolSelectionPrecision
from rag_score.agentic.metrics.tool_selection_recall import ToolSelectionRecall
from rag_score.agentic.metrics_base import TrajectoryMetric
from rag_score.agentic.runner import TrajectoryRunConfig, run_trajectory_evaluation
from rag_score.agentic.types import load_trajectory_dataset
from rag_score.core.dataset import load_dataset
from rag_score.core.runner import RunConfig, run_evaluation
from rag_score.core.types import DimRun
from rag_score.export.sqlite_export import export_to_sqlite
from rag_score.judges.base import LLMJudge
from rag_score.metrics.base import Metric
from rag_score.metrics.generation.answer_relevance import AnswerRelevance
from rag_score.metrics.generation.context_precision import ContextPrecision
from rag_score.metrics.generation.faithfulness import Faithfulness
from rag_score.metrics.generation.local_answer_relevance import (
    LocalSemanticAnswerRelevance,
)
from rag_score.metrics.generation.local_faithfulness import LocalSemanticFaithfulness
from rag_score.metrics.retrieval.mrr import MRR
from rag_score.metrics.retrieval.ndcg import NDCG
from rag_score.metrics.retrieval.precision_at_k import PrecisionAtK
from rag_score.metrics.retrieval.recall_at_k import RecallAtK
from rag_score.report.html_report import generate_html_report
from rag_score.synthesize import load_documents_from_dir, synthesize_test_set
from rag_score.telemetry import TelemetryConfig

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise click.ClickException(f"Config file not found: {path}")

    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:
            raise click.ClickException(
                "PyYAML is required for .yaml configs. "
                "Install it with: pip install rag-score[yaml]  "
                "(or use a .json config instead)"
            ) from e
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Dynamic "module:attribute" resolution for retriever/generator refs
# ---------------------------------------------------------------------------

def _import_from_string(ref: str):
    """Resolve 'my_module.submodule:my_function' to the actual object,
    importing the user's own code from their working directory."""
    if ":" not in ref:
        raise click.ClickException(
            f"Invalid reference '{ref}' - expected format 'module.path:attribute'"
        )
    module_path, _, attr_name = ref.partition(":")

    # So `rageval run config.yaml` finds user modules in the cwd, not
    # just installed packages.
    if str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise click.ClickException(f"Could not import module '{module_path}': {e}") from e

    if not hasattr(module, attr_name):
        raise click.ClickException(f"Module '{module_path}' has no attribute '{attr_name}'")
    return getattr(module, attr_name)


def _resolve_retriever(ref: str) -> RetrieverAdapter:
    obj = _import_from_string(ref)
    if isinstance(obj, RetrieverAdapter):
        return obj
    return CallableRetrieverAdapter(obj)  # plain function -> wrap it


def _resolve_generator(ref: str) -> GeneratorAdapter:
    obj = _import_from_string(ref)
    if isinstance(obj, GeneratorAdapter):
        return obj
    return CallableGeneratorAdapter(obj)


def _resolve_agent(ref: str) -> AgentAdapter:
    obj = _import_from_string(ref)
    if isinstance(obj, AgentAdapter):
        return obj
    return CallableAgentAdapter(obj)


# ---------------------------------------------------------------------------
# Judge construction from config: {"provider": "openai"|"anthropic"|"local", "model": ...}
# ---------------------------------------------------------------------------

def _build_judge(judge_config: dict[str, Any] | None) -> LLMJudge | None:
    if not judge_config:
        return None

    provider = judge_config.get("provider", "").strip().lower()
    model = judge_config.get("model")
    api_key = judge_config.get("api_key")  # falls back to provider's env var if omitted

    # Shared by every provider - only pass through what's actually set,
    # so each judge class's own defaults (2 retries, 1s base delay)
    # still apply when the config leaves these unspecified.
    retry_kwargs: dict[str, Any] = {}
    if "max_retries" in judge_config:
        retry_kwargs["max_retries"] = judge_config["max_retries"]
    if "retry_base_delay" in judge_config:
        retry_kwargs["retry_base_delay"] = judge_config["retry_base_delay"]

    if provider == "openai":
        from rag_score.judges.openai_judge import OpenAIJudge

        kwargs = {"api_key": api_key, **retry_kwargs}
        if model:
            kwargs["model"] = model
        return OpenAIJudge(**kwargs)

    if provider == "anthropic":
        from rag_score.judges.anthropic_judge import AnthropicJudge

        kwargs = {"api_key": api_key, **retry_kwargs}
        if model:
            kwargs["model"] = model
        return AnthropicJudge(**kwargs)

    if provider == "local":
        from rag_score.judges.local_judge import LocalJudge

        if not model:
            raise click.ClickException(
                "The 'local' judge provider requires a 'model' field, e.g. "
                '"model": "llama3.1" (the model name as your local server knows it).'
            )
        kwargs = {"model": model, **retry_kwargs}
        base_url = judge_config.get("base_url")
        if base_url:
            kwargs["base_url"] = base_url
        return LocalJudge(**kwargs)

    raise click.ClickException(
        f"Unknown judge provider '{provider}'. Available: openai, anthropic, local"
    )


# ---------------------------------------------------------------------------
# Metric name -> instance resolution, e.g. "precision_at_5" -> PrecisionAtK(k=5)
# ---------------------------------------------------------------------------

_K_PREFIXES: dict[str, type[Metric]] = {
    "precision_at_": PrecisionAtK,
    "recall_at_": RecallAtK,
    "ndcg_at_": NDCG,
}

_JUDGE_METRICS: dict[str, type[Metric]] = {
    "faithfulness": Faithfulness,
    "answer_relevance": AnswerRelevance,
    "context_precision": ContextPrecision,
}

_LOCAL_ML_METRICS: dict[str, type[Metric]] = {
    "local_faithfulness": LocalSemanticFaithfulness,
    "local_answer_relevance": LocalSemanticAnswerRelevance,
}


def _build_metric(name: str, judge: LLMJudge | None, encoder_model: str | None = None) -> Metric:
    name = name.strip().lower()
    if name == "mrr":
        return MRR()
    if name in _JUDGE_METRICS:
        if judge is None:
            raise click.ClickException(
                f"Metric '{name}' requires an LLM judge, but no 'judge' section was "
                f"found in the config. Add e.g.:\n"
                f'  judge: {{"provider": "openai", "model": "gpt-4o-mini"}}'
            )
        return _JUDGE_METRICS[name](judge=judge)
    if name in _LOCAL_ML_METRICS:
        kwargs = {"model_name": encoder_model} if encoder_model else {}
        return _LOCAL_ML_METRICS[name](**kwargs)
    for prefix, cls in _K_PREFIXES.items():
        if name.startswith(prefix):
            suffix = name[len(prefix):]
            if not suffix.isdigit():
                raise click.ClickException(f"Invalid metric '{name}' - expected e.g. '{prefix}5'")
            return cls(k=int(suffix))
    raise click.ClickException(
        f"Unknown metric '{name}'. Available: precision_at_<k>, recall_at_<k>, mrr, "
        f"ndcg_at_<k>, faithfulness, answer_relevance, context_precision, "
        f"local_faithfulness, local_answer_relevance"
    )


# ---------------------------------------------------------------------------
# Summary table (terminal + $GITHUB_STEP_SUMMARY-friendly markdown)
# ---------------------------------------------------------------------------

def _summarize(scores: list, num_results: int, num_errors: int) -> dict[str, float]:
    by_metric: dict[str, list[float]] = {}
    for s in scores:
        by_metric.setdefault(s.metric_name, []).append(s.score_value)
    return {name: statistics.mean(values) for name, values in by_metric.items()}


def _print_summary(summary: dict[str, float], num_results: int, num_errors: int) -> None:
    click.echo(f"\nEvaluated {num_results} test cases ({num_errors} failed)\n")
    if not summary:
        click.echo("No scores computed.")
        return
    name_width = max(len(n) for n in summary) + 2
    click.echo(f"{'Metric'.ljust(name_width)}Score")
    click.echo("-" * (name_width + 8))
    for name, value in summary.items():
        click.echo(f"{name.ljust(name_width)}{value:.3f}")


def _write_github_step_summary(summary: dict[str, float], num_results: int, num_errors: int) -> None:
    """Append a markdown table to $GITHUB_STEP_SUMMARY if running in GitHub Actions,
    so results show up directly in the workflow run UI with no extra config."""
    import os

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    lines = [
        "## rag-score results",
        f"Evaluated {num_results} test cases ({num_errors} failed)",
        "",
        "| Metric | Score |",
        "|---|---|",
    ]
    for name, value in summary.items():
        lines.append(f"| {name} | {value:.3f} |")

    with open(summary_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(package_name="rag-score")
def cli() -> None:
    """rag-score: framework-agnostic RAG evaluation, zero lock-in."""


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
def run(config_path: str) -> None:
    """Run an evaluation from a YAML or JSON config file."""
    config = _load_config(config_path)

    required = ["dataset", "retriever", "generator", "metrics"]
    missing = [k for k in required if k not in config]
    if missing:
        raise click.ClickException(f"Config is missing required field(s): {', '.join(missing)}")

    test_cases = load_dataset(config["dataset"])
    retriever = _resolve_retriever(config["retriever"])
    generator = _resolve_generator(config["generator"])
    judge = _build_judge(config.get("judge"))
    encoder_model = config.get("encoder_model")
    metrics = [_build_metric(name, judge, encoder_model) for name in config["metrics"]]

    telemetry_config = None
    if "telemetry" in config:
        telemetry_settings = config["telemetry"]
        kwargs: dict[str, Any] = {}
        if "model" in telemetry_settings:
            kwargs["model_name"] = telemetry_settings["model"]
        if "pricing" in telemetry_settings:
            # JSON can't express tuples, so a config's pricing table
            # comes in as {"model": [prompt_price, completion_price]} -
            # convert to the tuple shape TelemetryConfig expects.
            kwargs["pricing"] = {
                model: tuple(prices) for model, prices in telemetry_settings["pricing"].items()
            }
        telemetry_config = TelemetryConfig(**kwargs)

    run_config = RunConfig(
        run_id=f"run-{Path(config_path).stem}",
        project_name=config.get("project_name", "default"),
        top_k=config.get("top_k", 5),
        max_concurrency=config.get("max_concurrency", 8),
        telemetry=telemetry_config,
    )

    click.echo(f"Running {len(test_cases)} test cases with {len(metrics)} metrics...")
    try:
        with click.progressbar(length=len(test_cases), label="Evaluating") as bar:
            report = asyncio.run(
                run_evaluation(
                    test_cases, retriever, generator, metrics, run_config,
                    on_progress=lambda: bar.update(1),
                )
            )
    except Exception as e:  # noqa: BLE001 - intentionally broad, see comment below
        # A metric raising (e.g. a local ML model failing to download,
        # no network for an API judge) isn't isolated per-test-case the
        # way retrieval/generation failures are - it aborts the whole
        # run. Surface it as a clean error here rather than a raw
        # traceback through library internals.
        raise click.ClickException(
            f"Evaluation failed: {type(e).__name__}: {e}\n\n"
            f"If this is a local ML metric (local_faithfulness, "
            f"local_answer_relevance), the model may need to download on "
            f"first use - check your internet connection, or pre-download "
            f"it with: python -c \"from sentence_transformers import "
            f"SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')\""
        ) from None

    num_errors = sum(1 for r in report.results if r.error is not None)
    summary = _summarize(report.scores, len(report.results), num_errors)

    _print_summary(summary, len(report.results), num_errors)
    _write_github_step_summary(summary, len(report.results), num_errors)

    output_path = config.get("output")
    if output_path:
        payload = {
            "run_id": report.run_id,
            "summary": summary,
            "results": [r.model_dump(mode="json") for r in report.results],
            "scores": [s.model_dump(mode="json") for s in report.scores],
        }
        Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        click.echo(f"\nFull results written to {output_path}")

    needs_dim_run = config.get("sqlite_output") or config.get("html_output")
    dim_run = None
    if needs_dim_run:
        dim_run = DimRun(
            run_id=report.run_id,
            project_name=run_config.project_name,
            dataset_name=test_cases[0].dataset_name if test_cases else "default",
            retriever_config=config["retriever"],
            generator_config=config["generator"],
            environment=config.get("environment", "local"),
        )

    sqlite_output = config.get("sqlite_output")
    if sqlite_output:
        export_to_sqlite(sqlite_output, dim_run, test_cases, report)
        click.echo(f"Run appended to SQLite star schema at {sqlite_output}")

    html_output = config.get("html_output")
    if html_output:
        written = generate_html_report(html_output, dim_run, test_cases, report)
        click.echo(f"HTML report written to {written}")


@cli.command()
@click.argument("docs_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--output", "-o", default="test_set.json", help="Where to write the generated test set.")
@click.option("--judge", "judge_json", required=True, help='Judge config as JSON, e.g. \'{"provider": "anthropic", "model": "claude-haiku-4-5"}\'')
@click.option("--chunk-size", default=500, show_default=True, help="Words per chunk.")
@click.option("--chunk-overlap", default=50, show_default=True, help="Overlapping words between consecutive chunks.")
@click.option("--questions-per-chunk", default=1, show_default=True, help="How many questions to generate per chunk.")
@click.option("--max-concurrency", default=5, show_default=True, help="Concurrent judge calls.")
@click.option("--query-types", default="standard", show_default=True, help="Comma-separated list of query types to generate: standard, adversarial, multi_hop, unanswerable")
def synthesize(
    docs_dir: str,
    output: str,
    judge_json: str,
    chunk_size: int,
    chunk_overlap: int,
    questions_per_chunk: int,
    max_concurrency: int,
    query_types: str,
) -> None:
    """Generate a synthetic test_set.json from a directory of .txt/.md documents.

    Example:
        rageval synthesize ./docs --judge '{"provider": "anthropic"}' --output test_set.json
        rageval synthesize ./docs --judge '{"provider": "anthropic"}' --query-types standard,adversarial,unanswerable --output test_set.json
    """
    try:
        judge_config = json.loads(judge_json)
    except json.JSONDecodeError as e:
        raise click.ClickException(f"--judge must be valid JSON: {e}") from None

    judge = _build_judge(judge_config)
    if judge is None:
        raise click.ClickException("--judge config resolved to no judge - check the provider field.")

    documents = load_documents_from_dir(docs_dir)
    if not documents:
        raise click.ClickException(f"No .txt or .md files found in {docs_dir}")

    click.echo(f"Loaded {len(documents)} document(s) from {docs_dir}")
    click.echo(f"Chunking at {chunk_size} words (overlap {chunk_overlap}) and generating questions...")

    try:
        # Parse comma-separated query types
        query_types_list = [qt.strip() for qt in query_types.split(",") if qt.strip()]
        if not query_types_list:
            query_types_list = ["standard"]  # fallback to default

        report = asyncio.run(
            synthesize_test_set(
                documents,
                judge,
                dataset_name="synthesized",  # explicit for clarity
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                questions_per_chunk=questions_per_chunk,
                max_concurrency=max_concurrency,
                query_types=query_types_list,
            )
        )
    except ValueError as e:
        # Chunking parameter validation (e.g. overlap >= chunk_size) -
        # a user config mistake, not a bug, so no traceback needed.
        raise click.ClickException(str(e)) from None

    if not report.test_cases:
        raise click.ClickException(
            f"Synthesis produced zero test cases ({len(report.errors)} chunk(s) failed). "
            f"First error: {report.errors[0] if report.errors else 'unknown'}"
        )

    payload = [tc.model_dump(mode="json") for tc in report.test_cases]
    Path(output).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    click.echo(f"\nGenerated {len(report.test_cases)} test case(s), {len(report.errors)} chunk(s) failed")
    click.echo(f"Written to {output}")
    if report.errors:
        click.echo("\nFailed chunks:")
        for err in report.errors:
            click.echo(f"  {err}")






# ---------------------------------------------------------------------------
# Trajectory metric name -> instance resolution
# ---------------------------------------------------------------------------

_TRAJECTORY_METRICS: dict[str, type[TrajectoryMetric]] = {
    "tool_selection_recall": ToolSelectionRecall,
    "tool_selection_precision": ToolSelectionPrecision,
    "tool_call_order_correctness": ToolCallOrderCorrectness,
}


def _build_trajectory_metric(name: str) -> TrajectoryMetric:
    name = name.strip().lower()
    if name in _TRAJECTORY_METRICS:
        return _TRAJECTORY_METRICS[name]()
    raise click.ClickException(
        f"Unknown trajectory metric '{name}'. Available: "
        f"{', '.join(_TRAJECTORY_METRICS.keys())}"
    )


@cli.command(name="run-trajectory")
@click.argument("config_path", type=click.Path(exists=True))
def run_trajectory(config_path: str) -> None:
    """Run an agentic trajectory evaluation from a YAML or JSON config file.

    Config format:
        dataset: trajectory_test_set.json
        agent: my_module:my_agent          # "module.path:attribute"
        metrics: [tool_selection_recall, tool_selection_precision, tool_call_order_correctness]
        max_concurrency: 8
        output: results.json               # optional, raw per-row dump

    Unlike `run`, there's no separate retriever/generator - `agent`
    resolves to a function/AgentAdapter that takes a query and returns
    the full (tool_calls, final_answer) trajectory in one call, since
    the tool-calling loop is internal to the agent itself.
    """
    config = _load_config(config_path)

    required = ["dataset", "agent", "metrics"]
    missing = [k for k in required if k not in config]
    if missing:
        raise click.ClickException(f"Config is missing required field(s): {', '.join(missing)}")

    test_cases = load_trajectory_dataset(config["dataset"])
    agent = _resolve_agent(config["agent"])
    metrics = [_build_trajectory_metric(name) for name in config["metrics"]]

    run_config = TrajectoryRunConfig(
        run_id=f"run-{Path(config_path).stem}",
        project_name=config.get("project_name", "default"),
        max_concurrency=config.get("max_concurrency", 8),
    )

    click.echo(f"Running {len(test_cases)} test cases with {len(metrics)} metrics...")
    try:
        with click.progressbar(length=len(test_cases), label="Evaluating") as bar:
            report = asyncio.run(
                run_trajectory_evaluation(
                    test_cases, agent, metrics, run_config,
                    on_progress=lambda: bar.update(1),
                )
            )
    except Exception as e:  # noqa: BLE001 - intentionally broad, mirrors `run`'s error wrapping
        raise click.ClickException(f"Evaluation failed: {type(e).__name__}: {e}") from None

    num_errors = sum(1 for r in report.results if r.error is not None)
    summary = _summarize(report.scores, len(report.results), num_errors)
    _print_summary(summary, len(report.results), num_errors)
    _write_github_step_summary(summary, len(report.results), num_errors)

    output_path = config.get("output")
    if output_path:
        payload = {
            "run_id": report.run_id,
            "summary": summary,
            "results": [r.model_dump(mode="json") for r in report.results],
            "scores": [s.model_dump(mode="json") for s in report.scores],
        }
        Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        click.echo(f"\nFull results written to {output_path}")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
