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
from rag_score.core.dataset import load_dataset
from rag_score.core.runner import RunConfig, run_evaluation
from rag_score.core.types import DimRun
from rag_score.export.sqlite_export import export_to_sqlite
from rag_score.judges.base import LLMJudge
from rag_score.metrics.base import Metric
from rag_score.metrics.generation.answer_relevance import AnswerRelevance
from rag_score.metrics.generation.context_precision import ContextPrecision
from rag_score.metrics.generation.faithfulness import Faithfulness
from rag_score.metrics.retrieval.mrr import MRR
from rag_score.metrics.retrieval.ndcg import NDCG
from rag_score.metrics.retrieval.precision_at_k import PrecisionAtK
from rag_score.metrics.retrieval.recall_at_k import RecallAtK
from rag_score.report.html_report import generate_html_report

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


# ---------------------------------------------------------------------------
# Judge construction from config: {"provider": "openai"|"anthropic"|"local", "model": ...}
# ---------------------------------------------------------------------------

def _build_judge(judge_config: dict[str, Any] | None) -> LLMJudge | None:
    if not judge_config:
        return None

    provider = judge_config.get("provider", "").strip().lower()
    model = judge_config.get("model")
    api_key = judge_config.get("api_key")  # falls back to provider's env var if omitted

    if provider == "openai":
        from rag_score.judges.openai_judge import OpenAIJudge

        kwargs = {"api_key": api_key}
        if model:
            kwargs["model"] = model
        return OpenAIJudge(**kwargs)

    if provider == "anthropic":
        from rag_score.judges.anthropic_judge import AnthropicJudge

        kwargs = {"api_key": api_key}
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
        kwargs = {"model": model}
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


def _build_metric(name: str, judge: LLMJudge | None) -> Metric:
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
    for prefix, cls in _K_PREFIXES.items():
        if name.startswith(prefix):
            suffix = name[len(prefix):]
            if not suffix.isdigit():
                raise click.ClickException(f"Invalid metric '{name}' - expected e.g. '{prefix}5'")
            return cls(k=int(suffix))
    raise click.ClickException(
        f"Unknown metric '{name}'. Available: precision_at_<k>, recall_at_<k>, mrr, "
        f"ndcg_at_<k>, faithfulness, answer_relevance, context_precision"
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
    metrics = [_build_metric(name, judge) for name in config["metrics"]]

    run_config = RunConfig(
        run_id=f"run-{Path(config_path).stem}",
        project_name=config.get("project_name", "default"),
        top_k=config.get("top_k", 5),
        max_concurrency=config.get("max_concurrency", 8),
    )

    click.echo(f"Running {len(test_cases)} test cases with {len(metrics)} metrics...")
    report = asyncio.run(run_evaluation(test_cases, retriever, generator, metrics, run_config))

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


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
