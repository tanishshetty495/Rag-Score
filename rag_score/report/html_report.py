"""
Standalone HTML report generation.

jinja2 is an optional extra (pip install rag-score[report]) so the
core install stays dependency-light; this module only imports it at
call time. Output is a single self-contained .html file with inline
CSS - no server, no external assets, easy to attach to a PR or open
straight from a CI artifact.
"""

from __future__ import annotations

from pathlib import Path

from rag_score.core.runner import RunReport
from rag_score.core.types import DimRun, TestCase


_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _score_class(value: float) -> str:
    """Simple traffic-light coloring: >=0.8 good, >=0.5 mid, else bad.
    Kept as a template-callable rather than baked into the data so the
    threshold logic lives in one place."""
    if value >= 0.8:
        return "score-good"
    if value >= 0.5:
        return "score-mid"
    return "score-bad"


def _build_rows(
    report: RunReport, test_cases: list[TestCase]
) -> list[dict]:
    tc_by_id = {tc.test_case_id: tc for tc in test_cases}
    scores_by_eval: dict[str, dict[str, float]] = {}
    reasoning_by_eval: dict[str, dict[str, str]] = {}
    for s in report.scores:
        scores_by_eval.setdefault(s.evaluation_id, {})[s.metric_name] = s.score_value
        if s.judge_reasoning:
            reasoning_by_eval.setdefault(s.evaluation_id, {})[s.metric_name] = s.judge_reasoning

    rows = []
    for result in report.results:
        tc = tc_by_id.get(result.test_case_id)
        total_latency = None
        if result.retrieval_latency_ms is not None and result.generation_latency_ms is not None:
            total_latency = f"{result.retrieval_latency_ms + result.generation_latency_ms:.0f}ms"

        # Build telemetry string for display if telemetry data is present
        telemetry_parts = []
        if result.total_tokens is not None:
            telemetry_parts.append(f"{result.total_tokens:,} tokens")
        if result.estimated_cost_usd is not None:
            telemetry_parts.append(f"${result.estimated_cost_usd:.4f}")
        telemetry = " • ".join(telemetry_parts) if telemetry_parts else None

        rows.append(
            {
                "question": tc.question if tc else "(unknown question)",
                "generated_answer": result.generated_answer,
                "reasoning": reasoning_by_eval.get(result.evaluation_id, {}),
                "error": result.error,
                "scores": scores_by_eval.get(result.evaluation_id, {}),
                "total_latency_ms": total_latency or "—",
                "telemetry": telemetry,
            }
        )
    return rows


def _summarize(report: RunReport) -> dict[str, float]:
    import statistics

    by_metric: dict[str, list[float]] = {}
    for s in report.scores:
        by_metric.setdefault(s.metric_name, []).append(s.score_value)
    return {name: statistics.mean(values) for name, values in by_metric.items()}


def _summarize_telemetry(report: RunReport) -> dict[str, float | int]:
    """Summarize telemetry data across the report."""
    total_tokens = 0
    total_cost = 0.0
    has_telemetry = False

    for result in report.results:
        if result.total_tokens is not None:
            total_tokens += result.total_tokens
            has_telemetry = True
        if result.estimated_cost_usd is not None:
            total_cost += result.estimated_cost_usd
            has_telemetry = True

    if not has_telemetry:
        return {}

    return {
        "total_tokens": total_tokens,
        "total_cost": total_cost,
    }


# Chart is generated as inline SVG (not Chart.js/a CDN script) so the
# report stays a genuinely self-contained file - no internet
# connection needed to view it, which matters for CI artifacts and
# air-gapped environments.
_CHART_WIDTH = 640
_BAR_HEIGHT = 28
_BAR_GAP = 14
_LABEL_WIDTH = 160
_CHART_COLORS = {"good": "#3ecf8e", "mid": "#e8b339", "bad": "#f0546b"}


def _bar_color(value: float) -> str:
    if value >= 0.8:
        return _CHART_COLORS["good"]
    if value >= 0.5:
        return _CHART_COLORS["mid"]
    return _CHART_COLORS["bad"]


def _build_score_chart_svg(summary: dict[str, float]) -> str:
    """Render a horizontal bar chart of average score per metric as a
    plain SVG string. Returns an empty string if there's nothing to
    chart, so the template can skip the section cleanly."""
    if not summary:
        return ""

    bar_area_width = _CHART_WIDTH - _LABEL_WIDTH - 60  # leave room for the value label
    row_height = _BAR_HEIGHT + _BAR_GAP
    height = row_height * len(summary)

    bars = []
    for i, (name, value) in enumerate(summary.items()):
        y = i * row_height
        bar_width = max(2, value * bar_area_width)  # min width so 0.0 bars are still visible
        color = _bar_color(value)
        bars.append(
            f'<text x="{_LABEL_WIDTH - 10}" y="{y + _BAR_HEIGHT / 2 + 4}" '
            f'text-anchor="end" font-size="12" fill="#8b90a0">{name}</text>'
            f'<rect x="{_LABEL_WIDTH}" y="{y}" width="{bar_area_width}" height="{_BAR_HEIGHT}" '
            f'rx="4" fill="#1e222c" />'
            f'<rect x="{_LABEL_WIDTH}" y="{y}" width="{bar_width:.1f}" height="{_BAR_HEIGHT}" '
            f'rx="4" fill="{color}" />'
            f'<text x="{_LABEL_WIDTH + bar_area_width + 8}" y="{y + _BAR_HEIGHT / 2 + 4}" '
            f'font-size="12" fill="#e6e8ee">{value:.3f}</text>'
        )

    svg_body = "".join(bars)
    return (
        f'<svg viewBox="0 0 {_CHART_WIDTH} {height}" width="100%" '
        f'style="max-width: {_CHART_WIDTH}px;">{svg_body}</svg>'
    )


def generate_html_report(
    output_path: str | Path,
    run: DimRun,
    test_cases: list[TestCase],
    report: RunReport,
) -> Path:
    """Render report.html from a completed RunReport. Returns the
    written path."""
    try:
        from jinja2 import Environment, FileSystemLoader
    except ImportError as e:
        raise ImportError(
            "jinja2 is required for HTML reports. "
            "Install it with: pip install rag-score[report]"
        ) from e

    env = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)), autoescape=True)
    env.globals["score_class"] = _score_class
    template = env.get_template("report.html.j2")

    num_errors = sum(1 for r in report.results if r.error is not None)
    summary = _summarize(report)
    telemetry_summary = _summarize_telemetry(report)

    html = template.render(
        run=run,
        num_results=len(report.results),
        num_errors=num_errors,
        summary=summary,
        telemetry_summary=telemetry_summary,
        chart_svg=_build_score_chart_svg(summary),
        rows=_build_rows(report, test_cases),
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path