"""
Statistical significance testing for regression detection.

This module provides functions to determine if a difference between two
samples is statistically significant, accounting for judge variance and
other sources of noise in LLM-based evaluations.
"""

from __future__ import annotations

import random

# Try to import scipy for Welch's t-test, but make it optional
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = None  # type: ignore


def welch_t_test(
    baseline_scores: list[float],
    current_scores: list[float],
) -> tuple[float, float, float, float, str]:
    """
    Perform Welch's t-test (unequal variance t-test) on two samples.

    Args:
        baseline_scores: List of scores from the baseline run
        current_scores: List of scores from the current run

    Returns:
        Tuple of (baseline_mean, current_mean, delta, p_value, method_used)
        where method_used is either "welch_t_test" or "bootstrap"

    Raises:
        ValueError: If samples are too small or invalid
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy is not available. Install with: pip install rag-score[stats]")

    if len(baseline_scores) < 2:
        raise ValueError(f"Baseline sample too small for Welch's t-test: {len(baseline_scores)} < 2")
    if len(current_scores) < 2:
        raise ValueError(f"Current sample too small for Welch's t-test: {len(current_scores)} < 2")

    # Calculate means
    baseline_mean = sum(baseline_scores) / len(baseline_scores)
    current_mean = sum(current_scores) / len(current_scores)
    delta = current_mean - baseline_mean

    # Perform Welch's t-test
    # scipy.stats.ttest_ind with equal_var=False performs Welch's t-test
    _t_stat, p_value = stats.ttest_ind(baseline_scores, current_scores, equal_var=False)

    return baseline_mean, current_mean, delta, p_value, "welch_t_test"


def bootstrap_confidence_interval(
    baseline_scores: list[float],
    current_scores: list[float],
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
) -> tuple[float, float, float, tuple[float, float], str]:
    """
    Calculate confidence interval for the difference using bootstrap resampling.

    Args:
        baseline_scores: List of scores from the baseline run
        current_scores: List of scores from the current run
        n_iterations: Number of bootstrap iterations (default: 1000)
        confidence_level: Confidence level for the interval (default: 0.95)

    Returns:
        Tuple of (baseline_mean, current_mean, delta, (ci_lower, ci_upper), method_used)
        where method_used is "bootstrap"
    """
    if len(baseline_scores) == 0:
        raise ValueError("Baseline sample cannot be empty")
    if len(current_scores) == 0:
        raise ValueError("Current sample cannot be empty")

    # Calculate means
    baseline_mean = sum(baseline_scores) / len(baseline_scores)
    current_mean = sum(current_scores) / len(current_scores)
    delta = current_mean - baseline_mean

    # Bootstrap resampling
    bootstrap_deltas = []
    n_baseline = len(baseline_scores)
    n_current = len(current_scores)

    for _ in range(n_iterations):
        # Resample with replacement
        baseline_sample = [random.choice(baseline_scores) for _ in range(n_baseline)]
        current_sample = [random.choice(current_scores) for _ in range(n_current)]

        # Calculate delta for this iteration
        sample_baseline_mean = sum(baseline_sample) / len(baseline_sample)
        sample_current_mean = sum(current_sample) / len(current_sample)
        sample_delta = sample_current_mean - sample_baseline_mean
        bootstrap_deltas.append(sample_delta)

    # Calculate confidence interval
    alpha = 1.0 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    ci_lower = float(sorted(bootstrap_deltas)[int(len(bootstrap_deltas) * lower_percentile / 100)])
    ci_upper = float(sorted(bootstrap_deltas)[int(len(bootstrap_deltas) * upper_percentile / 100)])

    return baseline_mean, current_mean, delta, (ci_lower, ci_upper), "bootstrap"


def test_significance(
    baseline_scores: list[float],
    current_scores: list[float],
    use_scipy: bool = True,
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
) -> dict:
    """
    Test if the difference between two samples is statistically significant.

    This function tries to use Welch's t-test (via scipy) if available and requested,
    falling back to bootstrap confidence interval if scipy is not available or
    if use_scipy=False.

    Args:
        baseline_scores: List of scores from the baseline run
        current_scores: List of scores from the current run
        use_scipy: Whether to prefer scipy's Welch's t-test (default: True)
        alpha: Significance threshold (default: 0.05)
        n_bootstrap: Number of bootstrap iterations if fallback is used (default: 1000)

    Returns:
        Dictionary with keys:
        - baseline_mean: Mean of baseline scores
        - current_mean: Mean of current scores
        - delta: Difference (current - baseline)
        - p_value: p-value if using Welch's t-test, None if using bootstrap
        - ci_lower: Lower bound of confidence interval (bootstrap only)
        - ci_upper: Upper bound of confidence interval (bootstrap only)
        - significant: Boolean indicating if difference is statistically significant
        - method: String indicating which method was used ("welch_t_test" or "bootstrap")

    Handles degenerate cases:
        - Empty samples: returns None for means, delta=0, significant=False
        - Single sample: cannot compute significance, returns significant=False
        - Zero variance: handled appropriately by the statistical tests
    """
    # Handle degenerate cases
    if len(baseline_scores) == 0 and len(current_scores) == 0:
        return {
            "baseline_mean": None,
            "current_mean": None,
            "delta": 0.0,
            "p_value": None,
            "ci_lower": None,
            "ci_upper": None,
            "significant": False,
            "method": "degenerate_empty_both"
        }

    if len(baseline_scores) == 0:
        return {
            "baseline_mean": None,
            "current_mean": sum(current_scores) / len(current_scores) if current_scores else None,
            "delta": (sum(current_scores) / len(current_scores) if current_scores else 0.0),
            "p_value": None,
            "ci_lower": None,
            "ci_upper": None,
            "significant": False,
            "method": "degenerate_empty_baseline"
        }

    if len(current_scores) == 0:
        return {
            "baseline_mean": sum(baseline_scores) / len(baseline_scores) if baseline_scores else None,
            "current_mean": None,
            "delta": -(sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0.0),
            "p_value": None,
            "ci_lower": None,
            "ci_upper": None,
            "significant": False,
            "method": "degenerate_empty_current"
        }

    # For meaningful statistical testing, we need at least 2 samples in each group
    # for variance estimation. With n=1, we can't estimate variance.
    if len(baseline_scores) < 2 or len(current_scores) < 2:
        baseline_mean = sum(baseline_scores) / len(baseline_scores) if baseline_scores else None
        current_mean = sum(current_scores) / len(current_scores) if current_scores else None
        delta = (current_mean or 0.0) - (baseline_mean or 0.0)

        return {
            "baseline_mean": baseline_mean,
            "current_mean": current_mean,
            "delta": delta,
            "p_value": None,
            "ci_lower": None,
            "ci_upper": None,
            "significant": False,
            "method": "insufficient_sample_size"
        }

    # Try to use Welch's t-test if scipy is available and requested
    if use_scipy and SCIPY_AVAILABLE:
        try:
            baseline_mean, current_mean, delta, p_value, method = welch_t_test(
                baseline_scores, current_scores
            )
            significant = p_value < alpha
            return {
                "baseline_mean": baseline_mean,
                "current_mean": current_mean,
                "delta": delta,
                "p_value": p_value,
                "ci_lower": None,
                "ci_upper": None,
                "significant": significant,
                "method": method
            }
        except (ValueError, ImportError):
            # If Welch's t-test fails due to scipy unavailability or insufficient
            # sample sizes, we fall back to the bootstrap method.
            pass

    # Fall back to bootstrap confidence interval
    baseline_mean, current_mean, delta, ci, method = bootstrap_confidence_interval(
        baseline_scores, current_scores, n_iterations=n_bootstrap
    )
    ci_lower, ci_upper = ci
    # For bootstrap, we consider it significant if zero is NOT in the confidence interval
    significant = not (ci_lower <= 0 <= ci_upper)

    return {
        "baseline_mean": baseline_mean,
        "current_mean": current_mean,
        "delta": delta,
        "p_value": None,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "significant": significant,
        "method": method
    }


def extract_metric_scores(results_data: dict, metric_name: str) -> list[float]:
    """
    Extract all scores for a given metric from results data.

    Args:
        results_data: The loaded JSON results data from a results file
        metric_name: The name of the metric to extract scores for

    Returns:
        List of score values for the specified metric
    """
    scores = []
    if "scores" in results_data and isinstance(results_data["scores"], list):
        for score_obj in results_data["scores"]:
            if isinstance(score_obj, dict) and score_obj.get("metric_name") == metric_name:
                score_value = score_obj.get("score_value")
                if score_value is not None:
                    try:
                        scores.append(float(score_value))
                    except (ValueError, TypeError):
                        # Skip non-numeric scores
                        pass
    return scores