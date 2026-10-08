"""
Test that the public API remains stable.
"""

from __future__ import annotations

import rag_score


def test_public_api_exports():
    """Test that all expected public attributes are available."""
    # Core
    assert hasattr(rag_score, "TestCase")
    assert hasattr(rag_score, "EvalResult")
    assert hasattr(rag_score, "load_dataset")
    assert hasattr(rag_score, "RunConfig")
    assert hasattr(rag_score, "run_evaluation")
    assert hasattr(rag_score, "DimRun")
    assert hasattr(rag_score, "DimTestCase")

    # Metrics
    assert hasattr(rag_score, "PrecisionAtK")
    assert hasattr(rag_score, "RecallAtK")
    assert hasattr(rag_score, "MRR")
    assert hasattr(rag_score, "NDCG")
    assert hasattr(rag_score, "Faithfulness")
    assert hasattr(rag_score, "AnswerRelevance")
    assert hasattr(rag_score, "ContextPrecision")
    assert hasattr(rag_score, "ContextCarryOver")
    assert hasattr(rag_score, "LocalSemanticAnswerRelevance")
    assert hasattr(rag_score, "LocalSemanticFaithfulness")

    # Agentic
    assert hasattr(rag_score, "TrajectoryTestCase")
    assert hasattr(rag_score, "TrajectoryEvalResult")
    assert hasattr(rag_score, "ToolSelectionRecall")
    assert hasattr(rag_score, "ToolSelectionPrecision")
    assert hasattr(rag_score, "ToolCallOrderCorrectness")

    # Judges
    assert hasattr(rag_score, "LLMJudge")
    assert hasattr(rag_score, "JudgeVerdict")
    assert hasattr(rag_score, "OpenAIJudge")
    assert hasattr(rag_score, "AnthropicJudge")
    assert hasattr(rag_score, "LocalJudge")

    # Export
    assert hasattr(rag_score, "export_to_sqlite")
    assert hasattr(rag_score, "results_to_dataframe")
    assert hasattr(rag_score, "scores_to_dataframe")
    assert hasattr(rag_score, "scores_to_wide_dataframe")
    assert hasattr(rag_score, "full_report_dataframe")

    # Report
    assert hasattr(rag_score, "generate_html_report")

    # Synthesize
    assert hasattr(rag_score, "synthesize_test_set")

    # Cache
    assert hasattr(rag_score, "JudgeCache")
    assert hasattr(rag_score, "InMemoryCacheBackend")
    assert hasattr(rag_score, "FileCacheBackend")
    assert hasattr(rag_score, "CacheConfig")
    assert hasattr(rag_score, "get_cache_backend")

    # Telemetry
    assert hasattr(rag_score, "TelemetryConfig")
    assert hasattr(rag_score, "count_tokens")
    assert hasattr(rag_score, "estimate_cost")


def test_public_api_all():
    """Test that __all__ contains exactly what we expect."""
    expected_public_names = {
        # Core
        "TestCase",
        "EvalResult",
        "load_dataset",
        "RunConfig",
        "run_evaluation",
        "DimRun",
        "DimTestCase",
        # Metrics
        "CitationCompleteness",
        "CitationCorrectness",
        "ContextCoverage",
        "HitRateAtK",
        "NoAnswerAbstention",
        "PrecisionAtK",
        "RecallAtK",
        "MRR",
        "NDCG",
        "Faithfulness",
        "AnswerRelevance",
        "ContextPrecision",
        "ContextCarryOver",
        "LocalSemanticAnswerRelevance",
        "LocalSemanticFaithfulness",
        # Agentic
        "TrajectoryTestCase",
        "TrajectoryEvalResult",
        "ToolSelectionRecall",
        "ToolSelectionPrecision",
        "ToolCallOrderCorrectness",
        # Judges
        "LLMJudge",
        "JudgeVerdict",
        "OpenAIJudge",
        "AnthropicJudge",
        "LocalJudge",
        # Export
        "export_to_sqlite",
        "results_to_dataframe",
        "scores_to_dataframe",
        "scores_to_wide_dataframe",
        "full_report_dataframe",
        # Report
        "generate_html_report",
        # Synthesize
        "synthesize_test_set",
        # Cache
        "JudgeCache",
        "InMemoryCacheBackend",
        "FileCacheBackend",
        "CacheConfig",
        "get_cache_backend",
        # Telemetry
        "TelemetryConfig",
        "count_tokens",
        "estimate_cost",
    }

    actual_public_names = set(rag_score.__all__)
    assert actual_public_names == expected_public_names, (
        f"Public API mismatch.\n"
        f"Missing: {expected_public_names - actual_public_names}\n"
        f"Extra: {actual_public_names - expected_public_names}"
    )
