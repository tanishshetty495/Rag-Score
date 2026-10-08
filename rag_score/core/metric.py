"""
Metric base class and registry for the plugin-style metric system.
"""

from __future__ import annotations

import inspect
import logging

from .models import EvaluationCase, MetricResult

logger = logging.getLogger(__name__)


class Metric:
    """
    Base class for all metrics in the new system.

    Subclasses set `name` and implement `evaluate(case) -> MetricResult`.
    """

    name: str = "base_metric"

    def evaluate(self, case: EvaluationCase) -> MetricResult:
        """
        Evaluate the metric on the given case and return a MetricResult.

        This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<Metric name={self.name!r}>"


# Global registry for metrics
_registry: dict[str, type[Metric]] = {}


def _is_metric_like(cls: type) -> bool:
    """Check if a class looks like a metric (has name attribute)."""
    return inspect.isclass(cls) and hasattr(cls, "name")


def _get_metric_name(cls: type) -> str:
    """
    Get the name of a metric class.
    For metrics that set name as instance attribute (like our new metrics),
    we need to create a temporary instance to get the name.
    """
    # Check if this is one of our new Metric subclasses with name set in __init__
    # Heuristic: if cls is a subclass of our Metric and cls.name is the default
    # "base_metric", then the real name is likely set as an instance attribute
    if issubclass(cls, Metric) and hasattr(cls, "name") and cls.name == "base_metric":
        # Try to get the name from an instance
        try:
            # Try to create an instance with no args
            instance = cls()
            if hasattr(instance, "name") and isinstance(instance.name, str):
                return instance.name
        except (TypeError, ValueError, AttributeError) as exc:
            logger.debug("Failed to get metric name from instance: %s", exc)
            # If we can't create an instance, fall back to class attribute

    # For legacy metrics or if the instance approach didn't work, use the class attribute
    if hasattr(cls, "name") and not inspect.ismethod(cls.name):
        name_attr = cls.name
        if isinstance(name_attr, str):
            return name_attr

    # If all else fails, raise an error
    raise AttributeError(f"Cannot determine name for metric class {cls}")


def register_metric(cls: type) -> type:
    """
    Decorator to register a metric class.
    Accepts both new Metric subclasses and legacy metric subclasses.

    Usage:
        @register_metric
        class MyMetric(Metric):
            name = "my_metric"
            ...
    """
    if not _is_metric_like(cls):
        raise TypeError("Only metric-like classes (with a 'name' attribute) can be registered")

    try:
        metric_name = _get_metric_name(cls)
    except AttributeError as e:
        raise AttributeError(f"Registered metric must have a determinable 'name' attribute: {e}")

    # If it's already a subclass of our new Metric, register directly
    if issubclass(cls, Metric):
        _registry[metric_name] = cls
        return cls

    # Otherwise, it's a legacy metric - we need to wrap it
    # Create an adapter class that inherits from our Metric
    class LegacyMetricAdapter(Metric):
        """Adapter that wraps a legacy metric to conform to the new Metric interface."""

        def __init__(self) -> None:
            # Instantiate the legacy metric
            self._legacy_metric = cls()
            # Copy over the name
            self.name = getattr(cls, "name", "unknown_metric")
            # Copy requires_api_key if it exists
            if hasattr(self._legacy_metric, "requires_api_key"):
                self.requires_api_key = self._legacy_metric.requires_api_key  # type: ignore[attr-defined]

        def evaluate(self, case: EvaluationCase) -> MetricResult:
            """
            Convert EvaluationCase to TestCase and EvalResult, then call the legacy metric.
            """
            from rag_score.core.types import EvalResult, RetrievedChunk

            # Convert EvaluationCase to TestCase
            test_case = case.to_test_case()

            # Build EvalResult from EvaluationCase
            # We need to reconstruct RetrievedChunk objects from the stored data
            retrieved_chunks: list[RetrievedChunk] = []
            if case.metadata.get("retrieved_doc_ids") is not None:
                # We have stored doc_ids and possibly scores
                doc_ids = case.metadata["retrieved_doc_ids"]
                texts = case.contexts  # contexts field stores the texts
                scores = case.metadata.get("retrieved_scores", [None] * len(texts))

                for doc_id, text, score in zip(doc_ids, texts, scores):
                    retrieved_chunks.append(RetrievedChunk(doc_id=doc_id, text=text, score=score))
            else:
                # Fallback: assume contexts are texts with unknown doc_ids
                for text in case.contexts:
                    retrieved_chunks.append(RetrievedChunk(doc_id=None, text=text, score=None))

            eval_result = EvalResult(
                evaluation_id="",  # We don't store this in EvaluationCase; generate a placeholder
                run_id="",  # We don't store run_id in EvaluationCase
                test_case_id=case.id,
                retrieved_context=retrieved_chunks,
                generated_answer=case.answer,
                retrieval_latency_ms=case.metadata.get("retrieval_latency_ms"),
                generation_latency_ms=case.metadata.get("generation_latency_ms"),
                total_tokens=case.metadata.get("total_tokens"),
                estimated_cost_usd=case.metadata.get("estimated_cost_usd"),
                error=case.metadata.get("error"),
            )

            # Call the legacy metric's score method (which is async)
            import asyncio

            try:
                # Try to run the async score method
                score = asyncio.run(self._legacy_metric.score(test_case, eval_result))  # type: ignore[no-any-return]
            except AttributeError:
                # If the legacy metric doesn't have an async score method, try the sync one
                score = self._legacy_metric.score(test_case, eval_result)  # type: ignore[no-any-return]

            # Return a MetricResult
            return MetricResult(
                score_id="",  # We don't have an ID from the legacy metric
                evaluation_id="",  # We don't store this
                metric_name=self.name,
                score_value=score,
                judge_reasoning=None,  # Legacy metrics don't provide reasoning by default
            )

    # Register the adapter
    _registry[metric_name] = LegacyMetricAdapter
    return LegacyMetricAdapter


def get_metric(name: str) -> type[Metric]:
    """
    Get a metric class by name from the registry.
    """
    return _registry[name]


def list_metrics() -> list[str]:
    """
    List all registered metric names.
    """
    return list(_registry.keys())
