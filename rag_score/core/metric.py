"""
Metric base class and registry for the plugin-style metric system.
"""

from __future__ import annotations

import inspect
from typing import Callable, Dict, List, Optional, Type

from .models import EvaluationCase, MetricResult


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
_registry: Dict[str, Type[Metric]] = {}


def register_metric(cls: Type[Metric]) -> Type[Metric]:
    """
    Decorator to register a metric class.

    Usage:
        @register_metric
        class MyMetric(Metric):
            name = "my_metric"
            ...
    """
    if not inspect.isclass(cls) or not issubclass(cls, Metric):
        raise TypeError("Only subclasses of Metric can be registered")
    if not hasattr(cls, "name"):
        raise AttributeError("Registered metric must have a 'name' attribute")
    _registry[cls.name] = cls
    return cls


def get_metric(name: str) -> Type[Metric]:
    """
    Get a metric class by name from the registry.
    """
    return _registry[name]


def list_metrics() -> List[str]:
    """
    List all registered metric names.
    """
    return list(_registry.keys())