"""Metrics for model evaluation.

This module provides the base metric framework.
"""

from rfd3._core.metrics.metric import Metric, MetricInputError, MetricManager

__all__ = [
    "Metric",
    "MetricManager",
    "MetricInputError",
]
