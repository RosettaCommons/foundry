import inspect
from abc import ABC, abstractmethod
from functools import cached_property

import hydra
from atomworks.common import exists
from atomworks.ml.utils import error, nested_dict
from beartype.typing import Any
from omegaconf import DictConfig
from toolz import keymap

from foundry.utils.ddp import RankedLogger

ranked_logger = RankedLogger(__name__, rank_zero_only=True)


def instantiate_metric_manager(
    metrics_cfg: dict[str, Any] | DictConfig,
) -> "MetricManager":
    """Instantiate a MetricManager from a dictionary of metrics.

    Args:
        metrics: A dictionary where keys are metric names and values are Hydra configurations for the metrics.
    """
    metrics = {}
    for name, cfg in metrics_cfg.items():
        metric = hydra.utils.instantiate(cfg)
        if not isinstance(metric, Metric):
            raise TypeError(f"{name} must be a Metric instance")
        ranked_logger.info(f"Adding metric {name} to the validation metrics...")
        metrics[name] = metric
    return MetricManager(metrics)


class MetricInputError(Exception):
    """Exception raised when a metric fails to compute."""


class MetricManager:
    """Manages and computes a set of Metrics, where each Metric inherits from the Metric class.

    For model validation, additional metrics can be added through the Hydra configuration; they
    will be computed with the __call__ method automatically.

    For example, during AF-3, Metrics will receive `network_input`, `network_output`, `extra_info`,
    `ground_truth_atom_array_stack`, and `predicted_atom_array_stack` as input arguments.

    Example:
        >>> class ExampleMetric(Metric):
        ...     @cached_property
        ...     def kwargs_to_compute_args(self):
        ...         return {"x": "x", "y": "y", "extra_info": "extra_info"}
        ...
        ...     def compute(self, x, y, extra_info):
        ...         return {"value": x + y}
        >>> metric = ExampleMetric()
        >>> manager = MetricManager({"my_metric": metric}, raise_errors=True)
        >>> manager(x=1, y=2, extra_info={"example_id": "123"})
        {'example_id': '123', 'my_metric.value': 3}
    """

    def __init__(
        self,
        metrics: dict[str, "Metric"] = {},
        *,
        raise_errors: bool = True,
    ):
        """Initialize the MetricManager with a set of metrics.

        Args:
            raise_errors: Whether to raise errors when a metric fails to compute.
            metrics: A dictionary where keys are metric names and values are Metric instances.
        """
        self.raise_errors = raise_errors
        self.metrics = {}
        for name, metric in metrics.items():
            assert isinstance(
                metric, Metric
            ), f"{name} must be a Metric instance, not {type(metric)}"
            self.metrics[name] = metric

    @classmethod
    def instantiate_from_hydra(
        cls, metrics_cfg: dict[str, Any] | DictConfig
    ) -> "MetricManager":
        """Instantiate a MetricManager from a dictionary of metrics.

        Args:
            metrics_cfg: A dictionary where keys are metric names and values are Hydra configurations for the metrics.
        """
        return instantiate_metric_manager(metrics_cfg)

    @classmethod
    def from_metrics(
        cls,
        metrics: dict[str, "Metric"] | list[tuple[str, "Metric"]],
        raise_errors: bool = True,
    ) -> "MetricManager":
        """Create MetricManager from metric objects.

        Args:
            metrics: Either dict mapping names to Metric objects,
                or list of (name, Metric) tuples.
            raise_errors: Whether to raise errors on metric failures. Defaults to ``True``.
        """
        if isinstance(metrics, list):
            # Convert list of tuples to dict
            metrics = dict(metrics)

        # Validate all are Metric instances
        for name, metric in metrics.items():
            if not isinstance(metric, Metric):
                raise TypeError(
                    f"Metric '{name}' must be a Metric instance, got {type(metric)}"
                )

        return cls(metrics, raise_errors=raise_errors)

    def __repr__(self) -> str:
        """Return a string representation of the MetricManager."""
        return f"MetricManager({', '.join(self.metrics.keys())})"

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        """Compute all metrics and return their results as a dictionary mapping metric names to their computed values."""

        # Extract example_id if it exists
        example_id = nested_dict.get(
            kwargs, key=("extra_info", "example_id"), default=None
        )

        # Initialize results dictionary
        results = {"example_id": example_id}

        for name, metric in self.metrics.items():
            assert name not in results, f"Duplicate metric name: {name}"

            # Add some nice error handling context in case metrics fail
            example_msg = (
                f" for example '{example_id}'" if example_id is not None else ""
            )

            # Decide whether the Metric applies to this batch
            if "extra_info" in kwargs and "metrics_tags" in kwargs["extra_info"]:
                metrics_tags = kwargs["extra_info"]["metrics_tags"]
                metrics_tags = set(metrics_tags)
                if metric.required_tags_all and not metrics_tags.issuperset(
                    metric.required_tags_all
                ):
                    continue
                if metric.required_tags_any and not metrics_tags.intersection(
                    metric.required_tags_any
                ):
                    continue
                if metric.prohibited_tags and metrics_tags.intersection(
                    metric.prohibited_tags
                ):
                    continue

            with error.context(
                msg=f"Computing '{name}' ({type(metric).__name__}){example_msg}",
                raise_error=self.raise_errors,
                exc_types=(MetricInputError, ValueError, TypeError, AttributeError),
            ):
                # ... compute the metric
                metric_result = metric.compute_from_kwargs(**kwargs)

                # ... append 'name' to the keys of the metric result to ensure uniqueness
                if isinstance(metric_result, dict):
                    metric_result = keymap(lambda k: f"{name}.{k}", metric_result)
                    results.update(metric_result)
                elif isinstance(metric_result, list):
                    results[name] = metric_result
                else:
                    raise ValueError("Unexpected result type: expected dict or list.")

        return results


class Metric(ABC):
    """Abstract base class for Foundry metrics.

    Defines a framework for computing metrics based on arbitrary keyword arguments.
    A Metric can specify which batches a `MetricManager` applies it to, based on `tags` in the input batch.

    Args:
        required_tags_all: A set of tags that must all be present in the input batch for this metric to be computed.
        required_tags_any: A set of tags where at least one must be present in the input batch for this metric to be computed.
        prohibited_tags: A set of tags that must not be present in the input batch for this metric to be computed.

    To implement a new metric, subclass this class and implement the `compute` method, at a minimum.
    """

    def __init__(
        self,
        required_tags_all: list[str] | set[str] | None = None,
        required_tags_any: list[str] | set[str] | None = None,
        prohibited_tags: list[str] | set[str] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        # Set required and prohibited tags
        self.required_tags_all = (
            set(required_tags_all) if exists(required_tags_all) else set()
        )
        self.required_tags_any = (
            set(required_tags_any) if exists(required_tags_any) else set()
        )
        self.prohibited_tags = (
            set(prohibited_tags) if exists(prohibited_tags) else set()
        )
        required_tags = self.required_tags_all.union(self.required_tags_any)
        if required_tags.intersection(self.prohibited_tags):
            raise ValueError(
                "Conflicting tags found: required tags and prohibited tags must be disjoint."
            )

        # Check that the 'keys' of the kwargs_to_compute_args are a subset of the 'compute' method signature
        if self.kwargs_to_compute_args:
            assert self.kwargs_to_compute_args.keys() <= self.required_compute_args, (
                f"The keys of kwargs_to_compute_args must be a subset of the 'compute' method signature. "
                f"{self.kwargs_to_compute_args.keys()} is not a subset of {self.required_compute_args}"
            )

        # Check that optional_kwargs are also in the kwargs_to_compute_args
        if self.kwargs_to_compute_args and self.optional_kwargs:
            assert self.optional_kwargs <= set(self.kwargs_to_compute_args.keys()), (
                f"All optional_kwargs must be defined in kwargs_to_compute_args. "
                f"{self.optional_kwargs} is not a subset of {set(self.kwargs_to_compute_args.keys())}"
            )

    @cached_property
    def required_compute_args(self) -> frozenset[str]:
        """Required input keys for this metric"""
        return frozenset(inspect.signature(self.compute).parameters.keys())

    @cached_property
    def required_kwargs(self) -> frozenset[str]:
        """Required input keys for this metric"""
        if self.kwargs_to_compute_args is None:
            return frozenset()

        return frozenset(self.kwargs_to_compute_args.values())

    def compute_from_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        """Run compute with an arbitrary dictionary of input keys and values.

        The 'kwargs_to_compute_args' property here will determine
        where in the kwargs we will look for the values to pass to the compute method.

        Parameters marked in 'optional_kwargs' will only be passed if present in kwargs.
        """
        if self.kwargs_to_compute_args:
            compute_inputs = {}
            for compute_arg, kwargs_key in self.kwargs_to_compute_args.items():
                if compute_arg in self.optional_kwargs:
                    # Optional parameter - only add if present
                    try:
                        compute_inputs[compute_arg] = nested_dict.getitem(
                            kwargs, key=kwargs_key
                        )
                    except KeyError:
                        pass  # Don't pass this parameter to compute()
                else:
                    # Required parameter - use getitem (will raise if missing)
                    compute_inputs[compute_arg] = nested_dict.getitem(
                        kwargs, key=kwargs_key
                    )
        else:
            # If kwargs_to_compute_args is not defined, use kwargs directly
            compute_inputs = kwargs
        return self.compute(**compute_inputs)

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        """Map input keys to a flat dictionary.

        If not implemented, we return None, and pass the kwargs directly to the compute method.

        Override e.g. as:
        ```python
        @cached_property
        def kwargs_to_compute_args(self) -> dict[str, Any]:
            return {
                "y_true": ("network_input", "coords_unnoised"),
                "y_pred": ("network_output", "coords_pred"),
                "extra_info": ("extra_info",),
            }
        ```
        """
        return None

    @property
    def optional_kwargs(self) -> frozenset[str]:
        """Set of compute argument names that are optional.

        Optional parameters will only be passed to compute() if present in kwargs.
        The compute() method should have sensible defaults for these parameters.

        Override e.g. as:
        ```python
        @property
        def optional_kwargs(self) -> frozenset[str]:
            return frozenset(["confidence_indices", "interfaces_to_score"])
        ```
        """
        return frozenset()

    @abstractmethod
    def compute(self, **kwargs: Any) -> dict[str, Any] | list[dict[str, Any]]:
        """Implement actual metric calculation here

        Override e.g. as:
        ```python
        def compute(self, y_true, y_pred, extra_info):
            print(extra_info)
            return lddt(y_true, y_pred, thres=self.custom_thresholds)
        ```
        """
        raise NotImplementedError
