import gc
from collections import defaultdict
from typing import Any, types

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from lightning.fabric.utilities.rank_zero import rank_zero_only
from lightning.fabric.wrappers import (
    _FabricOptimizer,
)
from torch import Tensor

from foundry.callbacks.callback import BaseCallback

_DEFAULT_STATISTICS = types.MappingProxyType(
    {
        "mean": torch.mean,
        "std": torch.std,
        "norm": torch.norm,
        "max": torch.amax,
        "min": torch.amin,
    }
)
"""Summary statistics to log for gradients, weights, and activations."""

_DEFAULT_HISTOGRAMS = types.MappingProxyType(
    {
        "activations": lambda x: np.histogram(
            x.abs().to(torch.float32).cpu(), bins=40, range=(0, 10)
        ),
        "grads": lambda x: np.histogram(
            x.abs().to(torch.float32).cpu(), bins=40, range=(0, 1)
        ),
        "weights": lambda x: np.histogram(
            x.abs().to(torch.float32).cpu(), bins=40, range=(0, 1)
        ),
    }
)
"""Default histograms to log for activations, gradients, and weights."""


class ActivationsGradientsWeightsTracker(BaseCallback):
    """Fabric callback to track gradients, activations, and weights during training.

    This callback logs gradient, weight, and activation statistics at specified intervals.
    Integrates with FabricTrainer through the BaseCallback interface.

    Args:
        log_freq (int): Frequency of logging (every N steps). Defaults to 100.
        log_grads (bool): Whether to log gradient statistics. Defaults to True.
        log_weights (bool): Whether to log weight statistics. Defaults to True.
        log_activations (bool): Whether to log activation statistics. Defaults to True.
        keep_cache (bool): Whether to keep a local cache of all logged stats. Defaults to False.
        filter_grads (callable): Function (name, param) -> bool to filter gradient tracking. None means all.
        filter_weights (callable): Function (name, param) -> bool to filter weight tracking. None means all.
        filter_activations (callable): Function (name, module) -> bool to filter activation tracking.
            one means default types (Linear, Conv1d, Conv2d, MultiheadAttention).
    """

    def __init__(
        self,
        log_freq: int = 100,
        log_grads: dict[str, callable] = _DEFAULT_STATISTICS,
        log_weights: dict[str, callable] = _DEFAULT_STATISTICS,
        log_activations: dict[str, callable] = _DEFAULT_STATISTICS,
        log_histograms: dict[str, callable] = _DEFAULT_HISTOGRAMS,
        keep_cache: bool = False,
        filter_grads: callable = None,
        filter_weights: callable = None,
        filter_activations: callable = None,
    ):
        super().__init__()
        self.log_freq = log_freq
        self.log_grads = log_grads
        self.log_weights = log_weights
        self.log_activations = log_activations
        self.log_histograms = log_histograms
        self.keep_cache = keep_cache
        self.filter_grads = filter_grads
        self.filter_weights = filter_weights
        self.filter_activations = filter_activations

        self._hooks = []  # Store activation hooks for cleanup
        self._temp_cache = {"scalars": {}, "histograms": {}}
        self._cache = defaultdict(list)
        if not self.keep_cache:
            self.log_histograms = {}

    @rank_zero_only
    def on_fit_start(self, trainer):
        """Initialize the callback and register activation hooks."""
        # Check that we either have loggers attached or keep_cache is True, otherwise the
        #  data will be computed but not logged.
        if not self.keep_cache and not trainer.fabric.loggers:
            raise ValueError(
                "TrainingHealthTracker requires loggers or keep_cache=True. "
                "Otherwise the data will be computed but not logged."
            )

    @rank_zero_only
    def on_train_batch_start(self, trainer, batch: Any, batch_idx: int):
        step = trainer.state["global_step"]
        model = trainer.state["model"]
        if (self.log_activations or "activations" in self.log_histograms) and (
            step % self.log_freq == 0
        ):
            self._register_activation_hooks(model, step)

    @rank_zero_only
    def on_before_optimizer_step(self, trainer, optimizer: _FabricOptimizer, **kwargs):
        """Log gradients, weights, and activations before optimizer step."""
        step = trainer.state["global_step"]

        if step % self.log_freq == 0:
            model = trainer.state["model"]

            # Collect weight & gradient stats
            _should_log_some_grads = self.log_grads or ("grads" in self.log_histograms)
            _should_log_some_weights = self.log_weights or (
                "weights" in self.log_histograms
            )
            if _should_log_some_grads or _should_log_some_weights:
                self._collect_parameter_stats(model, step)

            # Log all collected stats at once using trainer's fabric instance
            if len(self._temp_cache["scalars"]) > 0 and trainer.fabric.loggers:
                trainer.fabric.log_dict(
                    self._temp_cache["scalars"],
                    step=step,
                )

            if self.keep_cache:
                self._cache["step"].append(torch.tensor(step))
                for key, value in self._temp_cache["scalars"].items():
                    self._cache[key].append(value)
                for key, value in self._temp_cache["histograms"].items():
                    if key.endswith("hist"):
                        self._cache[key].append(value)

    def on_train_batch_end(self, trainer, **kwargs):
        """Called at the end of a training batch - clear temporary cache."""
        self._temp_cache["scalars"].clear()
        self._temp_cache["histograms"].clear()
        self._remove_activation_hooks()

    def on_fit_end(self, trainer, **kwargs):
        """Clean up activation hooks at the end of training."""
        self._remove_activation_hooks()

    def on_validation_epoch_start(self, trainer):
        # Temporarily remove any hooks for validation
        self._remove_activation_hooks()

    @rank_zero_only
    def on_save_checkpoint(self, trainer, state: dict[str, Any]):
        self._remove_activation_hooks()

    def _collect_parameter_stats(self, model, step: int):
        """Collect gradient and weight statistics in a single parameter iteration."""
        cache = self._temp_cache  # alias

        for name, param in model.named_parameters():
            # Gradient stats
            if (
                (self.log_grads or "grads" in self.log_histograms)
                and param.grad is not None
                and self._should_track_grad(name)
            ):
                grad = param.grad.detach()
                for stat_name, stat_fn in self.log_grads.items():
                    cache["scalars"]["grads/" + name + "/" + stat_name] = stat_fn(grad)
                if "grads" in self.log_histograms:
                    counts, bin_edges = self.log_histograms["grads"](grad)
                    cache["histograms"]["grads/" + name + "/hist"] = counts
                    cache["histograms"]["grads/" + name + "/hist_bin_edges"] = bin_edges

            # Weight stats
            if (
                self.log_weights or "weights" in self.log_histograms
            ) and self._should_track_weight(name):
                for stat_name, stat_fn in self.log_weights.items():
                    cache["scalars"]["weights/" + name + "/" + stat_name] = stat_fn(
                        param.data
                    )
                if "weights" in self.log_histograms:
                    counts, bin_edges = self.log_histograms["weights"](param.data)
                    cache["histograms"]["weights/" + name + "/hist"] = counts
                    cache["histograms"]["weights/" + name + "/hist_bin_edges"] = (
                        bin_edges
                    )

    def _should_track_grad(self, name: str) -> bool:
        """Check if we should track gradients for this parameter."""
        if self.filter_grads is None:
            return True
        return self.filter_grads(name)

    def _should_track_weight(self, name: str) -> bool:
        """Check if we should track weights for this parameter."""
        if self.filter_weights is None:
            return True
        return self.filter_weights(name)

    def _should_track_activation(self, name: str, module_type: type[nn.Module]) -> bool:
        """Check if we should track activations for this module."""
        if self.filter_activations is None:
            return True
        return self.filter_activations(name, module_type)

    def _register_activation_hooks(self, model, step: int):
        """Register forward hooks to accumulate activations."""
        cache = self._temp_cache  # alias

        def create_activation_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor) and (step % self.log_freq == 0):
                    output = output.detach()
                    for stat_name, stat_fn in self.log_activations.items():
                        cache["activations/" + name + "/" + stat_name] = stat_fn(output)
                    if "activations" in self.log_histograms:
                        counts, bin_edges = self.log_histograms["activations"](output)
                        cache["histograms"]["activations/" + name + "/hist"] = counts
                        cache["histograms"][
                            "activations/" + name + "/hist_bin_edges"
                        ] = bin_edges

            return hook

        # Register hooks for filtered modules
        for name, module in model.named_modules():
            if self._should_track_activation(name, type(module)):
                hook = module.register_forward_hook(create_activation_hook(name))
                self._hooks.append(hook)

    def _remove_activation_hooks(self):
        """Remove activation hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __del__(self):
        self._remove_activation_hooks()
        del self._temp_cache
        del self._cache
        gc.collect()


def plot_tensor_hist(
    hist_values: Float[Tensor, "N M"],
    name: str = "",
    norms: Float[Tensor, "N"] = None,
    steps: Int[Tensor, "N"] = None,
    log_scale: bool = True,
) -> plt.Figure:
    """
    Plot a histogram of tensor values over time, optionally including norm values.

    Args:
        hist_values: Tensor of shape (N, M) containing histogram values for N steps and M bins.
        name: Title for the plot, usually the name of the parameter being plotted.
        norms: Optional tensor of shape (N,) containing norm values for each step.
        steps: Optional tensor of shape (N,) containing step indices. If None, uses range(N).
        log_scale: If True, applies log1p to histogram values before plotting.

    Returns:
        A matplotlib Figure object containing the plotted histogram.

    Example:
        >>> hist_values = torch.randn(100, 50)  # 100 steps, 50 bins
        >>> norms = torch.norm(hist_values, dim=1)
        >>> fig = plot_tensor_hist(hist_values, name="Weight Distribution", norms=norms)
        >>> plt.show()
    """
    font_size = 8
    with plt.rc_context({"font.size": font_size}):
        n_steps, n_bins = hist_values.shape  # (N, M)
        if log_scale:
            hist_values = np.log1p(hist_values)
        if steps is None:
            steps = np.arange(n_steps)
        fig, ax = plt.subplots(
            figsize=(6, 2), constrained_layout=True
        )  # Added constrained_layout
        mat = ax.matshow(hist_values.T, aspect="auto")
        ax.set_xlabel("step")

        # Get the automatically determined tick positions from matplotlib
        locs = ax.get_xticks()
        valid_locs = locs[(locs >= 0) & (locs < n_steps)].astype(int)
        ax.set_xticks(valid_locs)
        ax.set_xticklabels(steps[valid_locs])
        ax.set_ylabel("bins")

        # Create twin axis
        if norms is not None:
            ax2 = ax.twinx()
            ax2.plot(np.arange(len(norms)), norms, color="black")
            ax2.set_ylabel("norm")
            ax2.set_xlim(0, n_steps - 1)
            ax2.set_ylim(min(norms), max(norms))  # Independent scaling
            ax2.set_xticks(valid_locs)
            ax2.set_xticklabels(steps[valid_locs])

        # Add colorbar - constrained_layout will handle spacing automatically
        cbar = plt.colorbar(mat, ax=ax)
        cbar.ax.set_ylabel("log(1+count)" if log_scale else "count")

        ax.set_xlim(0, n_steps - 1)
        ax.set_ylim(0, n_bins - 1)
        if name:
            ax.set_title(name, pad=20, fontsize=8)

        return fig


def plot_tensor_stats(
    steps: Int[Tensor, "N"],
    mean: Float[Tensor, "N"] = None,
    std: Float[Tensor, "N"] = None,
    min_val: Float[Tensor, "N"] = None,
    max_val: Float[Tensor, "N"] = None,
    norm: Float[Tensor, "N"] = None,
    name: str = "",
    height_ratios: tuple[float, float] = (5, 1),
):
    """
    Plot comprehensive statistics with mean/std/min/max in top panel and norm in bottom panel.

    Args:
        steps: Training step indices
        mean: Mean values over time (optional)
        std: Standard deviation values over time (optional, requires mean)
        min_val: Minimum values over time (optional)
        max_val: Maximum values over time (optional)
        norm: Norm values over time (optional)
        name: Title for the plot, usually the name of the parameter being plotted.
        height_ratios: Relative heights of (stats_panel, norm_panel)

    Returns:
        matplotlib Figure object
    """
    # Determine what to plot
    has_stats = any([mean is not None, min_val is not None, max_val is not None])
    has_norm = norm is not None

    if not has_stats and not has_norm:
        raise ValueError(
            "At least one of mean, min_val, max_val, or norm must be provided"
        )

    # Create subplot layout based on available data
    if has_stats and has_norm:
        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=(5, 3),
            gridspec_kw={"height_ratios": height_ratios},
            sharex=True,
            constrained_layout=True,
        )
        norm_ax = ax2
        stats_ax = ax1
    elif has_stats:
        fig, ax1 = plt.subplots(figsize=(5, 3))
        stats_ax = ax1
        norm_ax = None
    else:  # only norm
        fig, ax2 = plt.subplots(figsize=(5, 3))
        norm_ax = ax2
        stats_ax = None

    # Top panel: statistics (if available)
    if has_stats and stats_ax is not None:
        if mean is not None:
            stats_ax.plot(steps, mean, label="mean", color="C0")
            if std is not None:
                stats_ax.fill_between(
                    steps, mean - std, mean + std, alpha=0.2, color="C0", label="±1 std"
                )

        if min_val is not None and max_val is not None:
            stats_ax.plot(
                steps, min_val, "--", color="gray", alpha=0.7, label="min/max"
            )
            stats_ax.plot(steps, max_val, "--", color="gray", alpha=0.7)
        elif min_val is not None:
            stats_ax.plot(steps, min_val, "--", color="gray", alpha=0.7, label="min")
        elif max_val is not None:
            stats_ax.plot(steps, max_val, "--", color="gray", alpha=0.7, label="max")

        stats_ax.ticklabel_format(style="plain", useOffset=False)
        stats_ax.set_ylabel("Stats", labelpad=0)
        if name:
            stats_ax.set_title(name, pad=5, fontsize=9)
        stats_ax.grid(True, alpha=0.3)
        stats_ax.legend(loc="upper right", bbox_to_anchor=(1, 1), ncol=2)

        # Set xlabel only if this is the only panel
        if not has_norm:
            stats_ax.set_xlabel("Step")

    # Bottom panel: norm (if available)
    if has_norm and norm_ax is not None:
        norm_ax.plot(steps, norm, label="norm", color="C1")
        norm_ax.set_ylabel("Norm", labelpad=0)
        norm_ax.set_xlabel("Step")
        norm_ax.grid(True, alpha=0.3)
        norm_ax.legend(loc="upper right", bbox_to_anchor=(1, 1))
        norm_ax.ticklabel_format(style="plain", useOffset=False)

        # Set title if this is the only panel and no stats panel exists
        if not has_stats and name:
            norm_ax.set_title(name, pad=5, fontsize=9)

    plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    return fig
