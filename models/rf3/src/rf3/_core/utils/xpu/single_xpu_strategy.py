"""Lightning Fabric strategy for single XPU device.

https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html
"""

import torch
from lightning.fabric.plugins import CheckpointIO
from lightning.fabric.plugins.precision import Precision
from lightning.fabric.strategies import SingleDeviceStrategy
from lightning.fabric.utilities.types import _DEVICE


class SingleXPUStrategy(SingleDeviceStrategy):
    """Strategy for training/inference on a single Intel XPU device.

    This strategy extends SingleDeviceStrategy to properly handle XPU devices.
    """

    strategy_name = "xpu_single"

    def __init__(
        self,
        device: _DEVICE = "xpu:0",
        checkpoint_io: CheckpointIO | None = None,
        precision_plugin: Precision | None = None,
    ) -> None:
        """Initialize the single XPU strategy.

        Args:
            device: The XPU device to use. Defaults to "xpu:0".
            checkpoint_io: Plugin for checkpoint I/O.
            precision_plugin: Plugin for precision handling (set via _precision property).

        Raises:
            RuntimeError: If XPU devices are not available.
        """
        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            msg = "`SingleXPUStrategy` requires XPU devices to run"
            raise RuntimeError(msg)

        super().__init__(
            device=device,
            checkpoint_io=checkpoint_io,
        )
        # Precision is handled via the _precision property in newer Lightning versions
        if precision_plugin is not None:
            self._precision = precision_plugin
