"""XPU Precision Plugin for Lightning Fabric."""

from contextlib import contextmanager
from typing import Any, Generator, Literal

import torch
from lightning.fabric.plugins.precision import MixedPrecision
from torch import Tensor


class XPUMixedPrecision(MixedPrecision):
    """Mixed precision plugin for Intel XPU devices.

    This overrides the default MixedPrecision plugin to use 'xpu' as the
    device type for torch.autocast instead of 'cuda'.
    """

    def __init__(
        self,
        precision: Literal["16-mixed", "bf16-mixed"] = "bf16-mixed",
    ) -> None:
        """Initialize XPU mixed precision.

        Args:
            precision: The precision mode. "16-mixed" uses float16,
                "bf16-mixed" uses bfloat16. Defaults to "bf16-mixed".

        Raises:
            ValueError: If precision is not "16-mixed" or "bf16-mixed".
        """
        # Determine dtype from precision string
        if precision == "16-mixed":
            dtype = torch.float16
        elif precision == "bf16-mixed":
            dtype = torch.bfloat16
        else:
            msg = f"Invalid precision: {precision}. Must be '16-mixed' or 'bf16-mixed'"
            raise ValueError(msg)

        # Initialize with xpu device type
        super().__init__(precision=precision, device="xpu")
        self._desired_input_dtype = dtype

    @contextmanager
    def forward_context(self) -> Generator[None, None, None]:
        """Context manager for forward pass with XPU autocast."""
        with torch.autocast(device_type="xpu", dtype=self._desired_input_dtype):
            yield

    def convert_input(self, data: Any) -> Any:
        """Convert input data to the appropriate precision.

        Args:
            data: Input data to convert.

        Returns:
            Converted data.
        """
        return self._convert_fp_tensor(data)

    def _convert_fp_tensor(self, data: Any) -> Any:
        """Convert floating point tensors to the desired dtype.

        Args:
            data: Data to convert.

        Returns:
            Converted data if it's a floating point tensor, otherwise unchanged.
        """
        if isinstance(data, Tensor) and data.is_floating_point():
            return data.to(self._desired_input_dtype)
        return data
