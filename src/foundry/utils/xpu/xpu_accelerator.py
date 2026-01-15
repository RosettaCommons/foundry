"""XPU Accelerator for Intel XPU devices."""

from typing import Any

import torch
from lightning.fabric.accelerators import Accelerator


class XPUAccelerator(Accelerator):
    """Accelerator for Intel XPU devices.

    This accelerator enables training and inference on Intel GPUs using
    PyTorch's native XPU support (torch.xpu).
    """

    @property
    def name(self) -> str:
        """Return the name of this accelerator."""
        return "xpu"

    @staticmethod
    def setup_device(device: torch.device) -> None:
        """Set up the specified XPU device.

        Args:
            device: The torch device to set up.

        Raises:
            RuntimeError: If device is not an XPU device.
        """
        if device.type != "xpu":
            msg = f"Device should be xpu, got {device} instead"
            raise RuntimeError(msg)

        torch.xpu.set_device(device)

    @staticmethod
    def parse_devices(devices: str | list | torch.device) -> list:
        """Parse devices specification for XPU training.

        Args:
            devices: Device specification (int, list of ints, or string).

        Returns:
            List of device indices.
        """
        if isinstance(devices, list):
            return devices
        return [devices]

    @staticmethod
    def get_parallel_devices(devices: list) -> list[torch.device]:
        """Generate a list of parallel XPU devices.

        Args:
            devices: List of device indices.

        Returns:
            List of torch.device objects for XPU.
        """
        return [torch.device("xpu", idx) for idx in devices]

    @staticmethod
    def auto_device_count() -> int:
        """Return the number of available XPU devices."""
        return torch.xpu.device_count()

    @staticmethod
    def is_available() -> bool:
        """Check if XPU is available."""
        return hasattr(torch, "xpu") and torch.xpu.is_available()

    @staticmethod
    def get_device_stats(device: str | torch.device) -> dict[str, Any]:
        """Return XPU device statistics.

        Currently returns an empty dict as XPU stats API may vary.

        Args:
            device: The device to get stats for.

        Returns:
            Dictionary of device statistics.
        """
        del device  # Unused
        return {}

    def teardown(self) -> None:
        """Clean up XPU accelerator resources."""
        # Empty implementation required by base class
        pass
