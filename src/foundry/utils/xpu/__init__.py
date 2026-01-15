"""XPU utilities for Intel GPU support.

XPU support in PyTorch is now native (torch.xpu.is_available()), but Lightning Fabric
requires custom Accelerator, Strategy, and Precision plugins for proper XPU handling.

These components are used directly (not via registry) when XPU is detected:
- XPUAccelerator: Custom accelerator for XPU devices
- SingleXPUStrategy: Strategy for single-device XPU training/inference
- XPUMixedPrecision: Precision plugin with proper XPU autocast support

Usage:
    from foundry.utils.xpu import XPUAccelerator, SingleXPUStrategy, XPUMixedPrecision

    # Check availability
    if XPUAccelerator.is_available():
        strategy = SingleXPUStrategy(precision_plugin=XPUMixedPrecision("bf16-mixed"))

Note:
    The FabricTrainer automatically uses these components when XPU is detected.
    You typically don't need to use them directly unless customizing behavior.
"""

from .single_xpu_strategy import SingleXPUStrategy
from .xpu_accelerator import XPUAccelerator
from .xpu_precision import XPUMixedPrecision

__all__ = ["SingleXPUStrategy", "XPUAccelerator", "XPUMixedPrecision"]
