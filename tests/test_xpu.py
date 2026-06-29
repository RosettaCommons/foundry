"""Unit tests for the Intel-XPU Lightning plugins in foundry.utils.xpu.

These plugins target Intel GPUs (``torch.xpu``), which are absent in CI, so the
parts that actually touch XPU hardware (``setup_device`` success, autocast
contexts, device counts) are not exercised. The device-independent contracts are
pinned here: device parsing, the no-XPU guards (which must raise on a CPU box),
and the precision→dtype mapping + tensor conversion.
"""

import pytest
import torch

from foundry.utils.xpu.single_xpu_strategy import SingleXPUStrategy
from foundry.utils.xpu.xpu_accelerator import XPUAccelerator
from foundry.utils.xpu.xpu_precision import XPUMixedPrecision


def test_accelerator_name():
    assert XPUAccelerator.name() == "xpu"


def test_accelerator_not_available_on_non_xpu_host():
    assert XPUAccelerator.is_available() is False


def test_parse_devices_passes_lists_through_and_wraps_scalars():
    assert XPUAccelerator.parse_devices([0, 1]) == [0, 1]
    assert XPUAccelerator.parse_devices(0) == [0]


def test_get_parallel_devices_builds_xpu_devices():
    devices = XPUAccelerator.get_parallel_devices([0, 1])
    assert devices == [torch.device("xpu", 0), torch.device("xpu", 1)]


def test_get_device_stats_is_empty():
    assert XPUAccelerator.get_device_stats("xpu:0") == {}


def test_setup_device_rejects_non_xpu_device():
    with pytest.raises(RuntimeError, match="Device should be xpu"):
        XPUAccelerator.setup_device(torch.device("cpu"))


def test_single_xpu_strategy_requires_xpu():
    with pytest.raises(RuntimeError, match="requires XPU devices"):
        SingleXPUStrategy()


def test_mixed_precision_maps_precision_to_dtype():
    assert XPUMixedPrecision("16-mixed")._desired_input_dtype == torch.float16
    assert XPUMixedPrecision("bf16-mixed")._desired_input_dtype == torch.bfloat16


def test_mixed_precision_rejects_invalid_precision():
    with pytest.raises(ValueError, match="Invalid precision"):
        XPUMixedPrecision("32-true")


def test_mixed_precision_converts_only_float_tensors():
    plugin = XPUMixedPrecision("bf16-mixed")

    converted = plugin.convert_input(torch.ones(3, dtype=torch.float32))
    assert converted.dtype == torch.bfloat16

    untouched = torch.ones(3, dtype=torch.int64)
    assert plugin.convert_input(untouched).dtype == torch.int64


if __name__ == "__main__":
    pytest.main(["-v", __file__])
