"""Automatic dispatch and persistent cache policy stay explicit and predictable."""

import torch
from rfd3.engine import RFD3InferenceConfig
from rfd3.inference.runtime import configure_compile_cache, select_kernel_backend


def test_default_policy():
    config = RFD3InferenceConfig()
    assert config.compile_model
    assert config.compile_cuda_graphs
    assert config.compile_shape_policy == "buckets"
    assert config.inference_kernel_backend == "auto"
    assert not config.low_memory_mode


def test_environment_cache_default_in_python_and_yaml(tmp_path, monkeypatch):
    from pathlib import Path

    from omegaconf import OmegaConf

    config_path = (
        Path(__file__).parents[1] / "configs/inference_engine/rfdiffusion3.yaml"
    )
    monkeypatch.delenv("RFD3_COMPILE_CACHE_DIR", raising=False)
    assert RFD3InferenceConfig().compile_cache_dir is None
    assert OmegaConf.load(config_path).compile_cache_dir is None
    cache = str(tmp_path / "persistent")
    monkeypatch.setenv("RFD3_COMPILE_CACHE_DIR", cache)
    assert RFD3InferenceConfig().compile_cache_dir == cache
    assert OmegaConf.load(config_path).compile_cache_dir == cache
    assert (
        RFD3InferenceConfig(compile_cache_dir="explicit").compile_cache_dir
        == "explicit"
    )
    assert RFD3InferenceConfig(compile_cache_dir=None).compile_cache_dir is None


def test_unset_cache_leaves_framework_environment_unchanged(monkeypatch):
    import os

    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    assert configure_compile_cache(None) == {}
    assert "TORCHINDUCTOR_CACHE_DIR" not in os.environ
    assert "TRITON_CACHE_DIR" not in os.environ


def test_auto_backend_compatibility(monkeypatch):
    monkeypatch.setenv("RFD3_LOW_MEMORY_MODE", "0")
    monkeypatch.setattr("importlib.util.find_spec", lambda _: object())
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (8, 6))
    options = dict(
        device=torch.device("cuda"),
        compile_model=False,
        inference_batch_size=1,
        low_memory_mode=False,
    )
    assert select_kernel_backend(**options) == "triton"
    assert select_kernel_backend(**(options | dict(compile_model=True))) == "torch"
    assert select_kernel_backend(**(options | dict(inference_batch_size=2))) == "triton"
    assert (
        select_kernel_backend(**(options | dict(device=torch.device("cpu")))) == "torch"
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (7, 5))
    assert select_kernel_backend(**options) == "torch"
    monkeypatch.setattr("importlib.util.find_spec", lambda _: None)
    assert select_kernel_backend(**options) == "torch"


def test_compiled_transition_policy_is_scoped_to_measured_hardware_and_batch(monkeypatch):
    monkeypatch.setenv("RFD3_LOW_MEMORY_MODE", "0")
    monkeypatch.setattr("importlib.util.find_spec", lambda _: object())
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, 0))
    options = dict(
        device=torch.device("cuda"), compile_model=True,
        inference_batch_size=4, low_memory_mode=False,
    )
    assert select_kernel_backend(**options) == "triton-transition"
    assert select_kernel_backend(**(options | dict(inference_batch_size=1))) == "torch"
    assert select_kernel_backend(**(options | dict(inference_batch_size=3))) == "torch"
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (9, 0))
    assert select_kernel_backend(**options) == "torch"
    monkeypatch.setattr("importlib.util.find_spec", lambda _: None)
    assert select_kernel_backend(**options) == "torch"


def test_cache_location_and_environment_override(tmp_path, monkeypatch):
    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    monkeypatch.delenv("TORCHINDUCTOR_FX_GRAPH_CACHE", raising=False)
    cache = tmp_path / "persistent"
    assert configure_compile_cache(cache) == {"fx_graph_cache": True}
    assert cache.is_dir()
    import os

    assert os.environ["TRITON_CACHE_DIR"] == str(cache / "triton")
    override = tmp_path / "environment"
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(override))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton-override"))
    monkeypatch.setenv("TORCHINDUCTOR_FX_GRAPH_CACHE", "0")
    assert configure_compile_cache(cache) == {"fx_graph_cache": False}
    assert override.is_dir()
    assert os.environ["TRITON_CACHE_DIR"] == str(tmp_path / "triton-override")
    assert configure_compile_cache(None) == {}


def test_framework_generated_temp_location_does_not_override_engine(
    tmp_path, monkeypatch
):
    from torch._inductor.runtime.cache_dir_utils import default_cache_dir

    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", default_cache_dir())
    cache = tmp_path / "persistent"
    configure_compile_cache(cache)
    assert cache.is_dir()
