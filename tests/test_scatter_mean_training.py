"""Regression checks at the actual optional RFD3 and RF3 training callsites."""

import runpy
from pathlib import Path

import pytest
import torch

pytest.importorskip("rfd3.model.layers.blocks")
pytest.importorskip("rf3.model.layers.pairformer_layers")

AUDIT = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "examples/validate_pooling_training.py")
)


@pytest.mark.parametrize("module", ["rfd3", "rf3_pairformer", "rf3_diffusion"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_pooling_training_matches_quantized_activation_oracle(module, dtype):
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        result = AUDIT["compare_case"](
            module,
            group_size=23,
            batch=1,
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=dtype,
            steps=2,
        )
        assert result["fixed_pass"], result
    finally:
        torch.set_num_threads(previous_threads)
