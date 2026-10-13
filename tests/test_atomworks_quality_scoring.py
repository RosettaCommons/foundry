"""Validate the paired-checkpoint audit's numerical gate."""

import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "quality_score",
    Path(__file__).parents[1] / "scripts/compare_atomworks_model_quality.py",
)
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)


@pytest.mark.parametrize("delta,passes", [(0.0, True), (1e-7, True), (1e-3, False)])
def test_tensor_comparison_threshold(tmp_path, delta, passes):
    old = tmp_path / "old.pt"
    new = tmp_path / "new.pt"
    torch.save({"log_probs": torch.tensor([-0.5, -1.0])}, old)
    torch.save({"log_probs": torch.tensor([-0.5 + delta, -1.0])}, new)
    assert score.compare_pt(old, new)["pass_threshold"] is passes


def test_missing_output_and_discrete_changes_fail(tmp_path):
    old = tmp_path / "old.pt"
    new = tmp_path / "new.pt"
    torch.save(
        {"sample": torch.tensor([1, 2]), "log_probs": torch.tensor([-0.5, -1.0])}, old
    )
    torch.save({"sample": torch.tensor([1, 3])}, new)
    result = score.compare_pt(old, new)
    assert not result["pass_threshold"]
    assert len(result["failures"]) == 2


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_matching_nonfinite_outputs_fail(tmp_path, value):
    path = tmp_path / "bad.pt"
    torch.save({"logits": torch.tensor([value])}, path)
    assert not score.compare_pt(path, path)["pass_threshold"]


def test_empty_output_fails(tmp_path):
    path = tmp_path / "empty.pt"
    torch.save({}, path)
    assert not score.compare_pt(path, path)["pass_threshold"]


def test_fixed_residue_constraint(tmp_path):
    inputs, outputs = tmp_path / "inputs.pt", tmp_path / "outputs.pt"
    torch.save(
        {
            "input": {
                "input_features": {
                    "S": torch.tensor([[1, 2]]),
                    "designed_residue_mask": torch.tensor([[True, False]]),
                }
            }
        },
        inputs,
    )
    torch.save(
        {"decoder_features": {"S_sampled": torch.tensor([[3, 2], [4, 2]])}}, outputs
    )
    assert score.fixed_residues_preserved(inputs, outputs)
    torch.save(
        {"decoder_features": {"S_sampled": torch.tensor([[3, 2], [4, 3]])}}, outputs
    )
    assert not score.fixed_residues_preserved(inputs, outputs)


def test_quality_gate_requires_complete_results():
    complete = dict(
        native_quality_gate=True, rf3_fixed_all_pass=True, mpnn_all_pass=True
    )
    assert score.quality_gates_pass(complete)
    assert not score.quality_gates_pass({})
    assert not score.quality_gates_pass({**complete, "rf3_fixed_all_pass": False})
