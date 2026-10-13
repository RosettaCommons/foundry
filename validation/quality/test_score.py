"""Independent checks for the paired-study analysis, without model checkpoints."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "quality_score", Path(__file__).with_name("score.py")
)
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)


def test_alignment_removes_rigid_motion_but_not_reflection():
    x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    rotation = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert score.aligned_rmsd(x, x @ rotation + 20) < 1e-12
    reflected = x.copy()
    reflected[:, 0] *= -1
    assert score.aligned_rmsd(x, reflected) > 0.1


def test_alignment_rejects_missing_coordinates():
    with pytest.raises(AssertionError):
        score.aligned_rmsd(np.full((4, 3), np.nan), np.zeros((4, 3)))


def test_constant_paired_effect_has_exact_interval():
    assert score.interval([0.25] * 16) == [0.25, 0.25]
    assert score.interval([0.25] * 16, [1, 2, 3, 4] * 4) == [0.25, 0.25]


def test_seed_clustering_does_not_treat_lengths_as_independent():
    effects = [-1, 0, 0, 1] * 4
    naive = score.interval(effects)
    clustered = score.interval(effects, [1, 2, 3, 4] * 4)
    assert clustered[1] - clustered[0] > naive[1] - naive[0]


def test_missing_artifacts_are_persisted_before_scoring(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "quality_audit", Path(__file__).with_name("audit.py")
    )
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    result = audit.inventory(
        tmp_path, {"arms": ["a"], "mpnn_sequences_per_backbone": 2}, ["L64_s11"]
    )
    assert result["incomplete_records"] == 3
    assert len(result["records"][0]["missing"]) == 3
    assert json.loads((tmp_path / "quality_inventory.json").read_text()) == result
