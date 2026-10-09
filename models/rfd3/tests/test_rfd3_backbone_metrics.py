"""Sparse clash counting and cached-mass Rg against their dense/biotite references."""

import numpy as np
import pytest
from biotite.structure import AtomArray, gyration_radius
from rfd3.metrics.design_metrics import get_clash_metrics
from rfd3.metrics.metrics_utils import atom_masses


def _protein(n_res, scale, seed):
    rng = np.random.default_rng(seed)
    names = ["N", "CA", "C", "O", "CB"]
    array = AtomArray(n_res * len(names))
    array.res_id = np.repeat(np.arange(1, n_res + 1), len(names))
    array.atom_name = np.tile(names, n_res)
    array.element = np.tile(["N", "C", "C", "O", "C"], n_res)
    array.chain_id[:] = "A"
    # Compact random coordinates give many pairs near and below the thresholds.
    array.coord = (rng.random((array.array_length(), 3)) * scale).astype(np.float32)
    for name in ("is_protein", "is_ligand", "is_motif_atom_unindexed"):
        array.add_annotation(name, bool)
    array.is_protein[:] = True
    array.add_annotation("chain_iid", array.chain_id.dtype)
    array.chain_iid = array.chain_id
    return array


def _dense_clashes(array, threshold, backbone_only):
    """Previous dense implementation, retained as the oracle."""
    resid = array.res_id - array.res_id.min()
    xyz = array.coord
    dists = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    mask = np.triu(np.ones_like(dists), k=1).astype(bool)
    mask[np.abs(resid[:, None] - resid[None, :]) <= 1] = False
    dists[~mask] = 999
    if backbone_only:
        backbone = np.isin(array.atom_name, ["N", "CA", "C"])
        dists[~(backbone[:, None] & backbone[None, :])] = 999
    return int((dists.min(axis=-1) < threshold).sum())


@pytest.mark.parametrize("scale", [8.0, 15.0, 40.0])
@pytest.mark.parametrize("seed", [0, 1])
def test_sparse_clashes_match_dense(scale, seed):
    array = _protein(60, scale, seed)
    metrics = get_clash_metrics(array, clash_threshold=1.5)
    expected = (_dense_clashes(array, 1.5, False), _dense_clashes(array, 1.5, True))
    got = (
        metrics["n_clashing.interresidue_clashes_w_sidechain"],
        metrics["n_clashing.interresidue_clashes_w_backbone"],
    )
    assert got == expected
    if scale == 8.0:
        assert expected[0] > 0  # the dense case actually exercises clashes


def test_sparse_clashes_strict_threshold():
    array = _protein(4, 1.0, 0)
    array.coord[:] = np.arange(array.array_length())[:, None] * 100
    array.coord[0] = 0
    array.coord[15] = [1.5, 0, 0]  # residue 4, exactly at the threshold
    metrics = get_clash_metrics(array, clash_threshold=1.5)
    assert metrics["n_clashing.interresidue_clashes_w_sidechain"] == 0
    array.coord[15] = [1.4999, 0, 0]
    metrics = get_clash_metrics(array, clash_threshold=1.5)
    assert metrics["n_clashing.interresidue_clashes_w_sidechain"] == 1


def test_cached_masses_match_biotite_gyration_radius():
    array = _protein(20, 30.0, 0)
    array.element[::7] = "V"  # unresolved virtual atoms keep biotite's lookup
    assert gyration_radius(array, masses=atom_masses(array)) == gyration_radius(array)
