"""
Tests that legacy accumulate_components preserves backbone bonds involving PTMs.
"""

import numpy as np
import pytest
from atomworks.io.tools.inference import components_to_atom_array
from biotite import structure as struc

from foundry.utils.components import fetch_mask_from_idx
from rfd3.inference.legacy_input_parsing import accumulate_components
from rfd3.transforms.conditioning_base import set_default_conditioning_annotations
from rfd3.utils.inference import set_common_annotations


def _create_ptm_structure():
    """Create a simple protein with PTMs: AG(PTR)(SEP)SA."""
    components = [
        {
            "seq": "AG(PTR)(SEP)SA",
            "chain_type": "polypeptide(l)",
            "is_polymer": True,
            "chain_id": "A",
        },
    ]
    atom_array = components_to_atom_array(components)
    # Add coordinates so bond inference code that may rely on coords won't hit NaNs
    atom_array.coord = np.random.randn(len(atom_array), 3).astype(np.float32) * 10
    return atom_array


def _prepare_indexed_tokens(atom_array, components):
    """Create motif tokens with required annotations for accumulate_components."""
    tokens = {}
    for component in components:
        mask = fetch_mask_from_idx(component, atom_array=atom_array)
        token = atom_array[mask].copy()
        token = set_default_conditioning_annotations(
            token, motif=True, unindexed=False, dtype=int
        )
        token = set_common_annotations(token)
        token.res_id = np.ones(token.shape[0], dtype=token.res_id.dtype)
        tokens[component] = token
    return tokens


def _connection_exists(bonds, a, b, bond_type=None):
    """Check if a connection between atoms a and b exists (in either direction)."""
    mask = ((bonds[:, 0] == a) & (bonds[:, 1] == b)) | (
        (bonds[:, 0] == b) & (bonds[:, 1] == a)
    )
    if bond_type is not None:
        mask &= bonds[:, 2] == bond_type
    return np.any(mask)


@pytest.mark.fast
def test_legacy_ptm_backbone_bonds():
    """
    Verify that PTM backbone bonds are restored in legacy parser.

    Setup: 5 diffused residues -> PTR-SEP (connected motif) -> 5 diffused residues
    Expect backbone bonds: diffused->PTR, PTR->SEP, SEP->diffused.
    """

    src_atom_array = _create_ptm_structure()
    components = [5, "A3", "A4", 5]

    result = accumulate_components(
        components=components,
        src_atom_array=src_atom_array,
        redesign_motif_sidechains=False,
        unindexed_components=[],
        unfixed_sequence_components=[],
        breaks=[None] * len(components),
        fixed_atoms={},
        unfix_all=False,
        optional_conditions=[],
        flexible_backbone=False,
        unfix_residues=[],
    )
    bonds = result.bonds.as_array()

    def atom_idx(chain, resid, atom_name):
        mask = (
            (result.chain_id == chain)
            & (result.res_id == resid)
            & (result.atom_name == atom_name)
        )
        idx = np.where(mask)[0]
        assert (
            len(idx) == 1
        ), f"Expected unique atom for {chain}{resid}:{atom_name}, got {len(idx)}"
        return idx[0]

    # Residue IDs after accumulation: 1-5 diffused, 6=A3 (PTR), 7=A4 (SEP), 8+=diffused
    diffused_c = atom_idx("A", 5, "C")
    ptr_n = atom_idx("A", 6, "N")
    assert _connection_exists(bonds, diffused_c, ptr_n, struc.BondType.SINGLE)

    ptr_c = atom_idx("A", 6, "C")
    sep_n = atom_idx("A", 7, "N")
    assert _connection_exists(bonds, ptr_c, sep_n, struc.BondType.SINGLE)

    sep_c = atom_idx("A", 7, "C")
    diffused_after_n = atom_idx("A", 8, "N")
    assert _connection_exists(bonds, sep_c, diffused_after_n, struc.BondType.SINGLE)
