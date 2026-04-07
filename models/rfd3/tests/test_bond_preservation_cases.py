#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../../../scripts/shebang/modelhub_exec.sh" "$0" "$@"'
"""
Bond preservation regression tests for representative connection types.
"""

from pathlib import Path

import numpy as np
import pytest
from atomworks.io.parser import STANDARD_PARSER_ARGS, parse
from atomworks.io.tools.inference import components_to_atom_array
from biotite import structure as struc
from rfd3.inference.input_parsing import (
    accumulate_components,
    create_atom_array_from_design_specification,
)
from rfd3.transforms.conditioning_base import (
    set_default_conditioning_annotations,
)
from rfd3.utils.inference import set_common_annotations

from foundry.utils.components import fetch_mask_from_idx

TEST_DATA_DIR = Path(__file__).parent / "test_data"


def _load_atom_array(pdb_id: str):
    path = TEST_DATA_DIR / f"{pdb_id.lower()}.cif"
    if not path.exists():
        pytest.skip(f"Test data file missing: {path}")
    parser_args = {
        **STANDARD_PARSER_ARGS,
        # Ignore metal coordination; only covalent/disulfide bonds are restored.
        "add_bond_types_from_struct_conn": ["covale", "disulf"],
    }
    result = parse(filename=path, build_assembly=("1",), **parser_args)
    return result["assemblies"]["1"][0]


def _prepare_token(atom_array, component: str):
    mask = fetch_mask_from_idx(component, atom_array=atom_array)
    token = atom_array[mask].copy()
    token = set_default_conditioning_annotations(token, motif=True, dtype=int)
    token = set_common_annotations(token)
    token.res_id = np.ones(token.shape[0], dtype=token.res_id.dtype)
    return token


def _accumulate(atom_array, components):
    tokens = {c: _prepare_token(atom_array, c) for c in components}
    return accumulate_components(
        components_to_accumulate=components,
        indexed_tokens=tokens,
        unindexed_tokens={},
        atom_array_accum=[],
        start_chain="A",
        start_resid=1,
        unindexed_breaks=[None] * len(components),
        src_atom_array=atom_array,
    )


def _create_ptm_atom_array():
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
    atom_array.coord = np.random.randn(len(atom_array), 3).astype(np.float32) * 10
    return atom_array


def _atom_index(arr, res_id: int, atom_name: str, chain: str = "A") -> int:
    mask = (
        (arr.chain_id == chain) & (arr.res_id == res_id) & (arr.atom_name == atom_name)
    )
    idx = np.where(mask)[0]
    assert len(idx) == 1, f"Atom {chain}{res_id}:{atom_name} not unique/found"
    return int(idx[0])


def _bond_exists(
    arr: struc.AtomArray,
    idx_a: int,
    idx_b: int,
    bond_type: struc.BondType | None = None,
) -> bool:
    bonds = arr.bonds.as_array()
    mask = ((bonds[:, 0] == idx_a) & (bonds[:, 1] == idx_b)) | (
        (bonds[:, 0] == idx_b) & (bonds[:, 1] == idx_a)
    )
    if bond_type is not None:
        mask &= bonds[:, 2] == bond_type
    return np.any(mask)


def _bond_label(arr: struc.AtomArray, idx: int) -> tuple[str, int, str, str]:
    """Return a human-friendly label for assertions."""
    return (
        arr.chain_id[idx],
        int(arr.res_id[idx]),
        arr.res_name[idx],
        arr.atom_name[idx],
    )


def _cross_residue_bonds(
    arr: struc.AtomArray,
) -> set[tuple[tuple[str, int, str, str], tuple[str, int, str, str]]]:
    bonds = set()
    for a, b, _ in arr.bonds.as_array():
        a = int(a)
        b = int(b)
        if arr.chain_id[a] == arr.chain_id[b] and arr.res_id[a] == arr.res_id[b]:
            continue
        bond = tuple(sorted((_bond_label(arr, a), _bond_label(arr, b))))
        bonds.add(bond)
    return bonds


@pytest.mark.slow
def test_disulfide_preserved():
    """
    1crn.cif struct_conn disulf1: A CYS 3 SG <-> A CYS 40 SG.
    """
    arr = _load_atom_array("1crn")
    accum = accumulate_components(
        components_to_accumulate=[1, "A3", "A40", 1],
        indexed_tokens={
            "A3": _prepare_token(arr, "A3"),
            "A40": _prepare_token(arr, "A40"),
        },
        unindexed_tokens={},
        atom_array_accum=[],
        start_chain="A",
        start_resid=1,
        unindexed_breaks=[None] * 4,
        src_atom_array=arr,
    )
    sg1 = _atom_index(accum, 2, "SG")  # CYS3 after one diffused residue
    sg2 = _atom_index(accum, 3, "SG")  # CYS40 after one diffused + one indexed
    assert _bond_exists(accum, sg1, sg2)
    expected_bonds = {tuple(sorted((_bond_label(accum, sg1), _bond_label(accum, sg2))))}
    assert _cross_residue_bonds(accum) == expected_bonds


@pytest.mark.slow
def test_covalent_ligand_preserved():
    """
    4qdv.cif struct_conn covale1: A TYR 143 OH <-> E 30U 401 S1.
    """
    arr = _load_atom_array("4qdv")
    accum = _accumulate(arr, ["A143", "E401"])
    oh_tyr = _atom_index(accum, 1, "OH")
    s1_30u = _atom_index(accum, 2, "S1")
    expected_bonds = {
        tuple(sorted((_bond_label(accum, oh_tyr), _bond_label(accum, s1_30u))))
    }
    assert _cross_residue_bonds(accum) == expected_bonds


@pytest.mark.slow
def test_dna_af_adduct_preserved():
    """
    1ua0.cif AF adduct: label chains B/E (auth C/C) DG4 C8 <-> AF333 N.
    """
    arr = _load_atom_array("1ua0")
    accum = _accumulate(arr, ["B4", "E333"])
    c8_dg = _atom_index(accum, 1, "C8")
    n_af = _atom_index(accum, 2, "N")
    assert _bond_exists(accum, c8_dg, n_af)
    expected_bonds = {
        tuple(sorted((_bond_label(accum, c8_dg), _bond_label(accum, n_af))))
    }
    assert _cross_residue_bonds(accum) == expected_bonds


@pytest.mark.slow
def test_cyclic_thioether_link_preserved():
    """
    6u6k.cif struct_conn:
    - covale1: B ACE 1 C   <-> B TRP 2 N
    - covale2: B ACE 1 CH3 <-> B CYS 12 SG
    """
    arr = _load_atom_array("6u6k")
    accum = _accumulate(arr, ["B1", "B2", "B12"])
    ch3 = _atom_index(accum, 1, "CH3")
    c_ace = _atom_index(accum, 1, "C")
    n_trp = _atom_index(accum, 2, "N")
    sg = _atom_index(accum, 3, "SG")
    assert _bond_exists(accum, ch3, sg)
    assert _bond_exists(accum, c_ace, n_trp)
    expected_bonds = {
        tuple(sorted((_bond_label(accum, ch3), _bond_label(accum, sg)))),
        tuple(sorted((_bond_label(accum, c_ace), _bond_label(accum, n_trp)))),
    }
    assert _cross_residue_bonds(accum) == expected_bonds


@pytest.mark.slow
def test_nonpeptide_noncanonical_not_backbone_linked():
    """
    NIO in 3o14.cif has an atom named N but no struct_conn.
    with only diffused neighbors no backbone bond should be synthesized.
    """
    arr = _load_atom_array("3o14")
    accum = accumulate_components(
        components_to_accumulate=[1, "D300", 1],
        indexed_tokens={"D300": _prepare_token(arr, "D300")},
        unindexed_tokens={},
        atom_array_accum=[],
        start_chain="A",
        start_resid=1,
        unindexed_breaks=[None] * 3,
        src_atom_array=arr,
    )
    # No cross-residue bonds because nothing is connected to the ligand.
    assert _cross_residue_bonds(accum) == set()


@pytest.mark.slow
def test_glycan_links_and_absence_when_partner_missing():
    """
    8f7t.cif struct_conn covale5: C ASN 403 ND2 <-> G NAG 1 C1.
    Also ensure NAG1 has no cross-res bonds if ASN403 is not included.
    """
    arr = _load_atom_array("8f7t")
    accum = _accumulate(arr, ["C403", "G1"])
    nd2 = _atom_index(accum, 1, "ND2")
    c1 = _atom_index(accum, 2, "C1")
    assert _bond_exists(accum, nd2, c1)
    expected_bonds = {tuple(sorted((_bond_label(accum, nd2), _bond_label(accum, c1))))}
    assert _cross_residue_bonds(accum) == expected_bonds

    accum_no_asn = accumulate_components(
        components_to_accumulate=[1, "G1", 1],
        indexed_tokens={"G1": _prepare_token(arr, "G1")},
        unindexed_tokens={},
        atom_array_accum=[],
        start_chain="A",
        start_resid=1,
        unindexed_breaks=[None] * 3,
        src_atom_array=arr,
    )
    c1_lonely = _atom_index(accum_no_asn, 2, "C1")
    assert _cross_residue_bonds(accum_no_asn) == set()
    partners = [
        int(bond[1]) if bond[0] == c1_lonely else int(bond[0])
        for bond in accum_no_asn.bonds.as_array()
        if c1_lonely in bond[:2]
    ]
    partner_res_ids = accum_no_asn.res_id[partners] if partners else []
    assert len(partner_res_ids) == 0 or np.all(partner_res_ids == 2)


@pytest.mark.slow
def test_backbone_struct_conn_preserved_with_diffusion():
    """
    1p5d.cif struct_conn covale1/2:
    GLY107 C <-> SEP108 N, SEP108 C <-> HIS109 N.
    """
    arr = _load_atom_array("1p5d")
    accum = accumulate_components(
        components_to_accumulate=[2, "A108", 2],
        indexed_tokens={"A108": _prepare_token(arr, "A108")},
        unindexed_tokens={},
        atom_array_accum=[],
        start_chain="A",
        start_resid=1,
        unindexed_breaks=[None] * 3,
        src_atom_array=arr,
    )
    c_prev = _atom_index(accum, 2, "C")
    n_sep = _atom_index(accum, 3, "N")
    c_sep = _atom_index(accum, 3, "C")
    n_next = _atom_index(accum, 4, "N")
    assert _bond_exists(accum, c_prev, n_sep)
    assert _bond_exists(accum, c_sep, n_next)
    expected_bonds = {
        tuple(sorted((_bond_label(accum, c_prev), _bond_label(accum, n_sep)))),
        tuple(sorted((_bond_label(accum, c_sep), _bond_label(accum, n_next)))),
    }
    assert _cross_residue_bonds(accum) == expected_bonds


@pytest.mark.fast
def test_ptm_backbone_bonds_preserved_with_diffusion():
    """
    Synthetic PTR/SEP motif with 5 diffused residues on each side; ensure backbone
    bonds span the diffused neighbors.
    """
    src_atom_array = _create_ptm_atom_array()
    indexed_tokens = {c: _prepare_token(src_atom_array, c) for c in ["A3", "A4"]}

    accum = accumulate_components(
        components_to_accumulate=[5, "A3", "A4", 5],
        indexed_tokens=indexed_tokens,
        unindexed_tokens={},
        atom_array_accum=[],
        start_chain="A",
        start_resid=1,
        unindexed_breaks=[None] * 4,
        src_atom_array=src_atom_array,
    )

    diffused_c = _atom_index(accum, 5, "C")
    ptr_n = _atom_index(accum, 6, "N")
    assert _bond_exists(accum, diffused_c, ptr_n, struc.BondType.SINGLE)

    ptr_c = _atom_index(accum, 6, "C")
    sep_n = _atom_index(accum, 7, "N")
    assert _bond_exists(accum, ptr_c, sep_n, struc.BondType.SINGLE)

    sep_c = _atom_index(accum, 7, "C")
    diffused_after_n = _atom_index(accum, 8, "N")
    assert _bond_exists(accum, sep_c, diffused_after_n, struc.BondType.SINGLE)

    cross_bonds = _cross_residue_bonds(accum)
    expected_cross = {
        tuple(sorted((_bond_label(accum, diffused_c), _bond_label(accum, ptr_n)))),
        tuple(sorted((_bond_label(accum, ptr_c), _bond_label(accum, sep_n)))),
        tuple(
            sorted((_bond_label(accum, sep_c), _bond_label(accum, diffused_after_n)))
        ),
    }
    assert expected_cross.issubset(cross_bonds)


@pytest.mark.fast
def test_ptm_backbone_bonds_preserved_full_pipeline():
    """
    End-to-end test through create_atom_array_from_design_specification (dialect 2).
    Ensures PTM backbone bonds survive the normal loading pipeline.
    """
    atom_array_input = _create_ptm_atom_array()

    contig = "5-5,A3-4,5-5"  # -> [5, A3, A4, 5]
    atom_array, _ = create_atom_array_from_design_specification(
        atom_array_input=atom_array_input,
        input=None,
        contig=contig,
        length="12-12",
        dialect=2,
    )

    diffused_c = _atom_index(atom_array, 5, "C")
    ptr_n = _atom_index(atom_array, 6, "N")
    ptr_c = _atom_index(atom_array, 6, "C")
    sep_n = _atom_index(atom_array, 7, "N")
    sep_c = _atom_index(atom_array, 7, "C")
    diffused_after_n = _atom_index(atom_array, 8, "N")

    assert _bond_exists(atom_array, diffused_c, ptr_n, struc.BondType.SINGLE)
    assert _bond_exists(atom_array, ptr_c, sep_n, struc.BondType.SINGLE)
    assert _bond_exists(atom_array, sep_c, diffused_after_n, struc.BondType.SINGLE)
