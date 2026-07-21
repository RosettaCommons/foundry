"""Integration tests for the three fundamental ``rf3 fold`` input modes.

All tests share the ``basic_folds_dir`` session fixture, which runs a single
``rf3 fold`` call with all three inputs batched together — amortising the
model-loading cost.

Input files used (all in ``models/rf3/tests/data/``):

    glke_from_json.json            Protein-only JSON (GLKE, 4 residues)
                                   → output name ``glke_from_json``
    glke_with_ligands.json         GLKE + MG (ccd_code) + HEM (sdf path)
                                   + imidazole (smiles)
                                   → output name ``glke_with_ligands``
    glke_with_ligands_from_cif.cif CIF with GLKE + the same three ligands
                                   → output name ``glke_with_ligands_from_cif``

GLKE is a minimal 4-residue peptide used for fast CPU testing.  The
ligand inputs pair that peptide with three ligands supplied via three
distinct input modes — CCD code (MG), SDF file path (HEM) and SMILES
(imidazole) — so a single fold exercises every ligand-loading path.
"""

import pytest
from conftest import (
    assert_chain_count,
    assert_standard_outputs,
    assert_valid_plddt,
    load_summary,
    residue_names_in_cif,
)


@pytest.mark.integration
def test_fold_from_json_protein_only(basic_folds_dir):
    """Protein-only JSON input produces all expected output files."""
    assert_standard_outputs(basic_folds_dir, "glke_from_json")

    summary = load_summary(basic_folds_dir, "glke_from_json")
    assert_valid_plddt(summary)
    # Single polymer chain → no polymer-polymer pair, so has_clash is always
    # False here (see clashing_chains.py); this only checks the field is present.
    assert not summary["has_clash"]


@pytest.mark.integration
def test_fold_from_json_with_ligand(basic_folds_dir):
    """JSON input with three ligands (ccd_code, sdf path, smiles) folds cleanly.

    The input pairs GLKE (chain A) with three ligands, each on its own chain:
    MG (chain B, via ``ccd_code``), HEM (chain C, via an SDF ``path``) and
    imidazole (chain D, via ``smiles``).  Ligands loaded from an SDF file or a
    SMILES string are assigned generated residue names (``L:0``, ``L:1``)
    rather than CCD codes, so presence is verified by the per-chain metric
    count plus the CCD-code ligand (MG) that keeps its name.
    """
    assert_standard_outputs(basic_folds_dir, "glke_with_ligands")

    summary = load_summary(basic_folds_dir, "glke_with_ligands")
    assert_valid_plddt(summary)
    assert not summary["has_clash"]
    assert_chain_count(summary, 4, "GLKE + MG + HEM + imidazole")

    model_cif = basic_folds_dir / "glke_with_ligands" / "glke_with_ligands_model.cif"
    assert "MG" in residue_names_in_cif(
        model_cif
    ), "MG ligand missing from predicted structure"


@pytest.mark.integration
def test_fold_from_cif_with_ligand(basic_folds_dir):
    """CIF file input (containing protein + the same three ligands) folds cleanly.

    The CIF was generated from ``glke_with_ligands.json`` and carries the same
    four chains, so the round-trip through the CIF parser must preserve them.
    """
    assert_standard_outputs(basic_folds_dir, "glke_with_ligands_from_cif")

    summary = load_summary(basic_folds_dir, "glke_with_ligands_from_cif")
    assert_valid_plddt(summary)
    assert not summary["has_clash"]
    assert_chain_count(summary, 4, "GLKE + MG + HEM + imidazole")

    model_cif = (
        basic_folds_dir
        / "glke_with_ligands_from_cif"
        / "glke_with_ligands_from_cif_model.cif"
    )
    assert "MG" in residue_names_in_cif(
        model_cif
    ), "MG ligand missing from predicted structure"
