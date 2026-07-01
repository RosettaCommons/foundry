"""Integration tests for the three fundamental ``rf3 fold`` input modes.

All tests share the ``basic_folds_dir`` session fixture, which runs a single
``rf3 fold`` call with all three inputs batched together — amortising the
model-loading cost.

Input files used (all in ``models/rf3/tests/data/``):

    agag_from_json.json            Protein-only JSON (AGAG, 4 residues)
                                   → output name ``agag_from_json``
    agag_with_ligands.json         AGAG + MG (ccd_code) + HEM (sdf path)
                                   + imidazole (smiles)
                                   → output name ``agag_with_ligands``
    agag_with_ligands_from_cif.cif CIF with AGAG + the same three ligands
                                   → output name ``agag_with_ligands_from_cif``

AGAG is a minimal 4-residue peptide used for fast CPU testing.  The
ligand inputs pair that peptide with three ligands supplied via three
distinct input modes — CCD code (MG), SDF file path (HEM) and SMILES
(imidazole) — so a single fold exercises every ligand-loading path.
"""

import pytest
from conftest import assert_standard_outputs, load_summary


@pytest.mark.integration
def test_fold_from_json_protein_only(basic_folds_dir):
    """Protein-only JSON input produces all expected output files."""
    assert_standard_outputs(basic_folds_dir, "agag_from_json")

    summary = load_summary(basic_folds_dir, "agag_from_json")
    assert 0 < summary["overall_plddt"] < 1
    assert not summary["has_clash"]


@pytest.mark.integration
def test_fold_from_json_with_ligand(basic_folds_dir):
    """JSON input with three ligands (ccd_code, sdf path, smiles) folds cleanly.

    The input pairs AGAG (chain A) with three ligands, each on its own chain:
    MG (chain B, via ``ccd_code``), HEM (chain C, via an SDF ``path``) and
    imidazole (chain D, via ``smiles``).  Ligands loaded from an SDF file or a
    SMILES string are assigned generated residue names (``L:0``, ``L:1``)
    rather than CCD codes, so presence is verified by the per-chain metric
    count plus the CCD-code ligand (MG) that keeps its name.
    """
    assert_standard_outputs(basic_folds_dir, "agag_with_ligands")

    summary = load_summary(basic_folds_dir, "agag_with_ligands")
    assert 0 < summary["overall_plddt"] < 1
    assert not summary["has_clash"]
    assert len(summary["chain_ptm"]) == 4, (
        "expected 4 chains (AGAG + MG + HEM + imidazole); "
        f"got {len(summary['chain_ptm'])}"
    )

    model_cif = (
        basic_folds_dir / "agag_with_ligands" / "agag_with_ligands_model.cif"
    ).read_text()
    assert "MG" in model_cif, "MG ligand missing from predicted structure"


@pytest.mark.integration
def test_fold_from_cif_with_ligand(basic_folds_dir):
    """CIF file input (containing protein + the same three ligands) folds cleanly.

    The CIF was generated from ``agag_with_ligands.json`` and carries the same
    four chains, so the round-trip through the CIF parser must preserve them.
    """
    assert_standard_outputs(basic_folds_dir, "agag_with_ligands_from_cif")

    summary = load_summary(basic_folds_dir, "agag_with_ligands_from_cif")
    assert 0 < summary["overall_plddt"] < 1
    assert not summary["has_clash"]
    assert len(summary["chain_ptm"]) == 4, (
        "expected 4 chains (AGAG + MG + HEM + imidazole); "
        f"got {len(summary['chain_ptm'])}"
    )

    model_cif = (
        basic_folds_dir
        / "agag_with_ligands_from_cif"
        / "agag_with_ligands_from_cif_model.cif"
    ).read_text()
    assert "MG" in model_cif, "MG ligand missing from predicted structure"
