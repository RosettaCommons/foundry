"""Integration tests for the three fundamental ``rf3 fold`` input modes.

All tests share the ``basic_folds_dir`` session fixture, which runs a single
``rf3 fold`` call with all three inputs batched together — amortising the
model-loading cost.

Input files used (all in ``models/rf3/tests/data/``):

    1cyo_from_json.json   Protein-only JSON  → output name ``1cyo_from_json``
    1cyo_with_ligand.json Protein + HEM JSON → output name ``1cyo_with_ligand``
    1cyo.cif              CIF with protein + HEM → output name ``1cyo``

1CYO is Cytochrome B5 (91-residue protein) with a heme (HEM) ligand.
No MSA is provided; the protein is short enough to fold without one.
"""

import pytest
from conftest import assert_standard_outputs, load_summary


@pytest.mark.integration
def test_fold_from_json_protein_only(basic_folds_dir):
    """Protein-only JSON input produces all expected output files."""
    assert_standard_outputs(basic_folds_dir, "1cyo_from_json")

    summary = load_summary(basic_folds_dir, "1cyo_from_json")
    assert 0 < summary["overall_plddt"] < 1
    assert not summary["has_clash"]


@pytest.mark.integration
def test_fold_from_json_with_ligand(basic_folds_dir):
    """JSON input with a CCD-code ligand produces a structure containing HEM."""
    assert_standard_outputs(basic_folds_dir, "1cyo_with_ligand")

    summary = load_summary(basic_folds_dir, "1cyo_with_ligand")
    assert 0 < summary["overall_plddt"] < 1
    assert not summary["has_clash"]

    model_cif = basic_folds_dir / "1cyo_with_ligand" / "1cyo_with_ligand_model.cif"
    assert "HEM" in model_cif.read_text(), "HEM ligand missing from predicted structure"


@pytest.mark.integration
def test_fold_from_cif_with_ligand(basic_folds_dir):
    """CIF file input (containing protein + HEM) produces a structure with both."""
    assert_standard_outputs(basic_folds_dir, "1cyo")

    summary = load_summary(basic_folds_dir, "1cyo")
    assert 0 < summary["overall_plddt"] < 1
    assert not summary["has_clash"]

    model_cif = basic_folds_dir / "1cyo" / "1cyo_model.cif"
    assert "HEM" in model_cif.read_text(), "HEM ligand missing from predicted structure"
