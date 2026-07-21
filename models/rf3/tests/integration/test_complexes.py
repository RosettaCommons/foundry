"""Integration tests for multi-entity complexes and multi-input batching.

These cover input compositions beyond the single-chain / single-ligand basics
in ``test_basic_fold.py``:

    two_protein_chains   two protein chains, so interface metrics (iptm,
                         cross-chain PAE) are actually computed
    protein_dna_complex  a protein + DNA complex (nucleic-acid support)
    peptide_glycan_bond  an explicit covalent bond via the JSON ``bonds`` field
    two_examples_*       two examples defined in a single JSON file
    dir_pep_*            a directory of inputs folded in one call

The multi-chain cases use short, unambiguous protein sequences (e.g. ``GLKE``):
sequences over only the ``A/C/G/T/U`` alphabet (such as ``AGAG``) are inferred
as nucleic acid rather than protein, so they cannot stand in for a peptide.
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
def test_fold_two_protein_chains(complex_folds_dir):
    """Two protein chains fold and produce genuine interface metrics."""
    name = "two_protein_chains"
    assert_standard_outputs(complex_folds_dir, name)

    summary = load_summary(complex_folds_dir, name)
    assert_valid_plddt(summary)
    # has_clash is intentionally not asserted: with two polymer chains it becomes
    # a real (non-vacuous) clash check, but its value depends on the batched RNG
    # stream and so is order-sensitive (see complex_folds_dir docstring).

    # Two chains → the model scores an interface.  Single-chain inputs instead
    # report iptm=0.0 with a 1x1 chain-pair matrix (known issue #1), so the
    # checks below distinguish a real two-chain fold from that degenerate case.
    assert_chain_count(summary, 2)
    chain_pair_pae = summary["chain_pair_pae"]
    assert (
        len(chain_pair_pae) == 2 and len(chain_pair_pae[0]) == 2
    ), f"expected a 2x2 chain-pair PAE matrix; got {chain_pair_pae}"
    assert (
        chain_pair_pae[0][1] is not None
    ), "cross-chain PAE was not scored for a two-chain input"
    assert summary["iptm"] > 0


@pytest.mark.integration
def test_fold_protein_dna_complex(complex_folds_dir):
    """A protein + DNA complex folds with the DNA nucleotides preserved."""
    name = "protein_dna_complex"
    assert_standard_outputs(complex_folds_dir, name)

    summary = load_summary(complex_folds_dir, name)
    assert_valid_plddt(summary)
    # has_clash not asserted: protein + DNA are both polymers, so the check is
    # non-vacuous but batch-order-sensitive (see complex_folds_dir docstring).
    assert_chain_count(summary, 2)

    resnames = residue_names_in_cif(complex_folds_dir / name / f"{name}_model.cif")
    assert (
        {"DA", "DT", "DG", "DC"} <= resnames
    ), f"expected all four DNA nucleotides in the output; got {sorted(resnames)}"


@pytest.mark.integration
def test_fold_with_covalent_bond(complex_folds_dir):
    """A peptide + NAG joined by an explicit JSON ``bonds`` entry folds cleanly."""
    name = "peptide_glycan_bond"
    assert_standard_outputs(complex_folds_dir, name)

    summary = load_summary(complex_folds_dir, name)
    assert_valid_plddt(summary)
    # has_clash is intentionally not asserted here: forcing a covalent bond onto
    # a tiny 4-residue peptide under aggressive speed flags yields a strained,
    # low-quality structure that may clash.  This test verifies that the JSON
    # `bonds` API is accepted and folds with the glycan retained, not fold quality.

    resnames = residue_names_in_cif(complex_folds_dir / name / f"{name}_model.cif")
    assert "NAG" in resnames, "NAG glycan missing from predicted structure"


@pytest.mark.integration
@pytest.mark.parametrize("name", ["two_examples_first", "two_examples_second"])
def test_fold_multiple_examples_in_one_json(complex_folds_dir, name):
    """Both examples defined in one JSON file produce their own outputs."""
    assert_standard_outputs(complex_folds_dir, name)
    summary = load_summary(complex_folds_dir, name)
    assert_valid_plddt(summary)


@pytest.mark.integration
@pytest.mark.parametrize("name", ["dir_pep_a", "dir_pep_b"])
def test_fold_from_directory(directory_input_dir, name):
    """Passing a directory folds every compatible input file inside it."""
    assert_standard_outputs(directory_input_dir, name)
    summary = load_summary(directory_input_dir, name)
    assert_valid_plddt(summary)
