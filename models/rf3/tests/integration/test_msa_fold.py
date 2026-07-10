"""End-to-end ``rf3 fold`` tests for pre-computed MSA inputs.

These are the checkpoint-gated "Layer 2" companion to the checkpoint-free
format-contract tests in ``tests/test_msa_format.py``. They feed real MSA files
through the CLI to verify that rf3 accepts the documented input format and that
supplying an MSA produces sensible confidence outputs across chain counts and
both input modes:

    monomer    monomer_msa.json          — JSON ``msa_path`` (1 chain)
    monomer    monomer_msa_from_cif.cif  — CIF ``_msa_paths_by_chain_id`` header
    homodimer  homodimer_msa.json        — JSON ``msa_path`` (2 chains, shared a3m)
    heteromer  heteromer_paired_msa.json — two distinct chains, each own a3m

All inputs use short synthetic sequences with hand-built a3m files (see the
``msa_fold_dir`` fixture): these tests exercise MSA format and plumbing, not
structure quality, so a real complex — which is far slower on CPU — is
unnecessary.

The multi-chain cases also assert ``iptm > 0``: interface scoring is only
meaningful with more than one chain (single-chain inputs report the ``iptm=0.0``
sentinel of known issue #1), so these are the cases where an MSA-fed interface
metric is a real, non-degenerate value.

A final pair of tests pins the ``raise_if_missing_msa_for_protein_of_length_n``
guard: it must error when a long protein chain has no MSA, and stay quiet when
every chain does.
"""

import json

import pytest
from conftest import (
    assert_standard_outputs,
    assert_valid_plddt,
    load_summary,
    run_rf3_fold,
)


@pytest.mark.integration
def test_fold_monomer_with_msa_json(msa_fold_dir):
    """A single-chain JSON input with a per-component ``msa_path`` folds cleanly."""
    name = "monomer_msa"
    assert_standard_outputs(msa_fold_dir, name)

    summary = load_summary(msa_fold_dir, name)
    assert_valid_plddt(summary)
    assert len(summary["chain_ptm"]) == 1, "expected a single-chain monomer"


@pytest.mark.integration
def test_fold_monomer_with_msa_cif(msa_fold_dir):
    """A single-chain CIF declaring its MSA via ``_msa_paths_by_chain_id`` folds.

    Same monomer sequence as ``test_fold_monomer_with_msa_json`` but the MSA is
    attached through the CIF header rather than a JSON ``msa_path``, so this
    exercises the CIF-side MSA-path parsing independently.
    """
    name = "monomer_msa_from_cif"
    assert_standard_outputs(msa_fold_dir, name)

    summary = load_summary(msa_fold_dir, name)
    assert_valid_plddt(summary)
    assert len(summary["chain_ptm"]) == 1, "expected a single-chain monomer"


@pytest.mark.integration
def test_fold_homodimer_shared_msa_json(msa_fold_dir):
    """A homodimer whose two chains share one JSON ``msa_path`` scores an interface."""
    name = "homodimer_msa"
    assert_standard_outputs(msa_fold_dir, name)

    summary = load_summary(msa_fold_dir, name)
    assert_valid_plddt(summary)
    assert len(summary["chain_ptm"]) == 2, "expected two (identical) chains"
    # Two chains → a real interface, so iptm is a genuine computed value rather
    # than the single-chain iptm=0.0 sentinel (known issue #1).
    assert summary["iptm"] > 0


@pytest.mark.integration
def test_fold_heteromer_paired_msa(msa_fold_dir):
    """Two distinct chains, each with its own a3m, fold with a scored interface.

    ``heteromer_paired_msa.json`` gives chain A and chain B different sequences
    and different ``msa_path`` files (``msas/heteromer_A.a3m`` /
    ``heteromer_B.a3m``). This is the case that documents per-chain MSA wiring
    for a true heteromer; a shared ``TaxID`` across the two a3m files makes it a
    genuine paired MSA.
    """
    name = "heteromer_paired_msa"
    assert_standard_outputs(msa_fold_dir, name)

    summary = load_summary(msa_fold_dir, name)
    assert_valid_plddt(summary)
    assert len(summary["chain_ptm"]) == 2, "expected two distinct chains"
    assert summary["iptm"] > 0


@pytest.mark.integration
def test_raise_if_missing_msa_errors_when_absent(require_ckpt, tmp_path):
    """The guard errors when a protein above the length threshold has no MSA.

    A 12-residue protein with no ``msa_path`` and
    ``raise_if_missing_msa_for_protein_of_length_n=10`` must fail: MSA loading
    raises before diffusion, so ``run_rf3_fold`` surfaces a non-zero exit.
    """
    no_msa = tmp_path / "no_msa_pep.json"
    no_msa.write_text(
        json.dumps(
            [
                {
                    "name": "no_msa_pep",
                    "components": [{"seq": "GLKEIWQYVRND", "chain_id": "A"}],
                }
            ]
        )
    )
    with pytest.raises(RuntimeError, match="rf3 fold failed"):
        run_rf3_fold(
            no_msa,
            tmp_path / "out",
            extra_flags=["raise_if_missing_msa_for_protein_of_length_n=10"],
        )


@pytest.mark.integration
def test_raise_if_missing_msa_succeeds_when_present(msa_present_flag_dir):
    """With MSAs supplied for every chain, the missing-MSA guard does not trip."""
    name = "monomer_msa"
    assert_standard_outputs(msa_present_flag_dir, name)
    assert_valid_plddt(load_summary(msa_present_flag_dir, name))
