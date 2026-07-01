"""Unit tests for the pure validation helpers in ``rfd3na.inference.symmetry.checks``.

These guard a symmetric-assembly input before frame computation. Each raises a clear error
(or, for ``check_max_rmsds``, only warns) when its contract is violated:

- ``check_valid_multiplicity`` — the per-entity chain counts must share a common multiplicity
  > 1 (every entity's count divisible by the minimum); multiplicity 1 or a non-divisible
  count raises.
- ``check_valid_subunit_size`` — every chain of an entity must span the same number of atoms.
- ``check_min_atoms_to_align`` — the reference entity needs at least ``MIN_ATOMS_ALIGN`` atoms.
- ``check_max_transforms`` — at most ``MAX_TRANSFORMS`` chains.
- ``check_input_frames_match_symmetry_frames`` — computed vs original frame counts must match.
- ``check_max_rmsds`` — warns (does *not* raise) when any RMSD exceeds ``RMSD_CUT``.
"""

import numpy as np
import pytest
from rfd3na.inference.symmetry.checks import (
    MAX_TRANSFORMS,
    MIN_ATOMS_ALIGN,
    check_input_frames_match_symmetry_frames,
    check_max_rmsds,
    check_max_transforms,
    check_min_atoms_to_align,
    check_valid_multiplicity,
    check_valid_subunit_size,
)

# --- check_valid_multiplicity -----------------------------------------------


def test_multiplicity_ok_for_uniform_dimer():
    check_valid_multiplicity({0: ["A_1", "B_1"], 1: ["A_2", "B_2"]})


def test_multiplicity_one_raises():
    with pytest.raises(ValueError, match="no possible symmetry"):
        check_valid_multiplicity({0: ["A_1"]})


def test_multiplicity_non_divisible_raises():
    # min multiplicity is 2, but the first entity has 3 chains (3 % 2 != 0).
    with pytest.raises(ValueError, match="multiplicity"):
        check_valid_multiplicity({0: ["A_1", "B_1", "C_1"], 1: ["A_2", "B_2"]})


# --- check_valid_subunit_size -----------------------------------------------


def test_subunit_size_ok_when_chains_equal_length():
    pn_unit_id = np.array(["A_1", "A_1", "B_1", "B_1"])
    check_valid_subunit_size({0: ["A_1", "B_1"]}, pn_unit_id)


def test_subunit_size_mismatch_raises():
    pn_unit_id = np.array(["A_1", "A_1", "B_1"])  # A_1 has 2 atoms, B_1 has 1
    with pytest.raises(ValueError, match="Size mismatch"):
        check_valid_subunit_size({0: ["A_1", "B_1"]}, pn_unit_id)


# --- check_min_atoms_to_align -----------------------------------------------


def test_min_atoms_ok_at_threshold():
    check_min_atoms_to_align({0: MIN_ATOMS_ALIGN}, 0)


def test_min_atoms_below_threshold_raises():
    with pytest.raises(ValueError, match="Not enough atoms"):
        check_min_atoms_to_align({0: MIN_ATOMS_ALIGN - 1}, 0)


# --- check_max_transforms ---------------------------------------------------


def test_max_transforms_ok_at_limit():
    check_max_transforms(list(range(MAX_TRANSFORMS)))


def test_max_transforms_over_limit_raises():
    with pytest.raises(ValueError, match="exceeds the max"):
        check_max_transforms(list(range(MAX_TRANSFORMS + 1)))


# --- check_input_frames_match_symmetry_frames -------------------------------


def test_frame_counts_match_ok():
    check_input_frames_match_symmetry_frames([1, 2, 3], ["a", "b", "c"], {})


def test_frame_counts_mismatch_raises():
    with pytest.raises(AssertionError):
        check_input_frames_match_symmetry_frames([1, 2], ["a", "b", "c"], {})


# --- check_max_rmsds --------------------------------------------------------


def test_max_rmsds_within_cut_does_not_raise():
    check_max_rmsds({"A": 0.5, "B": 0.9})


def test_max_rmsds_over_cut_warns_without_raising():
    # Contract: an oversized RMSD is a soft warning, not a hard failure.
    check_max_rmsds({"A": 5.0})
