import logging

import numpy as np
from atomworks.ml.utils.token import (
    get_token_starts,
)
from biotite.structure import AtomArray
from rfd3na.trainer.trainer_utils import (
    _cleanup_virtual_atoms_and_assign_atom_name_elements,
    _readout_seq_from_struc,
)
from rfd3na.transforms.na_geom_utils import annotate_na_ss

from foundry.metrics.metric import Metric
from foundry.utils.ddp import RankedLogger

logging.basicConfig(level=logging.INFO)
global_logger = RankedLogger(__name__, rank_zero_only=False)


def _get_bp_partners_annotation(atom_array: AtomArray):
    """Return bp-partners annotation."""
    categories = atom_array.get_annotation_categories()
    if "bp_partners" in categories:
        return atom_array.bp_partners
    raise ValueError("atom_array missing bp_partners annotation")


def _safe_f1_from_sizes(intersection_n: int, pred_n: int, gt_n: int) -> float:
    """Return F1 with sensible empty-set handling."""
    if pred_n == 0 and gt_n == 0:
        return 1.0

    precision = float(intersection_n / pred_n) if pred_n > 0 else 0.0
    recall = float(intersection_n / gt_n) if gt_n > 0 else 0.0

    if precision + recall == 0.0:
        return 0.0

    return float(2.0 * precision * recall / (precision + recall))


def _get_token_ids(atom_array: AtomArray) -> np.ndarray:
    token_starts = get_token_starts(atom_array)
    token_level_array = atom_array[token_starts]
    return np.asarray(token_level_array.token_id, dtype=int)


def _get_candidate_token_ids(
    atom_array: AtomArray,
    *,
    restrict_to_nucleic: bool,
    compute_for_diffused_region_only: bool,
) -> set[int]:
    """Return a set of token_ids to include for scoring."""
    token_starts = get_token_starts(atom_array)
    token_level_array = atom_array[token_starts]
    token_ids = np.asarray(token_level_array.token_id, dtype=int)

    token_mask = np.ones(len(token_ids), dtype=bool)

    if restrict_to_nucleic:
        is_rna = (
            np.asarray(getattr(token_level_array, "is_rna"), dtype=bool)
            if hasattr(token_level_array, "is_rna")
            else np.zeros(len(token_ids), dtype=bool)
        )
        is_dna = (
            np.asarray(getattr(token_level_array, "is_dna"), dtype=bool)
            if hasattr(token_level_array, "is_dna")
            else np.zeros(len(token_ids), dtype=bool)
        )
        token_mask &= (
            (is_rna | is_dna) if (is_rna.any() or is_dna.any()) else token_mask
        )

    if compute_for_diffused_region_only:
        if hasattr(token_level_array, "is_motif_atom"):
            token_mask &= ~np.asarray(token_level_array.is_motif_atom, dtype=bool)
        elif hasattr(token_level_array, "is_motif_token"):
            token_mask &= ~np.asarray(token_level_array.is_motif_token, dtype=bool)

    return set(int(t) for t in token_ids[token_mask].tolist())


def _extract_bp_pairs(
    atom_array: AtomArray,
    *,
    allowed_token_ids: set[int],
) -> set[tuple[int, int]]:
    """Extract unordered base-pair edges from bp-partner annotations.

    Pairs are represented as (min_token_id, max_token_id).
    """
    token_starts = get_token_starts(atom_array)
    token_level_array = atom_array[token_starts]
    token_ids = np.asarray(token_level_array.token_id, dtype=int)
    token_id_to_pos = {int(tid): i for i, tid in enumerate(token_ids.tolist())}

    bp_partner_ann = _get_bp_partners_annotation(atom_array)
    pairs: set[tuple[int, int]] = set()

    for pos, start_idx in enumerate(token_starts.tolist()):
        i_tid = int(token_ids[pos])
        if i_tid not in allowed_token_ids:
            continue

        partners = bp_partner_ann[int(start_idx)]
        if partners is None:
            continue
        if not isinstance(partners, (list, tuple, np.ndarray)):
            continue

        for partner_token_id in partners:
            try:
                j_tid = int(partner_token_id)
            except Exception:
                continue

            if j_tid == i_tid or j_tid not in allowed_token_ids:
                continue

            if j_tid not in token_id_to_pos:
                continue

            a, b = (i_tid, j_tid) if i_tid < j_tid else (j_tid, i_tid)
            pairs.add((a, b))

    return pairs


def _extract_loop_and_paired_token_ids(
    atom_array: AtomArray,
    *,
    allowed_token_ids: set[int],
) -> tuple[set[int], set[int]]:
    """Return (loop_token_ids, paired_token_ids) within the allowed token set."""
    token_starts = get_token_starts(atom_array)
    token_level_array = atom_array[token_starts]
    token_ids = np.asarray(token_level_array.token_id, dtype=int)
    token_id_to_pos = {int(tid): i for i, tid in enumerate(token_ids.tolist())}

    bp_partner_ann = _get_bp_partners_annotation(atom_array)

    loop_token_ids: set[int] = set()
    paired_token_ids: set[int] = set()

    for pos, start_idx in enumerate(token_starts.tolist()):
        i_tid = int(token_ids[pos])
        if i_tid not in allowed_token_ids:
            continue

        partners = bp_partner_ann[int(start_idx)]
        # New semantics:
        # - None => unannotated/masked (NOT a loop)
        # - []   => explicitly unpaired loop
        if partners is None:
            continue
        if not isinstance(partners, (list, tuple, np.ndarray)):
            continue
        if len(partners) == 0:
            loop_token_ids.add(i_tid)
            continue

        for partner_token_id in partners:
            try:
                j_tid = int(partner_token_id)
            except Exception:
                continue

            if j_tid == i_tid or j_tid not in allowed_token_ids:
                continue
            if j_tid not in token_id_to_pos:
                continue
            paired_token_ids.add(i_tid)
            paired_token_ids.add(j_tid)

    return loop_token_ids, paired_token_ids


def compute_from_two_arr(
    gt_arr, pred_arr, restrict_to_nucleic=True, compute_for_diffused_region_only=False
):
    gt_token_ids = _get_token_ids(gt_arr)
    pred_token_ids = _get_token_ids(pred_arr)
    if len(gt_token_ids) != len(pred_token_ids):
        None

    # Restrict to token_ids that are valid in both arrays.
    gt_allowed = _get_candidate_token_ids(
        gt_arr,
        restrict_to_nucleic=restrict_to_nucleic,
        compute_for_diffused_region_only=compute_for_diffused_region_only,
    )
    pred_allowed = _get_candidate_token_ids(
        pred_arr,
        restrict_to_nucleic=restrict_to_nucleic,
        compute_for_diffused_region_only=compute_for_diffused_region_only,
    )
    allowed = gt_allowed & pred_allowed

    if len(allowed) == 0:
        return None

    gt_pairs = _extract_bp_pairs(gt_arr, allowed_token_ids=allowed)
    pred_pairs = _extract_bp_pairs(pred_arr, allowed_token_ids=allowed)

    gt_loop, gt_paired_tokens = _extract_loop_and_paired_token_ids(
        gt_arr, allowed_token_ids=allowed
    )
    pred_loop, _pred_paired_tokens = _extract_loop_and_paired_token_ids(
        pred_arr, allowed_token_ids=allowed
    )

    pair_tp = len(gt_pairs & pred_pairs)
    pair_pred_n = len(pred_pairs)
    pair_gt_n = len(gt_pairs)

    loop_tp = len(gt_loop & pred_loop)
    loop_pred_n = len(pred_loop)
    loop_gt_n = len(gt_loop)

    pair_f1 = _safe_f1_from_sizes(pair_tp, pair_pred_n, pair_gt_n)
    loop_f1 = _safe_f1_from_sizes(loop_tp, loop_pred_n, loop_gt_n)

    pair_weight = len(gt_paired_tokens)
    loop_weight = len(gt_loop)
    total_weight = pair_weight + loop_weight
    if total_weight == 0:
        weighted_f1 = 1.0
    else:
        weighted_f1 = float(
            (pair_weight * pair_f1 + loop_weight * loop_f1) / total_weight
        )

    return pair_f1, loop_f1, weighted_f1


def get_NA_SS_F1(pred_array):
    ## save the original bop_partner annotation
    gt_array = pred_array.copy()

    ## replace by annotating again
    pred_array = annotate_na_ss(
        pred_array,
        NA_only=True,
        planar_only=True,
        overwrite=True,
        p_canonical_bp_filter=0.0,
    )

    try:
        pair_f1, loop_f1, weighted_f1 = compute_from_two_arr(gt_array, pred_array)
    except Exception:
        # fails when returns None because expects three returns
        return {}

    return {
        "pair_f1": pair_f1,
        "loop_f1": loop_f1,
        "weighted_f1": weighted_f1,
    }


class NucleicSSSimilarityMetrics(Metric):
    """Secondary-structure similarity for nucleic acids.

    Reports:
    - `pair_f1`: F1 over basepair edges from token-level bp-partner annotation.
    - `loop_f1`: F1 over explicitly-unpaired loop tokens (`bp_partners == []`).
        Unannotated tokens (`bp_partners is None`) are masked.
    - `weighted_f1`: GT-weighted average of `pair_f1` and `loop_f1`, weighted by
        the prevalence of paired vs loop tokens in the GT.
    """

    def __init__(
        self,
        *,
        restrict_to_nucleic: bool = True,
        compute_for_diffused_region_only: bool = False,
        annotate_predicted_fresh: bool = False,
        annotation_NA_only: bool = False,
        annotation_planar_only: bool = True,
    ):
        super().__init__()
        self.restrict_to_nucleic = restrict_to_nucleic
        self.compute_for_diffused_region_only = compute_for_diffused_region_only
        self.annotate_predicted_fresh = annotate_predicted_fresh
        self.annotation_NA_only = annotation_NA_only
        self.annotation_planar_only = annotation_planar_only

    @property
    def kwargs_to_compute_args(self):
        return {
            "ground_truth_atom_array_stack": ("ground_truth_atom_array_stack",),
            "predicted_atom_array_stack": ("predicted_atom_array_stack",),
        }

    def compute(self, *, ground_truth_atom_array_stack, predicted_atom_array_stack):
        if ground_truth_atom_array_stack is None or predicted_atom_array_stack is None:
            return {}

        pair_f1_list: list[float] = []
        loop_f1_list: list[float] = []
        weighted_f1_list: list[float] = []

        n_valid = 0

        for gt_arr, pred_arr in zip(
            ground_truth_atom_array_stack, predicted_atom_array_stack
        ):
            gt_categories = gt_arr.get_annotation_categories()
            if "bp_partners" not in gt_categories:
                continue

            # Important: predicted AtomArrays are built from a template AtomArray.
            # If that template already carries bp_partners (often GT-derived), the
            # prediction can inherit it, yielding artificially perfect scores.
            # Optionally recompute bp_partners from the *predicted coordinates*.
            if self.annotate_predicted_fresh:
                try:
                    # Infer res name from geometry first
                    pred_arr = _readout_seq_from_struc(
                        pred_arr,
                        central_atom="C1'",
                        threshold=0.5,
                        association_scheme="atom23",
                    )

                    # strip virtuals and set final atom names/elements
                    pred_arr = _cleanup_virtual_atoms_and_assign_atom_name_elements(
                        pred_arr,
                        association_scheme="atom23",
                    )
                except Exception:
                    # this can fail early in training
                    print("could not cleanup virtuals for nucleic ss metric compute")
                    pass
                # clear annotation to avoid potential info leak
                if "bp_partners" in pred_arr.get_annotation_categories():
                    pred_arr.del_annotation("bp_partners")

                # add nucleic-ss annotations
                annotate_na_ss(
                    pred_arr,
                    NA_only=self.annotation_NA_only,
                    planar_only=self.annotation_planar_only,
                    overwrite=True,
                    p_canonical_bp_filter=0.0,
                )
            pred_categories = pred_arr.get_annotation_categories()
            if "bp_partners" not in pred_categories:
                continue

            # Basic sanity check: token counts should match for aligned comparisons
            try:
                pair_f1, loop_f1, weighted_f1 = compute_from_two_arr(
                    gt_arr,
                    pred_arr,
                    restrict_to_nucleic=self.restrict_to_nucleic,
                    compute_for_diffused_region_only=self.compute_for_diffused_region_only,
                )
            except Exception:
                # fails when returns None because expects three returns
                continue

            pair_f1_list.append(pair_f1)
            loop_f1_list.append(loop_f1)
            weighted_f1_list.append(weighted_f1)
            n_valid += 1

        if n_valid == 0:
            return {}

        return {
            "pair_f1": float(np.mean(pair_f1_list)),
            "loop_f1": float(np.mean(loop_f1_list)),
            "weighted_f1": float(np.mean(weighted_f1_list)),
            "n_valid_samples": int(n_valid),
        }
