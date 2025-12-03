from collections import Counter, OrderedDict

import numpy as np
import torch
from atomworks.ml.encoding_definitions import AF3SequenceEncoding
from atomworks.ml.utils.token import (
    get_token_starts,
    spread_token_wise,
)
from biotite.structure import concatenate, infer_elements
from jaxtyping import Float, Int
from rfd3.constants import (
    ATOM14_ATOM_NAMES,
    VIRTUAL_ATOM_ELEMENT_NAME,
    association_schemes,
    association_schemes_stripped,
)
from rfd3.utils.io import (
    build_stack_from_atom_array_and_batched_coords,
)
from scipy.optimize import linear_sum_assignment

from foundry.common import exists
from foundry.utils.ddp import RankedLogger

global_logger = RankedLogger(__name__, rank_zero_only=False)

#######################################################################
# Pythonic Helper functions
#######################################################################


def _remap_outputs(
    xyz: Float[torch.Tensor, "D L 3"], mapping: Int[torch.Tensor, "D L"]
) -> Float[torch.Tensor, "D L 3"]:
    """Helper function to remap outputs using a mapping tensor."""
    for i in range(xyz.shape[0]):
        xyz[i, mapping[i]] = xyz[i].clone()
    return xyz


def _reorder_dict(d: dict) -> OrderedDict:
    """
    Reorders keys in the dictionary to ensure 'metrics' and 'specification' are last (in that order if both present).
    """
    ordered = OrderedDict()
    first_keys = ["task", "diffused_index_map"]
    last_keys = ["metrics", "specification", "inference_sampler"]
    # First
    for k in first_keys:
        if k in d:
            ordered[k] = d[k]
    # Middle
    for k in d:
        if k not in last_keys and k not in first_keys:
            ordered[k] = d[k]
    # Last
    for k in last_keys:
        if k in d:
            ordered[k] = d[k]
    return ordered


#######################################################################
# Biotite-related helper functions
#######################################################################


def _build_atom_array_stack(
    coords,
    src_atom_array,
    sequence_indices,
    sequence_logits,
    allow_sequence_outputs=True,
    read_sequence_from_sequence_head=True,
    association_scheme: str = "atom14",
):
    """
    Wraps around build_atom_array_and_batched_coords to also include additional modifications to atom array
    """
    atom_array_stack = build_stack_from_atom_array_and_batched_coords(
        coords, src_atom_array.copy()
    )

    # ... Spoof empty sequences to alanines
    atom_array_stack.res_name[
        atom_array_stack.is_protein & (atom_array_stack.res_name == "UNK")
    ] = "ALA"

    # ... Add sequence if available
    if allow_sequence_outputs:
        array_list = []
        if read_sequence_from_sequence_head and exists(sequence_logits):
            sequence_encoding = AF3SequenceEncoding()
            for i, (atom_array, seq_indices, seq_logits) in enumerate(
                zip(atom_array_stack, sequence_indices, sequence_logits)
            ):
                # Set residue names
                diffused_mask = ~atom_array.is_motif_atom_with_fixed_seq
                three_letter_sequence = sequence_encoding.decode(
                    seq_indices.cpu().numpy().astype(int)
                )  # [I]

                atom_array.res_name[diffused_mask] = three_letter_sequence[
                    atom_array.token_id
                ][diffused_mask]  # [L]

                # Set bfactor column as entropy of sequence logits
                p = torch.softmax(seq_logits, dim=-1).cpu().numpy()  # shape (L, 32)
                res_entropy = -np.sum(p * np.log(p + 1e-10), axis=-1)  # shape (L,)
                atom_array.b_factor = spread_token_wise(atom_array, res_entropy)
                array_list.append(atom_array.copy())
        else:
            # This automatically deletes virtual atoms and assigns resname, atom name, and elements
            for atom_array in atom_array_stack:
                atom_array = _readout_seq_from_struc(
                    atom_array, association_scheme=association_scheme
                )
                array_list.append(atom_array)

    # Return as list
    atom_array_stack = array_list

    return atom_array_stack


def _cleanup_virtual_atoms_and_assign_atom_name_elements(
    atom_array, association_scheme: str = "atom14"
):
    ## remove virtual atoms based on predicted residue and assign correct atom name and elements
    ret_mask = []
    atom_names = []
    # This is used to indicate which residue is unidentified, probably due to an invalid structure.
    # This is different from the ref_mask, which is used to delete virtual atoms, but this one is used to assign UNK resname for invalid residues.
    invalid_mask = []

    # ... Iterate through each residue.
    # Here we iterate through res_id instead of token_id to avoid some atomization cases or something else.
    res_ids = atom_array.res_id
    res_start_indices = np.concatenate(
        [[0], np.where(res_ids[1:] != res_ids[:-1])[0] + 1]
    )
    res_end_indices = np.concatenate([res_start_indices[1:], [len(res_ids)]])
    warning_issued = False
    for start, end in zip(res_start_indices, res_end_indices):
        res_array = atom_array[start:end]

        is_seq_known = all(
            np.array(res_array.is_motif_atom_with_fixed_seq, dtype=bool)
        ) or all(np.array(res_array.is_motif_atom_unindexed, dtype=bool))

        # ... If sequence is known for the original atom array, just skip
        if is_seq_known:
            ret_mask += [True] * len(res_array)
            invalid_mask += [False] * len(res_array)
            res_name = res_array[0].res_name
            atom_names += res_array.gt_atom_name.tolist()
            continue

        # ... If sequence is unknown for the original atom array, use the predicted / inferred sequence
        res_name = res_array[0].res_name
        if res_name not in association_schemes[association_scheme]:
            global_logger.warning(
                "Model predicted non-protein sequence for diffused residue. Cannot clean up outputs. Assigning unknown residue token."
            )
            warning_issued = True
            ret_mask += [True] * len(res_array)
            invalid_mask += [True] * len(res_array)
            atom_names += res_array.atom_name.tolist()
            continue

        scheme = association_schemes[association_scheme][res_name]
        ret_mask += [True if item is not None else False for item in scheme]
        atom_names += [item.strip() if item is not None else "VX" for item in scheme]
        invalid_mask += [False] * len(scheme)

    if len(atom_names) != atom_array.array_length():
        global_logger.warning(
            f"{atom_names=}\n{atom_array.atom_name=}\nAtom names length {len(atom_names)} does not match original array length {atom_array.array_length()}."
            "\nCould not cleanup atom array!!!"
        )
        if not warning_issued:
            raise ValueError("Atom names length does not match original array length. ")
        return atom_array
    atom_array.atom_name = atom_names
    atom_array.element = np.where(
        atom_array.element == VIRTUAL_ATOM_ELEMENT_NAME,
        infer_elements(atom_names),
        atom_array.element,
    )
    atom_array.res_name[invalid_mask] = np.array(["UNK"] * sum(invalid_mask))
    return atom_array[ret_mask]


def _readout_seq_from_struc(
    atom_array, central_atom="CB", threshold=0.5, association_scheme: str = "atom14"
):
    cur_atom_array_list = []

    # Iterate through each residue
    res_ids = atom_array.res_id
    res_start_indices = np.concatenate(
        [[0], np.where(res_ids[1:] != res_ids[:-1])[0] + 1]
    )
    res_end_indices = np.concatenate([res_start_indices[1:], [len(res_ids)]])

    for start, end in zip(res_start_indices, res_end_indices):
        # ... Check if the current residue is after padding (seq unknown):
        cur_res_atom_array = atom_array[start:end]
        is_seq_known = all(
            np.array(cur_res_atom_array.is_motif_atom_with_fixed_seq, dtype=bool)
        )

        # Here it assumes that every non-protein part has its sequence shown (not padded)
        if not is_seq_known:
            # For Glycine: it doesn't have CB, so set the virtual atom as CA.
            # The current way to handle this is to check if predicted CA and CB are too close, because in the case of glycine and we pad virtual atoms based on CB, CB's coords are set as CA.
            # There might be a better way to do this.
            CA_coord = cur_res_atom_array.coord[cur_res_atom_array.atom_name == "CA"]
            CB_coord = cur_res_atom_array.coord[cur_res_atom_array.atom_name == "CB"]
            if np.linalg.norm(CA_coord - CB_coord) < threshold:
                cur_central_atom = "CA"
            else:
                cur_central_atom = central_atom

            central_mask = cur_res_atom_array.atom_name == cur_central_atom

            # ... Calculate the distance to the central atom
            central_coord = cur_res_atom_array.coord[central_mask][
                0
            ]  # Should only have one central atom anyway
            dists = np.linalg.norm(cur_res_atom_array.coord - central_coord, axis=-1)

            # ... Select virtual atom by the distance. Shouldn't count the central atom itself.
            is_virtual = (dists < threshold) & ~central_mask

            # ... Throw away virtual atoms
            cur_res_atom_array_wo_virtual = cur_res_atom_array[~is_virtual]
            cur_pred_res_atom_names = (
                cur_res_atom_array_wo_virtual.atom_name
            )  # e.g. [N, CA, C, O, CB, V6, V2]

            # ... Iterate over the possible restypes and find the matched one if there is any
            has_restype_assigned = False
            for restype, atom_names in association_schemes_stripped[
                association_scheme
            ].items():
                atom_names = np.array(atom_names)

                # Shouldn't match these two
                if restype in ["UNK", "MSK"]:
                    continue

                # ... Find the index of virtual atom names in the standard atom14 names
                atom_name_idx_in_atom14_scheme = np.array(
                    [
                        np.where(ATOM14_ATOM_NAMES == atom_name)[0][0]
                        for atom_name in cur_pred_res_atom_names
                    ]
                )  # five backbone atoms + some virtual atoms, returning e.g. [0, 1, 2, 3, 4, 11, 7]
                atom14_scheme_mask = np.zeros_like(ATOM14_ATOM_NAMES, dtype=bool)
                atom14_scheme_mask[atom_name_idx_in_atom14_scheme] = True

                # ... Find the matched restype by checking if all the non-None posititons and None positions match
                # This is designed to keep virtual atoms and doesn't assign the atom names for now, which will be handled later.
                if all(x is not None for x in atom_names[atom14_scheme_mask]) and all(
                    x is None for x in atom_names[~atom14_scheme_mask]
                ):
                    cur_res_atom_array.res_name = np.array(
                        [restype] * len(cur_res_atom_array)
                    )
                    cur_atom_array_list.append(cur_res_atom_array)
                    has_restype_assigned = True
                    break
        else:
            cur_atom_array_list.append(cur_res_atom_array)
            has_restype_assigned = True

        # ... Give UNK as the residue name if the mapping fails (unrealistic sidechain)
        if not has_restype_assigned:
            cur_res_atom_array.res_name = np.array(["UNK"] * len(cur_res_atom_array))
            cur_atom_array_list.append(cur_res_atom_array)

    cur_atom_array = concatenate(cur_atom_array_list)

    return cur_atom_array


#######################################################################
# Unindexed output parsing
#######################################################################


def _reassign_unindexed_token_chains(atom_array):
    if np.any((mask := atom_array.is_motif_atom_unindexed)):
        # HACK: Since res_ids are the same, we should save them with a different chain index.
        atom_array.chain_id[mask] = "X"
        atom_array.res_id[mask] = atom_array.orig_res_id[mask]

        # Parse to separate chains
        starts = get_token_starts(atom_array)
        unindexed_starts = starts[mask[starts]]
        token_breaks = atom_array[
            unindexed_starts
        ].is_motif_atom_unindexed_motif_breakpoint
        token_group_id = np.cumsum(token_breaks, dtype=int)  # Group by motif breaks
        token_chain_id = np.array([f"X{i}" for i in token_group_id])

        chains = atom_array.chain_id[starts]
        chains[mask[starts]] = token_chain_id
        atom_array.chain_id = spread_token_wise(atom_array, chains)
    return atom_array


def process_unindexed_outputs(
    atom_array,
    match_atom_names=True,
    insert_guideposts=False,
    verbose=False,
):
    """
    Process design outputs containing unindexed tokens.
    Returns metadata such as the assigned positional indices from the input indices
    and the RMSD of the unindexed tokens.

    Returns:
        - Diffused atom array (without additional unindexed tokens)
        - Metadata:
            - diffused_indices: keys = original (contig) indices, values = diffused indices
            - insertion_rmsd: overall RMSD of insertion
            - insertion_rmsd_by_residue: RMSD of insertion for each token

        TODO: Add additional geometry metrics such as bond angle non-ideality, clashes etc.
        TODO: atom1d conditioning adherence - does the output contain HBonds in the right places, correct rasa values?
    """
    # ... Find assignments based on greedy search
    starts = get_token_starts(atom_array, add_exclusive_stop=True)

    # [N_diffused,]
    atom_array_diffused = atom_array[~atom_array.is_motif_atom_unindexed].copy()
    global_idx = np.arange(atom_array.array_length())[
        ~atom_array.is_motif_atom_unindexed
    ]

    metadata = {
        "diffused_index_map": {},
        "insertion_rmsd_by_token": {},
        "join_point_rmsd_by_token": {},
        "insertion_rmsd_by_restype": {},
    }
    token_maes = []
    token_rmcds = []
    n_conjoined_residues = 0

    # Initialize an empty array
    inserted_mask = np.full_like(atom_array_diffused.is_motif_atom_unindexed, False)

    for start, end in zip(starts[:-1], starts[1:]):
        token = atom_array[start:end]
        if not token.is_motif_atom_unindexed.all():
            continue

        if "src_component" in token.get_annotation_categories():
            token_pdb_id = token.src_component[0]
        else:
            raise ValueError(
                "Missing annotation 'src_component' in token. Is this inference?"
            )

        if "src_sym_component" in token.get_annotation_categories():
            # if symmetry, token_pdb_id are updated to match the symmetrized component
            token_pdb_id = token.src_sym_component[0]

        res_name = token.res_name[0]

        # ... Calculate [N_unindex, N_diffused] distance matrix
        dists = np.linalg.norm(
            token.coord[:, None] - atom_array_diffused.coord[None, :], axis=-1
        )

        # ... Match atom indices based on atom names (mask out non-identical) and remove already inserted
        dists[:, inserted_mask.copy()] = np.inf
        if match_atom_names:
            matching_atom_name = (
                token.atom_name[:, None] == atom_array_diffused.atom_name[None, :]
            )
            dists[~matching_atom_name] = np.inf

        # ... Find the res_id's in the diffused regions belonging to the diffused indices
        row_ind, col_ind = linear_sum_assignment(dists)
        res_id, chain_id, is_conjoined = indices_to_components_(
            atom_array_diffused, col_ind
        )
        n_conjoined_residues += int(is_conjoined)

        # ... Recompute distance indices based on single residue pairings only
        token_match = (atom_array_diffused.res_id == res_id) & (
            atom_array_diffused.chain_id == chain_id
        )
        dists[:, ~token_match] = np.nan
        BIG = 1e12
        dists = np.nan_to_num(dists, nan=BIG, posinf=BIG, neginf=BIG)
        row_ind, col_ind = linear_sum_assignment(dists)
        res_id_, chain_id_, _ = indices_to_components_(atom_array_diffused, col_ind)

        assert (res_id_ == res_id) & (chain_id_ == chain_id)
        inserted_mask = np.logical_or(inserted_mask, token_match)

        # ... Compute metrics based on the new distances
        diff = token.coord[row_ind] - atom_array_diffused.coord[col_ind]
        token_rmsd = float(np.sqrt((diff**2).sum(-1).mean()))
        token_rmcd = float(np.cbrt((np.abs(diff) ** 3).sum(-1).mean()))
        token_mae = float((np.abs(diff)).sum(-1).mean())

        metadata["insertion_rmsd_by_token"][token_pdb_id] = token_rmsd
        token_maes.append(token_mae)
        token_rmcds.append(token_rmcd)

        if res_name not in metadata["insertion_rmsd_by_restype"]:
            metadata["insertion_rmsd_by_restype"][res_name] = []
        metadata["insertion_rmsd_by_restype"][res_name].append(token_rmsd)
        if not np.any(np.isin(token.atom_name, ["N", "CA", "C", "O"])):
            if np.sum(token.atomize) == 1:
                join_atom = np.where(token.atomize)[0][0]
            elif "CB" in token.atom_name:
                join_atom = np.where(token.atom_name == "CB")[0][0]
            else:
                join_atom = None

            if join_atom is None:
                global_logger.warning(
                    f"Token {token_pdb_id} does not contain backbone atoms or CB, skipping join point distance calculation {token}."
                )
            else:
                dist = float(dists[row_ind[join_atom], col_ind[join_atom]])
            metadata["join_point_rmsd_by_token"][token_pdb_id] = dist

        metadata["diffused_index_map"][token_pdb_id] = f"{chain_id}{res_id}"

        # ... Decide whether to cleanup guideposts or not
        if insert_guideposts:
            atom_array_diffused.coord[global_idx[col_ind]] = token.coord[row_ind]
            if token.is_motif_atom_with_fixed_seq[0]:
                atom_array_diffused.res_name[token_match] = token.res_name[0]
            # atom_array_diffused.is_motif_token[token_match] = True
            # atom_array_diffused.is_motif_atom[global_idx[col_ind]] = True
            atom_array_diffused.is_motif_atom_with_fixed_coord[global_idx[col_ind]] = (
                True
            )

    # ... Calculate global metrics
    def safe_mean(x):
        """Return nan-safe mean for empty or nan arrays."""
        x = np.asarray(x, float)
        if x.size == 0 or not np.isfinite(x).any():
            return float("nan")
        return float(np.nanmean(x))

    metadata["insertion.mae"] = safe_mean(token_maes)
    metadata["insertion.rmcd"] = safe_mean(token_rmcds)
    metadata["insertion_rmsd"] = safe_mean(
        list(metadata["insertion_rmsd_by_token"].values())
    )
    metadata["join_point_rmsd"] = safe_mean(
        list(metadata["join_point_rmsd_by_token"].values())
    )
    metadata["insertion_rmsd_by_restype"] = {
        a: safe_mean(v) for a, v in metadata["insertion_rmsd_by_restype"].items()
    }
    metadata["n_conjoined_residues"] = n_conjoined_residues

    if not verbose:
        metadata = {
            k: v for k, v in metadata.items() if not k.startswith("insertion_rmsd_by_")
        }

    return atom_array_diffused, metadata


def indices_to_components_(atom_array, col_ind):
    """
    Fetch chain and resids in atom array given a set of raw indices
    will return 'conjoined' if indices to not map to a unique residue
    """
    res_ids, chain_ids = (
        atom_array.res_id[col_ind],
        atom_array.chain_id[col_ind],
    )
    if len(set(res_ids.tolist())) > 1 or len(set(chain_ids.tolist())) > 1:
        global_logger.warning(
            f"Unindexed token mapped its atoms to multiple diffused residues: {res_ids.tolist()} and chains {chain_ids.tolist()}."
        )
        # Handle by majority
        pair_counts = Counter(zip(chain_ids.tolist(), res_ids.tolist()))
        (chain_id, res_id), _ = pair_counts.most_common(1)[0]
        conjoined = True
    else:
        res_id = res_ids[0]
        chain_id = chain_ids[0]
        conjoined = False

    return res_id, chain_id, conjoined
