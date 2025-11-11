from collections import Counter

import networkx as nx
import numpy as np
from atomworks.io.utils.bonds import _atom_array_to_networkx_graph
from atomworks.ml.utils.token import get_token_starts
from scipy.optimize import linear_sum_assignment

from modelhub.utils.ddp import RankedLogger

global_logger = RankedLogger(__name__, rank_zero_only=False)


#################################################################################
# Training sample conditioning utilities
#################################################################################


def sample_island_tokens(
    array_length,
    island_len_min=5,
    island_len_max=30,
    n_islands_min=1,
    n_islands_max=30,
    max_length=None,
):
    """
    Generate a boolean mask of length `array_length` with random contiguous islands (True segments)
    while optionally constraining the total number of True values.

    Args:
        array_length (int): Total length of the boolean array.
        island_len_min (int): Minimum island length (inclusive).
        island_len_max (int): Maximum island length (inclusive).
        n_islands (int): Number of islands to attempt to generate.
        max_length (int, optional): Maximum allowed total number of True values in the output.
                                    If None, no constraint is applied.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        np.ndarray: Boolean array of length `array_length` with island positions set to True.
    """
    n_islands = np.random.randint(n_islands_min, n_islands_max + 1)

    mask = np.zeros(array_length, dtype=bool)
    for _ in range(n_islands):
        current_total = mask.sum()
        if max_length is not None:
            if current_total >= max_length:
                break
            remaining = max_length - current_total
        else:
            remaining = None  # not used

        # Randomly select a candidate island length.
        candidate_length = np.random.randint(island_len_min, island_len_max + 1)
        candidate_length = min(candidate_length, array_length)  # Fit into array

        # Choose a random starting index ensuring the island fits.
        high_start = array_length - candidate_length
        start = np.random.randint(0, high_start + 1)

        # Evaluate the segment that would be activated.
        segment = mask[start : start + candidate_length]
        new_trues = np.sum(~segment)

        # If we have a maximum True budget and adding all new positions would exceed it, adjust the island.
        if max_length is not None and new_trues > remaining:
            # We try to trim the island so that it adds at most `remaining` new True values.
            count_new = 0
            adjusted_length = 0
            for i in range(candidate_length):
                if not mask[start + i]:
                    count_new += 1
                adjusted_length += 1
                # Once we've added as many new trues as allowed, break.
                if count_new >= remaining:
                    break
            # Only add the island if its adjusted length meets the minimum requirement.
            if adjusted_length < island_len_min:
                continue  # Skip this island and try the next one.
            mask[start : start + adjusted_length] = True
        else:
            # No max constraint or this candidate island fits within the remaining budget.
            mask[start : start + candidate_length] = True

    assert mask.sum() <= array_length, "Generated mask exceeds array length."
    return mask


def sample_subgraph_atoms(
    subarray, p_seed_furthest_from_o=0.8, n_bond_expectation=3, p_fix_all=0.0
):
    """
    subarray: atom array for a single token (e.g. ligand or residue)
    n_bond_expectation: expected number of bonds to sample from geometric distribution
    p_seed_furthest_from_o: probability of choosing the furthest atom from the backbone oxygen atom as seed
    p_fix_all: probability of fixing all atoms in the subarray (skips this function this function)

    returns:
        np.ndarray: boolean mask of atoms to be shown as motif (length of subarray)
    """
    if random_condition(p_fix_all):
        return np.ones(subarray.array_length(), dtype=bool)

    # ... Create graph from subarray
    G = _atom_array_to_networkx_graph(
        subarray,
        annotations=["atom_name"],
        bond_order=False,
        cast_aromatic_bonds_to_same_type=True,
    )

    # ... Determine if subarray is a residue
    is_protein = subarray.is_protein.all()

    # ... Choose a seed atom
    if random_condition(p_seed_furthest_from_o) and is_protein:
        seed_atom = choose_furthest_from_oxygen(G)
    else:
        seed_atom = choose_uniformly_random_atom_name(subarray)

    # ... Sample atoms within n bonds
    # sample bonded fragment to show as motif from geom. distribution
    p = 1 / (1 + n_bond_expectation)
    n_bonds = np.random.geometric(p=p) - 1
    atom_names = get_atom_names_within_n_bonds(
        G, src_atom_name=seed_atom, n_bonds=n_bonds
    )
    is_motif_atom = np.isin(subarray.atom_name, atom_names)

    return is_motif_atom


#################################################################################
# Graph traversal utilities  |  assume each node has "atom_name" attribute
#################################################################################


def get_node_idx_from_atom_name(G, atom_name):
    matches = [
        node for node, data in G.nodes(data=True) if data.get("node_data") == atom_name
    ]

    if len(matches) == 0:
        raise ValueError(
            f"No node with atom_name = '{atom_name}' found. Got {G.nodes(data=True)}"
        )
    elif len(matches) > 1:
        raise ValueError(
            f"Multiple nodes with atom_name = '{atom_name}' found: {matches}. Got {G.nodes(data=True)}"
        )
    else:
        src_node = matches[0]

    return src_node


def get_atom_names_within_n_bonds(G, src_atom_name, n_bonds):
    src_node = get_node_idx_from_atom_name(G, src_atom_name)

    paths = nx.single_source_shortest_path_length(G, source=src_node, cutoff=n_bonds)
    atom_indices = list(paths.keys())
    atom_names = [G.nodes[i]["node_data"] for i in atom_indices]
    return atom_names


def choose_furthest_from_oxygen(G):
    """Chooses furthest node in graph from backbone oxygen atom"""
    src_node = get_node_idx_from_atom_name(G, "O")
    shortest_paths = nx.single_source_shortest_path_length(G, source=src_node)

    max_dist = max(shortest_paths.values())
    furthest_nodes = [node for node, dist in shortest_paths.items() if dist == max_dist]

    sampled_node = np.random.choice(furthest_nodes)
    return G.nodes[sampled_node]["node_data"]


def choose_uniformly_random_atom_name(subarray):
    valid_indices = np.where(subarray.occupancy > 0)[0]
    if len(valid_indices) == 0:
        # raise ValueError("No atoms with occupancy > 0")
        # global_logger.warning("No atoms with occupancy > 0")
        valid_indices = np.arange(subarray.array_length())
    sampled_idx = np.random.choice(valid_indices)
    return subarray.atom_name[sampled_idx]


#################################################################################
# Utility functions
#################################################################################


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
        res_id, chain_id, is_conjoined = indices_to_components(
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
        res_id_, chain_id_, _ = indices_to_components(atom_array_diffused, col_ind)

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


def random_condition(p_cond):
    """
    Made this function because I always get confused by which order the
    inequality should be
    """
    assert 0 <= p_cond <= 1, "p_cond must be between 0 and 1"
    if p_cond == 0:
        return False
    else:
        return np.random.rand() < p_cond


def indices_to_components(atom_array, col_ind):
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
