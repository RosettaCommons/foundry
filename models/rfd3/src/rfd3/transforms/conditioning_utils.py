import networkx as nx
import numpy as np
from atomworks.io.utils.bonds import _atom_array_to_networkx_graph

from foundry.utils.ddp import RankedLogger

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
