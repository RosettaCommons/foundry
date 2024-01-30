import numpy as np
import torch
import networkx as nx
from typing import Dict, Optional, Tuple, List, Set, Any
import rf2aa.cifutils as cifutils
from rf2aa.util import get_ligand_atoms_bonds, cif_ligand_to_xyz, cif_ligand_to_obmol, get_automorphs


def get_left_out_chain_letters(
    chains: Dict[str, cifutils.Chain],
    lig_partners: List[
        Tuple[List[Tuple[str, str, str]], List[Tuple[str, int]], float, float, str]
    ],
) -> List[str]:
    """get_left_out_chain_letters This function returns a list of chain letters
    that were not already loaded into lig partners but are nonpoly chains.

    Args:
        chains (Dict[str, cifutils.Chain]): From the cif-parsed PDB database.
        lig_partners (List[Tuple[List[Tuple[str, str, str]], List[Tuple[str, int]], float, float, str]]): A list
            of ligand partners from the original data frame.

    Returns:
        List[str]: A list of chain letters.
    """
    loaded_chains = [p[0][0][0] for p in lig_partners]
    all_names = set()
    for partner in lig_partners:
        for ligand_key in partner[0]:
            all_names.add(ligand_key[-1])

    remaining_nonpoly_chains = [
        chain.id for chain in chains.values() if chain.type == "nonpoly"
    ]
    remaining_nonpoly_chains = [
        c for c in remaining_nonpoly_chains if c not in loaded_chains
    ]
    remaining_nonpoly_chains = [
        c
        for c in remaining_nonpoly_chains
        if next(iter(chains[c].atoms.keys()))[-2] in all_names
    ]
    return remaining_nonpoly_chains


def merge_chain_sets(
    chains: Dict[str, cifutils.Chain],
    covale: List[cifutils.Bond],
    chain_list: List[str],
    exclude_covalent_to_protein: bool = False,
) -> List[Set[str]]:
    """merge_chain_sets This function takes in a list of chain letters
    representing ligands in the cif assembly, and merges the letters into sets
    if the ligand chains are connected by covalent bonds, e.g. multi-chain ligands.

    Args:
        chains (Dict[str, cifutils.Chain]): From the cif-parsed PDB database.
        covale (List[cifutils.Bond]): From the cif-parsed PDB database.
        chain_list (List[str]): A list of chains.
        exclude_covalent_to_protein (bool, optional): Set to True to remove
            any ligand set that is covalently bonded to a protein chain. Defaults to False.

    Returns:
        _type_: A list of chain letter sets, representing possibly multi-chain ligands.
    """
    chain_sets = [set([c]) for c in chain_list]
    for covalent_bond in covale:
        key_a = covalent_bond.a[0]
        key_b = covalent_bond.b[0]

        key_set_index_a = None
        key_set_index_b = None
        for key_set_index, key_set in enumerate(chain_sets):
            if key_a in key_set:
                key_set_index_a = key_set_index
            if key_b in key_set:
                key_set_index_b = key_set_index
        if exclude_covalent_to_protein:
            if (
                chains[key_a].type.startswith("polypeptide")
                and key_set_index_b is not None
            ):
                _ = chain_sets.pop(key_set_index_b)
                continue
            elif (
                chains[key_b].type.startswith("polypeptide")
                and key_set_index_a is not None
            ):
                _ = chain_sets.pop(key_set_index_a)
                continue
        if (
            key_set_index_a is None
            or key_set_index_b is None
            or key_set_index_a == key_set_index_b
        ):
            continue

        # Note: we need to pop the larger index before the smaller index,
        # because the larger index would change if we popped the smaller index first.
        key_set_index_larger, key_set_index_smaller = max(
            key_set_index_a, key_set_index_b
        ), min(key_set_index_a, key_set_index_b)

        key_set_larger = chain_sets.pop(key_set_index_larger)
        key_set_smaller = chain_sets.pop(key_set_index_smaller)
        merged_key_set = key_set_larger.union(key_set_smaller)
        chain_sets.append(merged_key_set)
    return chain_sets


def get_ligand_partial_akeys(
    chains: Dict[str, cifutils.Chain],
    chain_set: Set[str],
    covale: List[cifutils.Bond],
    exclude_covalent_to_protein: bool = False,
) -> List[Tuple[str, str, str]]:
    """get_ligand_partial_akeys For a given set of chains representing a
    possibly multi-chain ligand, gets the partial (chain_letter, residue_number, molecule_id)
    list of atom keys representing that ligand. This handles multi-residue ligands.

    Args:
        chains (Dict[str, cifutils.Chain]): From the cif-parsed PDB database.
        chain_set (List[Set[str]]): A set of chains that are covalently bonded.
        covale (List[cifutils.Bond]): From the cif parsed PDB database.
        exclude_covalent_to_protein (bool, optional): See the previous function. Defaults to False.

    Returns:
        _type_: A set of partial akeys.
    """
    ligand_partial_akeys = set()
    for chain_id in chain_set:
        chain = chains[chain_id]
        for atom_key in chain.atoms.keys():
            ligand_partial_akeys.add(atom_key[:3])

    if not exclude_covalent_to_protein:
        original_akeys = ligand_partial_akeys.copy()
        for bond in covale:
            key_a = bond.a[:3]
            key_b = bond.b[:3]
            if key_a in original_akeys:
                ligand_partial_akeys.add(key_b)
            elif key_b in original_akeys:
                ligand_partial_akeys.add(key_a)
    return list(ligand_partial_akeys)


def get_nonpoly_akeys(
    chains: Dict[str, cifutils.Chain], akeys: List[Tuple[str, str, str]]
) -> List[Tuple[str, str, str]]:
    nonpoly_akeys = [akey for akey in akeys if chains[akey[0]].type == "nonpoly"]
    return nonpoly_akeys


def get_partial_akeys(
    akeys: List[Tuple[str, str, str, str]]
) -> List[Tuple[str, str, str]]:
    return list(set([akey[:3] for akey in akeys]))


def compute_sorted_id_count_string(
    chains: Dict[str, cifutils.Chain],
    partial_or_full_akeys: List[Tuple[str, str, str, str]],
    exclude_covalent_to_protein: bool = True,
) -> str:
    """compute_sorted_id_count_string Turns a list of akeys into a minimal
    string representation of all of the molecules present in that list of akeys.
    This way you can compare two list of akeys to see if they have the same
    molecules and number of copies of each molecule.

    Args:
        chains (Dict[str, cifutils.Chain]): From the cif-parsed PDB database.
        partial_or_full_akeys (List[Tuple[str, str, str, str]]): A list of full (or partial) akeys.
        exclude_covalent_to_protein (bool, optional): _description_. Defaults to True.

    Returns:
        str: A string representation of a bag of molecule ids.
    """
    if exclude_covalent_to_protein:
        partial_or_full_akeys = get_nonpoly_akeys(chains, partial_or_full_akeys)
    partial_akeys = get_partial_akeys(partial_or_full_akeys)
    id_string_list = [akey[2] for akey in partial_akeys]
    sorted_string_list = sorted(id_string_list)
    return "_".join(sorted_string_list)


def get_query_left_out_map(
    sorted_query_string_list: List[str], sorted_left_out_string_list: List[str]
):
    """get_query_left_out_map Maps query strings to left out strings.

    Args:
        sorted_query_string_list (List[str]): A list of strings.
        sorted_left_out_string_list (List[str]): A list of strings.

    Returns:
        _type_: A mapping from strings in the query list to indices in the left out list.
    """
    query_string_to_left_out_string_map = {}

    for query_string in sorted_query_string_list:
        for index, left_out_string in enumerate(sorted_left_out_string_list):
            if query_string == left_out_string:
                if query_string not in query_string_to_left_out_string_map:
                    query_string_to_left_out_string_map[query_string] = []
                query_string_to_left_out_string_map[query_string].append(index)
    return query_string_to_left_out_string_map


def get_graph_from_partial_akeys(
    chains: Dict[str, cifutils.Chain],
    covale: List[cifutils.Bond],
    partial_akeys: List[Tuple[str, str, str]],
) -> nx.Graph:
    """get_graph_from_partial_akeys Gets a networkx graph from a list of partial akeys.
    This is needed to do supernode graph ismorphism on the connectivity of multi-chain/multi-residue ligands.

    Args:
        chains (Dict[str, cifutils.Chain]): From the cif-parsed PDB database.
        covale (List[cifutils.Bond]): From the cif-parsed PDB database.
        partial_akeys (List[Tuple[str, str, str]]): A list of partial akeys.

    Returns:
        nx.Graph: A graph representation of the connectivity of the ligand pieces.
    """
    graph = nx.Graph()
    nodes = [(i, {"id": akey[2]}) for i, akey in enumerate(partial_akeys)]
    graph.add_nodes_from(nodes)

    potential_bonds = covale
    unique_chain_letters = set([akey[0] for akey in partial_akeys])
    for chain_letter in unique_chain_letters:
        potential_bonds.extend(chains[chain_letter].bonds)

    for bond in potential_bonds:
        key_a = bond.a[:3]
        key_b = bond.b[:3]
        if key_a == key_b:
            continue
        if key_a in partial_akeys and key_b in partial_akeys:
            graph.add_edge(partial_akeys.index(key_a), partial_akeys.index(key_b))
    return graph


def match_partial_akeys(
    chains: Dict[str, cifutils.Chain],
    covale: List[cifutils.Bond],
    query_akeys: List[Tuple[str, str, str]],
    left_out_akeys: List[Tuple[str, str, str]],
) -> Optional[Dict[Tuple[str, str, str], Tuple[str, str, str]]]:
    """match_partial_akeys Matches akeys based on graph connectivity.

    Args:
        chains (Dict[str, cifutils.Chain]): From the cif-parsed PDB database.
        covale (List[cifutils.Bond]): From the cif-parsed PDB database.
        query_akeys (List[Tuple[str, str, str]]): A list of partial akeys from the query ligand.
        left_out_akeys (List[Tuple[str, str, str]]): A list of partial akeys from the left out ligand.

    Returns:
        Optional[List[Tuple[str, str, str]]]: None if the lists do not match, otherwise returns a
        dictionary mapping query akeys to left out akeys.
    """
    query_graph = get_graph_from_partial_akeys(chains, covale, query_akeys)
    left_out_graph = get_graph_from_partial_akeys(chains, covale, left_out_akeys)

    matcher = nx.algorithms.isomorphism.GraphMatcher(
        query_graph, left_out_graph, node_match=lambda a, b: a["id"] == b["id"]
    )
    if matcher.is_isomorphic():
        return {
            query_akeys[i]: left_out_akeys[matcher.mapping[i]]
            for i in range(len(query_akeys))
        }
    else:
        return None


def get_full_akeys_from_partial(
    full_akeys_sm: List[Tuple[str, str, str, str]],
    query_to_left_out_map: Dict[Tuple[str, str, str], Tuple[str, str, str]],
) -> List[Tuple[str, str, str, str]]:
    """get_full_akeys_from_partial Expands a list of partial akeys mapped to query akeys
    to include the full akey + atom name representation.

    Args:
        full_akeys_sm (List[Tuple[str, str, str, str]]): The full akey list of the query ligand.
        query_to_left_out_map (Dict[Tuple[str, str, str], Tuple[str, str, str]]): The
        partial akey map from query to left out ligand.

    Returns:
        List[Tuple[str, str, str, str]]: A full list of akeys for the left out ligand.
    """
    left_out_akeys_full = []
    for full_akey in full_akeys_sm:
        partial_akey = full_akey[:3]
        if partial_akey not in query_to_left_out_map:
            return None

        mapped_akey = query_to_left_out_map[partial_akey]
        left_out_akey = mapped_akey + (full_akey[3],)
        left_out_akeys_full.append(left_out_akey)
    return left_out_akeys_full


def get_left_out_ch2xfs(
    left_out_chain_set: Set[str], asmb_xfs: List[Tuple[str, np.ndarray]]
) -> List[Dict[str, int]]:
    """get_left_out_ch2xfs Given a chain set, returns the possible
    transform index dictinoaries that are needed to compute
    the coordinates of the ligand in that chain set.

    Args:
        left_out_chain_set (Set[str]): A set of chains representing a possibly multi-residue ligand.
        asmb_xfs (List[Tuple[str, np.ndarray]]): A list of chain-letter, transform paris.

    Returns:
        List[Dict[str, int]]: A list of all of the different transforms that this ligand
        may undergo.
    """
    if len(left_out_chain_set) == 1:
        left_out_chain_letter = next(iter(left_out_chain_set))
        left_out_ch2xf_list = []
        for transform_index, (transform_chain, _) in enumerate(asmb_xfs):
            if transform_chain == left_out_chain_letter:
                left_out_ch2xf = {left_out_chain_letter: transform_index}
                left_out_ch2xf_list.append(left_out_ch2xf)
    else:
        # If it is a multi-chain ligand, we assume that
        # there is only one transform in the cif file that
        # transforms all of the chains in the ligand. It is very costly
        # to check all possible combinations of transforms.
        left_out_ch2xf = {}
        for left_out_chain_letter in left_out_chain_set:
            for transform_index, (transform_chain, _) in enumerate(asmb_xfs):
                if transform_chain == left_out_chain_letter:
                    left_out_ch2xf[left_out_chain_letter] = transform_index
                    break
        left_out_ch2xf_list = [left_out_ch2xf]
    return left_out_ch2xf_list


def pad_cat_symmetry_tensors(
    tensor_list: List[torch.Tensor], fill_value: Any
) -> torch.Tensor:
    """pad_cat_symmetry_tensors Given a list of tensors
    of shapes (n_symm_0, length_0, ...), (n_symm_1, length_1, ...)
    pads each tensor with fill value in the first dimension up to
    max(n_symm_i) over i, and then concatenates the tensors in the length
    dimension to produce a tensor of shape (max(n_symm_i), sum(length_i), ...).

    Args:
        tensor_list (List[torch.Tensor]): A list of tensors.
        fill_value (Any): Fill value for padding.

    Returns:
        torch.Tensor: The padded, concatenated tensor result.
    """
    padded_tensor_list = []
    max_symmetry = max([t.shape[0] for t in tensor_list])

    for tensor in tensor_list:
        tensor_shape = (max_symmetry,) + tensor.shape[1:]
        padded_tensor = torch.full(
            tensor_shape,
            fill_value=fill_value,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        padded_tensor[: tensor.shape[0]] = tensor
        padded_tensor_list.append(padded_tensor)
    padded_tensor = torch.cat(padded_tensor_list, dim=1)
    return padded_tensor


def get_extra_identical_copies_from_chains(
    chains: Dict[str, cifutils.Chain],
    covale: List[cifutils.Bond],
    asmb_xfs: List[Tuple[str, np.ndarray]],
    xyz_sm: torch.Tensor,
    mask_sm: torch.Tensor,
    Ls_sm: List[int],
    akeys_sm: List[Tuple[str, str, str, str]],
    lig_partners: List[
        Tuple[List[Tuple[str, str, str]], List[Tuple[str, int]], float, float, str]
    ],
    exclude_covalent_to_protein: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """get_extra_identical_copies_from_chains This function pulls out
    extra identical copies of ligands in lig_partners from the cif assembly, and
    appends them to the correct existing coordinates in xyz_sm and mask_sm, and returns the result.

    Args:
        chains (Dict[str, cifutils.Chain]): From the cif-parsed PDB database.
        covale (List[cifutils.Bond]): From the cif-parsed PDB database.
        asmb_xfs (List[Tuple[str, np.ndarray]]): From the cif-parsed PDB database.
        xyz_sm (torch.Tensor): A tensor of shape (n_symm, sum(Ls_sm), 3) representing the coordinates of the ligands.
        mask_sm (torch.Tensor): A tensor of shape (n_symm, sum(Ls_sm)) representing the mask of the ligands.
        Ls_sm (List[int]): A list of ligand lengths.
        akeys_sm (List[Tuple[str, str, str, str]]): A list of full akeys.
        lig_partners (List[Tuple[List[Tuple[str, str, str]], List[Tuple[str, int]], float, float, str]]): A list
            representing ligand partners from the training dataframe.
        exclude_covalent_to_protein (bool, optional): Probably should keep this as False. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Updated xyz_sm and mask_sm tensors.
    """
    left_out_chain_letters = get_left_out_chain_letters(chains, lig_partners)
    left_out_chain_sets = merge_chain_sets(
        chains,
        covale,
        left_out_chain_letters,
        exclude_covalent_to_protein=exclude_covalent_to_protein,
    )

    left_out_partial_akeys_list = [
        get_ligand_partial_akeys(
            chains,
            chain_set,
            covale,
            exclude_covalent_to_protein=exclude_covalent_to_protein,
        )
        for chain_set in left_out_chain_sets
    ]
    sorted_query_string_list = [
        compute_sorted_id_count_string(
            chains, akeys, exclude_covalent_to_protein=exclude_covalent_to_protein
        )
        for akeys in akeys_sm
    ]
    sorted_left_out_string_list = [
        compute_sorted_id_count_string(
            chains, akeys, exclude_covalent_to_protein=exclude_covalent_to_protein
        )
        for akeys in left_out_partial_akeys_list
    ]

    query_string_to_left_out_string_map = get_query_left_out_map(
        sorted_query_string_list, sorted_left_out_string_list
    )

    xyz_sm_list_with_extra_copies = []
    mask_sm_list_with_extra_copies = []
    cumulative_length = 0
    for query_index, query_string in enumerate(sorted_query_string_list):
        query_length = Ls_sm[query_index]
        query_akeys = akeys_sm[query_index]
        query_xyz = xyz_sm[:, cumulative_length: cumulative_length + query_length]
        query_mask = mask_sm[:, cumulative_length: cumulative_length + query_length]

        partial_query_akeys = get_partial_akeys(query_akeys)

        try:
            left_out_index_list = query_string_to_left_out_string_map.pop(query_string)
        except KeyError:
            left_out_index_list = []

        for left_out_index in left_out_index_list:
            left_out_partial_akeys = left_out_partial_akeys_list[left_out_index]

            query_to_left_out_map = match_partial_akeys(
                chains, covale, partial_query_akeys, left_out_partial_akeys
            )
            if query_to_left_out_map is None:
                continue

            left_out_full_akeys = get_full_akeys_from_partial(
                query_akeys, query_to_left_out_map
            )
            if left_out_full_akeys is None:
                continue

            left_out_atoms, left_out_bonds = get_ligand_atoms_bonds(
                left_out_partial_akeys, chains, covale
            )
            left_out_chain_set = left_out_chain_sets[left_out_index]
            left_out_ch2xf_list = get_left_out_ch2xfs(left_out_chain_set, asmb_xfs)

            for left_out_ch2xf in left_out_ch2xf_list:
                left_out_xyz, left_out_occ, _, _, _ = cif_ligand_to_xyz(
                    left_out_atoms, asmb_xfs, left_out_ch2xf, left_out_full_akeys
                )

                if (left_out_occ <= 0.0).all():
                    continue

                left_out_mask = left_out_occ > 0
                left_out_mol, _ = cif_ligand_to_obmol(
                    left_out_xyz, left_out_full_akeys, left_out_atoms, left_out_bonds
                )
                left_out_xyz, left_out_mask = get_automorphs(
                    left_out_mol, left_out_xyz, left_out_mask
                )
                query_xyz = torch.cat([query_xyz, left_out_xyz], dim=0)
                query_mask = torch.cat([query_mask, left_out_mask], dim=0)
        cumulative_length += query_length
        xyz_sm_list_with_extra_copies.append(query_xyz)
        mask_sm_list_with_extra_copies.append(query_mask)

    if len(xyz_sm_list_with_extra_copies) == 0:
        return xyz_sm, mask_sm

    xyz_sm = pad_cat_symmetry_tensors(xyz_sm_list_with_extra_copies, fill_value=0.0)
    mask_sm = pad_cat_symmetry_tensors(mask_sm_list_with_extra_copies, fill_value=False)
    return xyz_sm, mask_sm
