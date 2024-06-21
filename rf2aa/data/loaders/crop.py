import torch
import numpy as np
import warnings
from scipy.sparse.csgraph import shortest_path
from typing import Optional, Tuple, Dict, Any, List
from rf2aa.data.chain_crop import (
    crop_sm_compl_asmb_contig,
    crop_sm_compl_assembly,
)
from rf2aa.util import replace_missing_with_nearest_neighbors


def get_polymer_chain_index(
    merged_outs,
    preferred_chain: str,
    rng: Optional[np.random.Generator] = None,
) -> Optional[int]:
    if rng is None:
        rng = np.random.default_rng()

    chain_letters_poly = merged_outs["ch_letters_poly"]
    if preferred_chain not in chain_letters_poly:
        warnings.warn(
            f"Preferred chain {preferred_chain} not found in chain_letters_poly {chain_letters_poly}."
        )
        return None

    matching_indices = [
        i for i, c in enumerate(chain_letters_poly) if c == preferred_chain
    ]
    selected_index = rng.choice(matching_indices)
    return selected_index


def get_nonpolymer_chain_index(
    merged_outs,
    preferred_chain: List[Tuple[str, str, str]],
    rng: Optional[np.random.Generator] = None,
) -> Optional[int]:
    if rng is None:
        rng = np.random.default_rng()

    akeys_sm = merged_outs["akeys_sm"]
    representative_keys = [[key[:3] for key in akeys] for akeys in akeys_sm]

    matching_indices = []
    for i, keys in enumerate(representative_keys):
        all_keys_contained = all([key in keys for key in preferred_chain])
        if all_keys_contained:
            matching_indices.append(i)

    assert len(matching_indices) > 0, (
        f"Preferred chain {preferred_chain} not found in representative_keys {representative_keys}."
    )

    selected_index = rng.choice(matching_indices)
    return selected_index + len(merged_outs["Ls_poly"])


def get_preferred_chain_or_interface(
    merged_outs,
    item: Dict[str, Any],
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Optional[int], Optional[Tuple[int, int]]]:
    if rng is None:
        rng = np.random.default_rng()

    polymer_types = ["polypeptide(L)", "polydeoxyribonucleotide", "polyribonucleotide"]

    if "preferred_chain" in item:
        preferred_chain = item["preferred_chain"]
        preferred_chain_type = item["preferred_chain_type"]

        if preferred_chain_type in polymer_types:
            return get_polymer_chain_index(merged_outs, preferred_chain, rng=rng), None
        else:
            return (
                get_nonpolymer_chain_index(merged_outs, preferred_chain, rng=rng),
                None,
            )
    elif "preferred_interface" in item:
        preferred_interface = item["preferred_interface"]
        preferred_interface_type = item["preferred_interface_type"]

        preferred_chain_a = preferred_interface[0]
        preferred_chain_b = preferred_interface[1]
        preferred_chain_type_a = preferred_interface_type[0]
        preferred_chain_type_b = preferred_interface_type[1]

        if preferred_chain_type_a in polymer_types:
            index_a = get_polymer_chain_index(merged_outs, preferred_chain_a, rng=rng)
        else:
            index_a = get_nonpolymer_chain_index(
                merged_outs, preferred_chain_a, rng=rng
            )

        if preferred_chain_type_b in polymer_types:
            index_b = get_polymer_chain_index(merged_outs, preferred_chain_b, rng=rng)
        else:
            index_b = get_nonpolymer_chain_index(
                merged_outs, preferred_chain_b, rng=rng
            )

        return None, (index_a, index_b)
    else:
        raise ValueError(f"Either preferred_chain or preferred_interface must be present in item keys: {item.keys()}.")


def select_preferred_token(
    merged_outs,
    preferred_chain: Optional[int] = None,
    preferred_interface: Optional[Tuple[int, int]] = None,
    interface_selection_cutoff: float = 15.0,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """
    This function dictates which token should be selected to crop around.
    """
    assert not (
        preferred_chain is not None and preferred_interface is not None
    ), "You can only specify one of preferred_chain or preferred_interface"

    if rng is None:
        rng = np.random.default_rng()

    lengths_list = merged_outs["Ls_poly"] + merged_outs["Ls_sm"]
    if preferred_chain is not None:
        assert preferred_chain < len(
            lengths_list
        ), f"preferred_chain {preferred_chain} is out of bounds for lengths_list {lengths_list}"

        start_index = sum(lengths_list[:preferred_chain])
        end_index = sum(lengths_list[: preferred_chain + 1])
        mask_true = merged_outs["mask"][0, :, 1]  # (n_tokens,)

        valid_indices = torch.where(mask_true[start_index:end_index])[0]
        if valid_indices.numel() == 0:
            return rng.integers(start_index, end_index)
        return rng.choice(valid_indices.numpy()) + start_index
    elif preferred_interface is not None:
        index_a = preferred_interface[0]
        index_b = preferred_interface[1]

        start_a = sum(lengths_list[:index_a])
        end_a = sum(lengths_list[: index_a + 1])

        start_b = sum(lengths_list[:index_b])
        end_b = sum(lengths_list[: index_b + 1])

        xyz_true = merged_outs["xyz"][
            0, :, 1
        ]  # (n_tokens, 3): only "token center atoms"
        mask_true = merged_outs["mask"][0, :, 1]  # (n_tokens,)

        xyz_a = xyz_true[start_a:end_a]
        xyz_b = xyz_true[start_b:end_b]

        mask_a = mask_true[start_a:end_a]
        mask_b = mask_true[start_b:end_b]

        dists_a_b = torch.cdist(xyz_a, xyz_b)

        # make sure that masked atoms do not count towards interface distances
        mask_a_b = mask_a[:, None] * mask_b[None, :]
        dists_a_b[~mask_a_b] = interface_selection_cutoff + 1.0

        min_dist_a = torch.min(dists_a_b, dim=1).values
        min_dist_b = torch.min(dists_a_b, dim=0).values

        index_a_is_valid = min_dist_a < interface_selection_cutoff
        index_b_is_valid = min_dist_b < interface_selection_cutoff

        global_index_is_valid = torch.full(
            (xyz_true.shape[0],), False, dtype=torch.bool
        )
        global_index_is_valid[start_a:end_a] = index_a_is_valid
        global_index_is_valid[start_b:end_b] = index_b_is_valid
        if not torch.any(global_index_is_valid):
            # if no valid interface residues are found, return a random residue
            return rng.integers(0, xyz_true.shape[0])

        possible_indices = torch.where(global_index_is_valid)[0]
        return rng.choice(possible_indices.numpy())
    else:
        raise ValueError("Either preferred_chain or preferred_interface must be specified in token selection.")


def radial_crop_index(
    merged_outs,
    crop_index: int,
    crop_size: int,
    epsilon: float = 1e-8,
    rng: Optional[np.random.Generator] = None,
):
    if rng is None:
        rng = np.random.default_rng()

    xyz_true = merged_outs["xyz"][0, :, 1]  # (n_tokens, 3): only "token center atoms"
    mask_true = merged_outs["mask"][0, :, 1]  # (n_tokens,)

    central_token = xyz_true[crop_index][None]  # (1, 3)
    distance_to_central_token = torch.norm(xyz_true - central_token, dim=-1)

    random_bias = rng.normal(0, epsilon, size=distance_to_central_token.shape)
    random_bias = torch.from_numpy(random_bias)
    distance_to_central_token += random_bias

    distance_to_central_token = replace_missing_with_nearest_neighbors(
        distance_to_central_token, mask_true
    )

    crop_indices = torch.topk(
        distance_to_central_token, crop_size, largest=False
    ).indices
    return crop_indices


def contiguous_crop_index(
    merged_outs,
    crop_index: int,
    crop_size: int,
    spatial_contact_distance: float = 10.0,
    epsilon: float = 1e-6,
    rng: Optional[np.random.Generator] = None,
    graph_edge_value: float = 1.0,
    graph_no_edge_value: float = 9999.0,
):
    if rng is None:
        rng = np.random.default_rng()

    graph = merged_outs["bond_feats"] > 0
    xyz_true = merged_outs["xyz"][0, :, 1]  # (n_tokens, 3): only "token center atoms"
    mask_true = merged_outs["mask"][0, :, 1]  # (n_tokens,)
    xyz_true = replace_missing_with_nearest_neighbors(xyz_true, mask_true)

    # connect nearby atoms by an edge
    all_by_all_distances = torch.cdist(xyz_true, xyz_true)
    graph = graph | (all_by_all_distances < spatial_contact_distance)

    # remove self edges
    graph[torch.eye(graph.shape[0]).bool()] = False

    graph_weights = torch.where(graph, graph_edge_value, graph_no_edge_value)

    # add random bias to reduce ambiguity in identical distances
    random_bias = rng.normal(0, epsilon, size=all_by_all_distances.shape)
    random_bias = torch.from_numpy(random_bias)
    random_bias = torch.clip(random_bias, -epsilon, epsilon)
    graph_weights = graph_weights + random_bias

    graph_weights = graph_weights.detach().cpu().numpy()
    path_distances_to_index = shortest_path(
        graph_weights, directed=False, indices=crop_index
    )

    crop_indices = torch.topk(
        torch.from_numpy(path_distances_to_index), crop_size, largest=False
    ).indices
    return crop_indices


def radial_crop(
    merged_outs: Dict[str, torch.Tensor],
    item: Dict[str, Any],
    crop_size: int = 384,
    interface_selection_cutoff: float = 15.0,
    epsilon: float = 1e-8,
    rng: Optional[np.random.Generator] = None,
) -> torch.Tensor:
    xyz = merged_outs["xyz"]
    if xyz.shape[1] < crop_size:
        return torch.arange(xyz.shape[1])

    if rng is None:
        rng = np.random.default_rng()

    preferred_chain, preferred_interface = get_preferred_chain_or_interface(
        merged_outs, item, rng
    )

    preferred_token = select_preferred_token(
        merged_outs,
        preferred_chain=preferred_chain,
        preferred_interface=preferred_interface,
        interface_selection_cutoff=interface_selection_cutoff,
    )

    return radial_crop_index(
        merged_outs,
        crop_index=preferred_token,
        crop_size=crop_size,
        epsilon=epsilon,
        rng=rng,
    )


def contiguous_crop(
    merged_outs: Dict[str, torch.Tensor],
    item: Dict[str, Any],
    crop_size: int = 384,
    interface_selection_cutoff: float = 15.0,
    epsilon: float = 1e-8,
    rng: Optional[np.random.Generator] = None,
) -> torch.Tensor:
    xyz = merged_outs["xyz"]
    if xyz.shape[1] < crop_size:
        return torch.arange(xyz.shape[1])

    if rng is None:
        rng = np.random.default_rng()

    preferred_chain, preferred_interface = get_preferred_chain_or_interface(
        merged_outs, item, rng
    )

    preferred_token = select_preferred_token(
        merged_outs,
        preferred_chain=preferred_chain,
        preferred_interface=preferred_interface,
        interface_selection_cutoff=interface_selection_cutoff,
    )

    return contiguous_crop_index(
        merged_outs,
        crop_index=preferred_token,
        crop_size=crop_size,
        epsilon=epsilon,
        rng=rng,
    )


def radial_crop_sm_compl(
    merged_outs,
    item,
    crop_size: int = 384,
    **unused,
) -> torch.Tensor:
    xyz = merged_outs["xyz"]
    mask = merged_outs["mask"]
    Ls_poly = merged_outs["Ls_poly"]
    Ls_sm = merged_outs["Ls_sm"]

    if xyz.shape[1] < crop_size:
        return torch.arange(xyz.shape[1])

    return crop_sm_compl_assembly(xyz[0], mask[0], Ls_poly, Ls_sm, crop_size)


def contiguous_crop_sm_compl(
    merged_outs,
    item,
    crop_size: int = 384,
    **unused,
):
    xyz = merged_outs["xyz"]
    mask = merged_outs["mask"]
    Ls_poly = merged_outs["Ls_poly"]
    Ls_sm = merged_outs["Ls_sm"]
    bond_feats = merged_outs["bond_feats"]

    if xyz.shape[1] < crop_size:
        return torch.arange(xyz.shape[1])

    return crop_sm_compl_asmb_contig(
        xyz[0],
        mask[0],
        Ls_poly,
        Ls_sm,
        bond_feats,
        crop_size,
        use_partial_ligands=False,
    )


universal_crop_factory = {
    "radial_crop": radial_crop,
    "contig_crop": contiguous_crop,
}
sm_compl_crop_factory = {
    "radial_crop": radial_crop_sm_compl,
    "contig_crop": contiguous_crop_sm_compl,
}
