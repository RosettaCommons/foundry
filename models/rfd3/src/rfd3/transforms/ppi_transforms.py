# from atomworks.ml.utils.token import get_token_masks, get_token_starts
from typing import Any

import biotite.structure as struc
import numpy as np
from assertpy import assert_that
from atomworks.ml.preprocessing.utils.structure_utils import (
    get_atom_mask_from_cell_list,
)
from atomworks.ml.transforms._checks import (
    check_atom_array_annotation,
    check_contains_keys,
    check_is_instance,
)
from atomworks.ml.transforms.atom_array import atom_id_to_atom_idx, atom_id_to_token_idx
from atomworks.ml.transforms.base import Transform
from atomworks.ml.transforms.crop import (
    get_spatial_crop_center,
    get_token_count,
    resize_crop_info_if_too_many_atoms,
)
from atomworks.ml.utils.token import (
    get_af3_token_center_coords,
    get_af3_token_center_masks,
    get_token_starts,
    spread_token_wise,
)
from biotite.structure import AtomArray
from rfd3.transforms.conditioning_utils import sample_island_tokens
from scipy.spatial import KDTree

from foundry.utils.ddp import RankedLogger

ranked_logger = RankedLogger(__name__, rank_zero_only=True)

# NOTE: This transform is based off of `rf_diffusion_aa.rf_diffusion.ppi.FindHotspotsTrainingTransform`
# However, this is progressing piecewise, and many features of that transform are not yet implemented.
# If this seems to be working, those should definitely be added in the future!

# NOTE: In contrast to RFD, we are providing hotspots at the atom level, not the residue level.
# Future hotspot subsampling schemes might want to avoid giving redundant information via (say) bonded atoms


def get_hotspot_atoms(atom_array, binder_pn_unit_iid, distance_cutoff=4.5):
    """Get hotspot atoms for a given distance cutoff.

    Args:
        atom_array (AtomArray): The atom array containing the protein structure.
        binder_pn_unit_iid (str): The chain ID of the binder (diffused chain).
        distance_cutoff (float): The interchain distance cutoff that defines hotspot atoms.

    Hotspots are atoms on non-binder chains that are within the distance cutoff of some residue on the binder.
    Residue distances are computed as the minimum pairwise distance between the two atoms.
    """

    # We can only perform distance computations on atoms with non-NaN coordinates
    nan_coords_mask = np.any(np.isnan(atom_array.coord), axis=1)
    non_nan_atom_array = atom_array[~nan_coords_mask]

    binder_atom_array = non_nan_atom_array[
        non_nan_atom_array.pn_unit_iid == binder_pn_unit_iid
    ]

    # Perform the hotspot computation
    cell_list = struc.CellList(non_nan_atom_array, cell_size=distance_cutoff)

    full_contacting_atom_mask = get_atom_mask_from_cell_list(
        binder_atom_array.coord, cell_list, len(non_nan_atom_array), distance_cutoff
    )  # (n_query, n_cell_list)
    contacting_atoms_mask = np.any(full_contacting_atom_mask, axis=0)  # (n_cell_list,)

    # Filter out atoms in the binder chain
    non_query_atoms_mask = non_nan_atom_array.pn_unit_iid != binder_pn_unit_iid
    hotspot_atom_mask = contacting_atoms_mask & non_query_atoms_mask

    # Convert from mask over non-nan coords to mask over all coords
    full_hotspot_atom_mask = np.zeros(len(atom_array), dtype=bool)
    full_hotspot_atom_mask[~nan_coords_mask] = hotspot_atom_mask

    return full_hotspot_atom_mask


def get_secondary_structure_types(atom_array: AtomArray) -> np.ndarray:
    """Get the secondary structure types for a given atom array.

    For now, only three categories will be one-hot encoded: helix, sheet, and loop.
    """
    ss_types = np.zeros((atom_array.array_length(), 3), dtype=bool)

    # HACK: Temporarily overwrite res_id with token_id so that the sse_array will have length n_tokens
    actual_res_id = atom_array.res_id.copy()
    atom_array.res_id = atom_array.token_id

    # Since annotate_sse detects chainbreaks based on res_id discontinuities, we create discontinuities where needed
    _, chain_offsets = np.unique(atom_array.chain_iid, return_inverse=True)
    atom_array.res_id += chain_offsets

    # Compute secondary structure information
    sse_array = struc.annotate_sse(atom_array)
    assert len(sse_array) == len(
        np.unique(atom_array.token_id)
    ), "SSE array length does not match number of tokens."

    # Restore original res_id
    atom_array.res_id = actual_res_id

    sse_array = spread_token_wise(atom_array, sse_array)
    ss_types[:, 0] = sse_array == "a"
    ss_types[:, 1] = sse_array == "b"
    ss_types[:, 2] = sse_array == "c"

    return ss_types


class AddGlobalIsNonLoopyFeature(Transform):
    """Add feature indicating whether the global loop content in the non-motif region is below 30%.

    For this initial implementation, only three categories will be one-hot encoded: helix, sheet, and loop.
    """

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["is_motif_token"])

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array = data["atom_array"]

        # Compute all ground-truth secondary structure types for the binder chain.
        # For now boolean, later could include distances as in RFD. But maybe that's better as a 2D condition
        gt_secondary_structures = get_secondary_structure_types(atom_array)
        atom_array.set_annotation("is_loop_gt", gt_secondary_structures[:, 2])

        is_motif_atom = atom_array.is_motif_token
        is_non_loopy = atom_array.is_loop_gt[~is_motif_atom].mean() < 0.3
        is_non_loopy_annot = np.full(
            atom_array.array_length(), 1 if is_non_loopy else -1, dtype=int
        )

        atom_array.set_annotation("is_non_loopy", is_non_loopy_annot)

        # HACK: Enables adding as atom-level features as well
        atom_array.set_annotation("is_non_loopy_atom_level", is_non_loopy_annot)

        return data


class Add1DSSFeature(Transform):
    """Add secondary structure features to training examples.

    For this initial implementation, only three categories will be one-hot encoded: helix, sheet, and loop.
    """

    def __init__(
        self,
        max_secondary_structure_frac_to_provide: float = 0.4,
        min_ss_island_len: int = 1,
        max_ss_island_len: int = 10,  # Might want to expand later, this is only average. Done for now to avoid over-conditioning.
        n_islands_max: int = 3,
    ):
        self.max_secondary_structure_frac_to_provide = (
            max_secondary_structure_frac_to_provide
        )
        self.min_ss_island_len = min_ss_island_len
        self.max_ss_island_len = max_ss_island_len
        self.n_islands_max = n_islands_max

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["is_motif_token"])

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array = data["atom_array"]

        # Compute all ground-truth secondary structure types for the binder chain.
        gt_secondary_structures = get_secondary_structure_types(atom_array)
        atom_array.set_annotation("is_helix_gt", gt_secondary_structures[:, 0])
        atom_array.set_annotation("is_sheet_gt", gt_secondary_structures[:, 1])
        atom_array.set_annotation("is_loop_gt", gt_secondary_structures[:, 2])

        if not data["conditions"]["add_1d_ss_features"]:
            return data

        # Always add the secondary structure type annotation, even if all zeros
        atom_array.set_annotation(
            "is_helix_conditioning", np.zeros(atom_array.array_length(), dtype=bool)
        )
        atom_array.set_annotation(
            "is_sheet_conditioning", np.zeros(atom_array.array_length(), dtype=bool)
        )
        atom_array.set_annotation(
            "is_loop_conditioning", np.zeros(atom_array.array_length(), dtype=bool)
        )

        # # Uniformly sample a number of tokens to receive secondary structure conditioning, up to the given maximum fraction
        max_residues_with_ss_conditioning = int(
            np.ceil(
                gt_secondary_structures.sum()
                * self.max_secondary_structure_frac_to_provide
            )
        )

        # Compute islands within the subset that is diffused and has secondary structure types.
        token_level_array = atom_array[get_token_starts(atom_array)]
        is_motif_token = token_level_array.is_motif_token
        eligible_for_ss_info_mask = (
            ~is_motif_token
            & token_level_array.is_protein
            & (  # Protein atoms with NaN coordinates would have no secondary structure annotation
                token_level_array.is_helix_gt
                | token_level_array.is_sheet_gt
                | token_level_array.is_loop_gt
            )
        )
        where_to_show_ss = sample_island_tokens(
            eligible_for_ss_info_mask.sum(),
            island_len_min=self.min_ss_island_len,
            island_len_max=self.max_ss_island_len,
            n_islands_min=1,
            n_islands_max=self.n_islands_max,
            max_length=max_residues_with_ss_conditioning,
        )

        # Convert this to a mask over the entire token-level atom array
        token_level_ss_mask = np.zeros(token_level_array.array_length(), dtype=bool)
        token_level_ss_mask[eligible_for_ss_info_mask] = where_to_show_ss
        ss_mask = spread_token_wise(atom_array, token_level_ss_mask)

        # Add the secondary structure type annotation
        atom_array.is_helix_conditioning[ss_mask] = atom_array.is_helix_gt[ss_mask]
        atom_array.is_sheet_conditioning[ss_mask] = atom_array.is_sheet_gt[ss_mask]
        atom_array.is_loop_conditioning[ss_mask] = atom_array.is_loop_gt[ss_mask]

        return data


class AddPPIHotspotFeature(Transform):
    """Add hotspot features to PPI training examples."""

    def __init__(
        self,
        max_hotspots_frac_to_provide: float = 0.2,
        hotspot_max_distance: float = 7.0,
    ):
        """
        Args:
            max_hotspots_frac_to_provide (int): Maximum fraction of ground-truth hotspots to add to the training example.
                The actual number added will be sampled uniformly from 0 to the number dictated by this parameter.
            hotspot_min_distance (float): Maximum distance to the binder for an atom to be considered a hotspot.
        """
        self.max_hotspots_frac_to_provide = max_hotspots_frac_to_provide
        self.hotspot_max_distance = hotspot_max_distance

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["is_motif_token"])

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array = data["atom_array"]

        # Always add the hotspot annotation, even if all zeros
        atom_array.set_annotation(
            "is_atom_level_hotspot", np.zeros(atom_array.array_length(), dtype=bool)
        )

        # Compute all ground-truth hotspots for the binder chain.
        # For now boolean, later could include distances as in RFD. But maybe that's better as a 2D condition
        is_hotspot_atom_mask = get_hotspot_atoms(
            atom_array,
            binder_pn_unit_iid=data["binder_pn_unit"],
            distance_cutoff=self.hotspot_max_distance,
        )
        atom_array.set_annotation("is_hotspot_gt", is_hotspot_atom_mask)

        # Uniformly sample a number of hotspots to include, up to the given maximum fraction
        max_hotspots_to_keep = int(
            np.ceil(sum(is_hotspot_atom_mask) * self.max_hotspots_frac_to_provide)
        )
        if max_hotspots_to_keep == 0:
            ranked_logger.warning("No hotspots found in the input data")
            return data
        else:
            num_hotspots_to_keep = np.random.randint(
                0,
                int(
                    np.ceil(
                        sum(is_hotspot_atom_mask) * self.max_hotspots_frac_to_provide
                    )
                ),
            )

        # Subsample hotspots to add.
        # For now random, later could add speckle_or_region from RFD
        true_hotspot_indices = np.where(is_hotspot_atom_mask)[0]
        hotspots_to_provide = np.random.choice(
            true_hotspot_indices, size=num_hotspots_to_keep, replace=False
        )
        atom_array.is_atom_level_hotspot[hotspots_to_provide] = True

        return data


class PPIFullBinderCropSpatial(Transform):
    """Crop which keeps the entire binder chain, then crops spatially around the given interface.
    Args:
        crop_size (int): The maximum number of tokens to crop. Must be greater than 0.
        jitter_scale (float, optional): The scale of the jitter to apply to the crop center.
            This is to break ties between atoms with the same spatial distance. Defaults to 1e-3.
        crop_center_cutoff_distance (float, optional): The cutoff distance to consider for
            selecting crop centers. Measured in Angstroms. Defaults to 15.0.
        keep_uncropped_atom_array (bool, optional): Whether to keep the uncropped atom array in the data.
            If `True`, the uncropped atom array will be stored in the `crop_info` dictionary
            under the key `"atom_array"`. Defaults to `False`.
        force_crop (bool, optional): Whether to force crop even if the atom array is already small enough.
            Defaults to `False`.
        max_atoms_in_crop (int, optional): Maximum number of atoms allowed in a crop. If None, no resizing is performed.
            Defaults to None.
    """

    def __init__(
        self,
        crop_size: int,
        jitter_scale: float = 1e-3,
        crop_center_cutoff_distance: float = 15.0,
        keep_uncropped_atom_array: bool = False,
        force_crop: bool = False,
        max_atoms_in_crop: int | None = None,
    ):
        self.crop_size = crop_size
        self.jitter_scale = jitter_scale
        self.crop_center_cutoff_distance = crop_center_cutoff_distance
        self.keep_uncropped_atom_array = keep_uncropped_atom_array
        self.force_crop = force_crop
        self.max_atoms_in_crop = max_atoms_in_crop

    def check_input(self, data: dict):
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["pn_unit_iid", "atomize", "atom_id"])

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]

        if "query_pn_unit_iids" in data and data["query_pn_unit_iids"]:
            query_pn_units = data["query_pn_unit_iids"]
        else:
            query_pn_units = np.unique(atom_array.pn_unit_iid)
            ranked_logger.info(
                f"No query PN unit(s) provided for spatial crop. Randomly selecting from {query_pn_units}."
            )

        if "binder_pn_unit" not in data:
            raise ValueError("Data dict must contain 'binder_pn_unit' key.")

        crop_info = crop_spatial_keep_full_binder(
            atom_array=atom_array,
            query_pn_unit_iids=query_pn_units,
            binder_pn_unit_iid=data["binder_pn_unit"],
            crop_size=self.crop_size,
            jitter_scale=self.jitter_scale,
            crop_center_cutoff_distance=self.crop_center_cutoff_distance,
            force_crop=self.force_crop,
        )
        crop_info = resize_crop_info_if_too_many_atoms(
            crop_info=crop_info,
            atom_array=atom_array,
            max_atoms=self.max_atoms_in_crop,
        )

        data["crop_info"] = {"type": self.__class__.__name__} | crop_info

        if self.keep_uncropped_atom_array:
            data["crop_info"]["atom_array"] = atom_array

        # Update data with cropped atom array
        data["atom_array"] = atom_array[crop_info["crop_atom_idxs"]]

        return data


def crop_spatial_keep_full_binder(
    atom_array: AtomArray,
    query_pn_unit_iids: list[str],
    binder_pn_unit_iid: str,
    crop_size: int,
    jitter_scale: float = 1e-3,
    crop_center_cutoff_distance: float = 15.0,
    force_crop: bool = False,
) -> dict:
    """
    Crop spatial tokens around a given `crop_center` by keeping the entire binder chain, then taking nearest
    neighbors (with jitter) until reaching the `crop_size`.

    Args:
        - atom_array (AtomArray): The atom array to crop.
        - query_pn_unit_iids (list[str]): List of query polymer/non-polymer unit instance IDs.
        - binder_pn_unit_iid (str): The polymer/non-polymer unit instance ID corresponding to the binder.
        - crop_size (int): The maximum number of tokens to crop.
        - jitter_scale (float, optional): Scale of jitter to apply when calculating distances.
            Defaults to 1e-3.
        - crop_center_cutoff_distance (float, optional): Maximum distance from query units to
            consider for crop center. Defaults to 15.0 Angstroms.
        - force_crop (bool, optional): Whether to force crop even if the atom array is already small enough.
            Defaults to False.

    Returns:
        dict: A dictionary containing crop information, including:
            - requires_crop (bool): Whether cropping was necessary.
            - crop_center_atom_id (int or np.nan): ID of the atom chosen as crop center.
            - crop_center_atom_idx (int or np.nan): Index of the atom chosen as crop center.
            - crop_center_token_idx (int or np.nan): Index of the token containing the crop center.
            - crop_token_idxs (np.ndarray): Indices of tokens included in the crop.
            - crop_atom_idxs (np.ndarray): Indices of atoms included in the crop.

    Note:
        This function implements the spatial cropping procedure as described in AlphaFold 3 and AlphaFold 2 Multimer.

    References:
        - AF3 https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf
        - AF2 Multimer https://www.biorxiv.org/content/10.1101/2021.10.04.463034v2.full.pdf
    """
    if binder_pn_unit_iid not in query_pn_unit_iids:
        raise ValueError(
            f"Binder polymer/non-polymer unit instance ID '{binder_pn_unit_iid}' "
            f"not found in query polymer/non-polymer unit instance IDs: {query_pn_unit_iids}"
        )
    n_tokens = get_token_count(atom_array)
    requires_crop = n_tokens > crop_size

    # ... get binder information
    binder_token_mask = (
        atom_array[get_af3_token_center_masks(atom_array)].pn_unit_iid
        == binder_pn_unit_iid
    )
    binder_atom_mask = atom_array.pn_unit_iid == binder_pn_unit_iid
    n_binder_tokens = get_token_count(atom_array[binder_atom_mask])

    if force_crop or requires_crop:
        # Get possible crop centers
        can_be_crop_center = get_spatial_crop_center(
            atom_array, query_pn_unit_iids, crop_center_cutoff_distance
        )

        # ... sample crop center atom
        crop_center_atom_id = np.random.choice(atom_array[can_be_crop_center].atom_id)
        crop_center_atom_idx = atom_id_to_atom_idx(atom_array, crop_center_atom_id)

        # ... sample crop, excluding the binder polymer/non-polymer unit
        token_coords = get_af3_token_center_coords(atom_array)
        crop_center_token_idx = atom_id_to_token_idx(atom_array, crop_center_atom_id)
        is_token_in_crop = get_spatial_crop_excluding_mask(
            token_coords,
            crop_center_token_idx,
            crop_size=crop_size
            - n_binder_tokens,  # reserve space for the binder tokens
            mask_to_exclude=binder_token_mask,
            jitter_scale=jitter_scale,
        )
        # ... spread token-level crop mask to atom-level
        is_atom_in_crop = spread_token_wise(atom_array, is_token_in_crop)

        # ... add in binder tokens and atoms
        is_token_in_crop = is_token_in_crop | binder_token_mask
        is_atom_in_crop = is_atom_in_crop | binder_atom_mask
    else:
        # ... no need to crop since the atom array is already small enough
        crop_center_atom_id = np.nan
        crop_center_atom_idx = np.nan
        crop_center_token_idx = np.nan
        is_atom_in_crop = np.ones(len(atom_array), dtype=bool)
        is_token_in_crop = np.ones(n_tokens, dtype=bool)

    return {
        "requires_crop": requires_crop,  # whether cropping was necessary
        "crop_center_atom_id": crop_center_atom_id,  # atom_id of crop center
        "crop_center_atom_idx": crop_center_atom_idx,  # atom_idx of crop center
        "crop_center_token_idx": crop_center_token_idx,  # token_idx of crop center
        "crop_token_idxs": np.where(is_token_in_crop)[0],  # token_idxs in crop
        "crop_atom_idxs": np.where(is_atom_in_crop)[0],  # atom_idxs in crop
    }


def get_spatial_crop_excluding_mask(
    coord: np.ndarray,
    crop_center_idx: int,
    crop_size: int,
    mask_to_exclude: np.ndarray,
    jitter_scale: float = 1e-3,
) -> np.ndarray:
    """
    Crop spatial tokens around a given `crop_center`, keeping nearest neighbors (with jitter) and excluding atoms in a
    specified mask, until reaching the `crop_size`.

    Args:
        coord (np.ndarray): A 2D numpy array of shape (N, 3) representing the 3D token-level coordinates.
            Coordinates are expected to be in Angstroms.
        crop_center_idx (int): The index of the token to be used as the center of the crop.
        crop_size (int): The number of nearest neighbors to include in the crop.
        mask_to_exclude (siwnp.ndarray): A mask indicating atoms to be excluded from the crop.
        jitter_scale (float): The scale of the jitter to add to the coordinates.

    Returns:
        crop_mask (np.ndarray): A boolean mask of shape (N,) where True indicates that the token is within the crop.

    """
    assert_that(coord.ndim).is_equal_to(2)
    assert_that(coord.shape[1]).is_equal_to(3)
    assert_that(crop_center_idx).is_less_than(coord.shape[0])
    assert_that(crop_size).is_greater_than(0)
    assert_that(jitter_scale).is_greater_than_or_equal_to(0)

    # Add small jitter to coordinates to break ties
    if jitter_scale > 0:
        coord = coord + np.random.normal(scale=jitter_scale, size=coord.shape)

    # ... get query center
    query_center = coord[crop_center_idx]

    # ... extract a mask for valid coordinates (i.e. no `nan`'s, which indicate unknown token centers)
    #     including including unoccupied tokens in the crop
    is_valid = np.isfinite(coord).all(axis=1)

    # ... exclude the specified pn_unit
    is_valid = is_valid & ~mask_to_exclude

    # ... build a KDTree for efficient querying, excluding invalid coordinates
    tree = KDTree(coord[is_valid])

    # ... query the `crop_size` nearest neighbors of the crop center
    _, nearest_neighbor_idxs = tree.query(query_center, k=crop_size, p=2)
    # ... filter out missing neighbours (index equal to `tree.n`)
    nearest_neighbor_idxs = nearest_neighbor_idxs[nearest_neighbor_idxs < tree.n]

    # ... crop mask is True for the `crop_size` nearest neighbors of the crop center
    crop_mask = np.zeros(coord.shape[0], dtype=bool)
    is_valid_and_in_crop_idxs = np.where(is_valid)[0][nearest_neighbor_idxs]
    crop_mask[is_valid_and_in_crop_idxs] = True

    return crop_mask
