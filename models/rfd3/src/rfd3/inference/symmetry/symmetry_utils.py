from typing import Optional

import biotite.structure as struc
import numpy as np
import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)
from rfd3.inference.symmetry.atom_array import (
    FIXED_ENTITY_ID,
    FIXED_TRANSFORM_ID,
    add_2d_entity_annotations,
    add_src_sym_component_annotations,
    add_sym_annotations,
    annotate_unsym_atom_array,
    fix_3D_sym_motif_annotations,
    get_symmetry_unit,
    reannotate_2d_conditions,
)
from rfd3.inference.symmetry.checks import (
    check_symmetry_config,
)
from rfd3.inference.symmetry.contigs import (
    expand_contig_unsym_motif,
    get_unsym_motif_mask,
)
from rfd3.inference.symmetry.frames import (
    get_symmetry_frames_from_atom_array,
    get_symmetry_frames_from_symmetry_id,
)
from rfd3.transforms.conditioning_base import get_motif_features

from foundry.utils.components import fetch_mask_from_component
from foundry.utils.ddp import RankedLogger

ranked_logger = RankedLogger(__name__, rank_zero_only=True)


class SymmetryConfig(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
    )
    id: Optional[str] = Field(
        None,
        description="Symmetry group ID. e.g. 'C3', 'D2'. Only C and D symmetry types are supported currently.",
    )
    is_unsym_motif: Optional[str] = Field(
        None,
        description="Comma separated list of contig/ligand names that should not be symmetrized such as DNA strands. \
         e.g. 'HEM' or 'Y1-11,Z16-25'",
    )
    is_symmetric_motif: bool = Field(
        True,
        description="If True, the input motifs are expected to be already symmetric and won't be symmetrized. \
        If False, the all input motifs are expected to be ASU and will be symmetrized.",
    )


def convery_sym_conf_to_symmetry_config(sym_conf: dict):
    return SymmetryConfig(**sym_conf)


def make_symmetric_atom_array(
    asu_atom_array,
    sym_conf: SymmetryConfig | dict,
    sm=None,
    has_dist_cond=False,
    src_atom_array=None,
):
    """
    apply symmetry to an atom array.
    Arguments:
        asu_atom_array: atom array of the asymmetric unit
        sym_conf: symmetry configuration (dict, "id" key is required)
        sm: optional small molecule names (str, comma separated)
        has_dist_cond: whether to add 2d entity annotations
    Returns:
        new_asu_atom_array: atom array with symmetry applied
    """
    if not isinstance(sym_conf, SymmetryConfig):
        sym_conf = convery_sym_conf_to_symmetry_config(sym_conf)

    check_symmetry_config(
        asu_atom_array, sym_conf, sm, has_dist_cond, src_atom_array=src_atom_array
    )
    # Adding utility annotations to the asu atom array
    asu_atom_array = _add_util_annotations(asu_atom_array, sym_conf, sm)

    if has_dist_cond:  # NB: this will only work for asymmetric motifs at the moment - need to add functionality for symmetric motifs
        asu_atom_array = add_2d_entity_annotations(asu_atom_array)

    frames = get_symmetry_frames_from_symmetry_id(sym_conf)

    # If the motif is symmetric, we get the frames instead from the source atom array.
    if sym_conf.is_symmetric_motif:
        assert (
            src_atom_array is not None
        ), "Source atom array must be provided for symmetric motifs"
        # if symmetric motif is provided, get the frames from the src atom array.
        frames = get_symmetry_frames_from_atom_array(src_atom_array, frames)
    elif (asu_atom_array._is_motif[~asu_atom_array._is_unsym_motif]).any():
        # if the motifs that's not unsym motifs are present.
        raise NotImplementedError(
            "Asymmetric motif inputs are not implemented yet. please symmetrize the motif."
        )

    # Add symmetry annotations to the asu atom array
    asu_atom_array = add_sym_annotations(asu_atom_array, sym_conf)

    # Extracting all things at this moment that we will not want to symmetrize.
    # This includes: 1) unsym motifs, 2) ligands
    unsym_atom_arrays = []
    if sym_conf.is_unsym_motif:
        # unsym_motif_atom_array = get_unsym_motif(asu_atom_array, asu_atom_array._is_unsym_motif)
        # Now remove the unsym motifs from the asu atom array
        unsym_atom_arrays.append(asu_atom_array[asu_atom_array._is_unsym_motif])
        asu_atom_array = asu_atom_array[~asu_atom_array._is_unsym_motif]
    if sm:
        unsym_atom_arrays.append(asu_atom_array[asu_atom_array._is_sm])
        asu_atom_array = asu_atom_array[~asu_atom_array._is_sm]
    unsym_atom_array = (
        struc.concatenate(unsym_atom_arrays) if len(unsym_atom_arrays) > 0 else None
    )

    # Annotate symmetric subunits
    symmetry_unit_list = []
    for transform_id, frame in enumerate(frames):
        # this is to build the fully symmetrized atom array containing all the symmetry subunits
        symmetry_unit = get_symmetry_unit(asu_atom_array, transform_id, frame)
        symmetry_unit_list.append(symmetry_unit)
    if unsym_atom_array:  # only if exists
        unsym_atom_array = annotate_unsym_atom_array(unsym_atom_array)
        symmetry_unit_list.append(
            unsym_atom_array
        )  # add the motifs to the end of the asu atom array list (motifs at end of atom array)
    # build the full symmetrized atom array
    symmetrized_atom_array = struc.concatenate(symmetry_unit_list)

    # add 2D conditioning annotations
    if has_dist_cond:
        symmetrized_atom_array = reannotate_2d_conditions(symmetrized_atom_array)

    # set all motifs to not have any symmetrization applied to them
    # TODO: this needs to be adapted to work with 2D cond (in 2D cond, we WANT to apply symmetry to the motifs since they move in space)
    symmetrized_atom_array = fix_3D_sym_motif_annotations(symmetrized_atom_array)

    # This is needed to output correct motif residue mappings in the output json
    symmetrized_atom_array = add_src_sym_component_annotations(symmetrized_atom_array)
    # remove utility annotations
    symmetrized_atom_array = _del_util_annotations(symmetrized_atom_array)
    return symmetrized_atom_array


def make_symmetric_atom_array_for_partial_diffusion(atom_array, sym_conf):
    """
    Apply symmetry to an atom array with partial diffusion.
    Arguments:
        atom_array: atom array of the asymmetric unit
        sym_conf: symmetry configuration (dict, "id" key is required)
    Returns:
        atom_array: atom array with symmetry applied
    """
    # TODO: clean up this function

    # For partial diffusion with symmetric inputs, preserve exact positioning
    ranked_logger.info(
        "Partial diffusion with symmetry - preserving exact input coordinates"
    )
    ranked_logger.info("SKIPPING symmetry reconstruction to preserve input structure")
    # Add full symmetry annotations without changing coordinates
    from rfd3.inference.symmetry.checks import (
        check_atom_array_is_symmetric,
    )
    from rfd3.inference.symmetry.frames import (
        decompose_symmetry_frame,
    )

    check_symmetry_config(
        atom_array,
        sym_conf,
        sm=None,
        has_dist_cond=False,
        src_atom_array=None,
        partial=True,
    )

    atom_array = add_sym_annotations(atom_array, sym_conf)
    assert check_atom_array_is_symmetric(atom_array), "Atom array is not symmetric"

    n = atom_array.shape[0]
    chain_ids = np.unique(atom_array.chain_id)
    frames = get_symmetry_frames_from_symmetry_id(sym_conf)

    # Add symmetry ID
    symmetry_ids = np.full(n, sym_conf.id, dtype="U6")
    atom_array.set_annotation("symmetry_id", symmetry_ids)

    # Initialize transform annotations (use same format as original system)
    symmetry_transform_id = np.zeros(n, dtype=np.int32)
    symmetry_entity_id = np.zeros(n, dtype=np.int32)
    is_asu = np.zeros(n, dtype=bool)

    # Add transform annotations for each chain (same format as add_symmetry_transform_annotations)
    for i, chain_id in enumerate(chain_ids):
        chain_mask = atom_array.chain_id == chain_id
        transform_id = i % len(frames)  # Cycle through available frames
        frame = frames[transform_id]

        # Decompose frame to packed scalars
        Ori, X, Y = decompose_symmetry_frame(frame)

        # Set annotations for this chain (use np.full like original system)
        if i == 0:  # First chain - initialize arrays
            sym_transform_Ori = np.full(n, Ori)
            sym_transform_X = np.full(n, X)
            sym_transform_Y = np.full(n, Y)
            is_asu[chain_mask] = True
        else:  # Subsequent chains - update specific atoms
            sym_transform_Ori[chain_mask] = Ori
            sym_transform_X[chain_mask] = X
            sym_transform_Y[chain_mask] = Y

        symmetry_transform_id[chain_mask] = transform_id
        symmetry_entity_id[chain_mask] = 0  # All chains same entity for C9

    # Set all annotations
    atom_array.set_annotation("sym_transform_Ori", sym_transform_Ori)
    atom_array.set_annotation("sym_transform_X", sym_transform_X)
    atom_array.set_annotation("sym_transform_Y", sym_transform_Y)
    atom_array.set_annotation("sym_transform_id", symmetry_transform_id)
    atom_array.set_annotation("sym_entity_id", symmetry_entity_id)
    atom_array.set_annotation("is_sym_asu", is_asu)

    ranked_logger.info(
        f"Added full symmetry annotations to {len(chain_ids)} existing chains WITHOUT changing coordinates"
    )

    return atom_array


########################################################
# Private functions only used in make_symmetric_atom_array
########################################################


def _add_util_annotations(asu_atom_array, sym_conf, sm):
    """
    Add symmetry-specific utility annotations to the asu atom array.
    Arguments:
        asu_atom_array: atom array of the asymmetric unit
        sym_conf: symmetry configuration
        sm: small molecule names (str, comma separated)
    """
    n = asu_atom_array.shape[0]
    is_motif = get_motif_features(asu_atom_array)["is_motif_atom"].astype(np.bool_)
    is_sm = np.zeros(n, dtype=bool)
    is_asu = np.ones(n, dtype=bool)
    is_unsym_motif = np.zeros(n, dtype=bool)

    if sm:
        is_sm = np.logical_or.reduce(
            [
                fetch_mask_from_component(lig, atom_array=asu_atom_array)
                for lig in sm.split(",")
            ]
        )

    # assign unsym motifs
    if sym_conf.is_unsym_motif:
        unsym_motif_names = sym_conf.is_unsym_motif.split(",")
        unsym_motif_names = expand_contig_unsym_motif(unsym_motif_names)
        is_unsym_motif = get_unsym_motif_mask(asu_atom_array, unsym_motif_names)

    is_unindexed_motif = asu_atom_array.is_motif_atom_unindexed.astype(np.bool_)
    is_indexed_motif = ~is_sm & ~is_unindexed_motif & is_motif

    asu_atom_array.set_annotation(
        "_is_asu", is_asu
    )  # Currently not used but will needed for 2D cond
    asu_atom_array.set_annotation("_is_motif", is_motif)
    asu_atom_array.set_annotation("_is_sm", is_sm)
    asu_atom_array.set_annotation("_is_indexed_motif", is_indexed_motif)
    asu_atom_array.set_annotation("_is_unindexed_motif", is_unindexed_motif)
    asu_atom_array.set_annotation("_is_unsym_motif", is_unsym_motif)
    return asu_atom_array


def _del_util_annotations(aary):
    """
    Delete symmetry-specific utility annotations from the atom array.
    Arguments:
        aary: atom array
    """
    aary.del_annotation("_is_asu")  # Currently not used but will needed for 2D cond
    aary.del_annotation("_is_motif")
    aary.del_annotation("_is_sm")
    aary.del_annotation("_is_indexed_motif")
    aary.del_annotation("_is_unindexed_motif")
    aary.del_annotation("_is_unsym_motif")
    return aary


#########################
# Symmetrization functions
#########################


def center_symmetric_src_atom_array(src_atom_array):
    """
    Center the src atom array at the origin.
    Arguments:
        src_atom_array: atom array of the source
    Returns:
        src_atom_array: atom array of the source centered at the origin
    """
    # Compute COM of the src atom array (protein only elements)
    src_atom_array_com = np.mean(
        src_atom_array[src_atom_array.chain_type == 6].coord, axis=0
    )
    # center the src atom array
    src_atom_array.coord -= src_atom_array_com
    return src_atom_array


def apply_symmetry_to_xyz_atomwise(X_L, sym_feats, partial_diffusion=False):
    """
    Apply symmetry to the xyz coordinates.
    Arguments:
        X_L: [B, L, 3] xyz coordinates
        sym_feats: dictionary containing symmetry features (id, transform, entity_id, is_sym_asu)
    Returns:
        X_L: [B, L, 3] xyz coordinates with symmetry applied
    """
    sym_entity_id = sym_feats["sym_entity_id"]
    sym_transform_id = sym_feats["sym_transform_id"]
    is_sym_asu = sym_feats["is_sym_asu"]
    fixed_motif_mask = sym_entity_id == FIXED_ENTITY_ID
    sym_transforms = {
        int(k): v
        for k, v in sym_feats["sym_transform"].items()
        if int(k) != FIXED_TRANSFORM_ID
    }  # {str(id): tensor(3,3)} -> {int(id): tensor(3,3)}
    # COM correction (in case there is drift)
    if not partial_diffusion:
        X_L[:, ~fixed_motif_mask, :] = X_L[:, ~fixed_motif_mask, :] - X_L[
            :, ~fixed_motif_mask, :
        ].mean(dim=1, keepdim=True)
    sym_X_L = X_L.clone()

    # Loop through each symmetry entity id - making sure that we apply the matching symmetry transform to asu id
    unique_entity_id = torch.unique(sym_entity_id)
    unique_entity_id = unique_entity_id[unique_entity_id != FIXED_ENTITY_ID]
    for entity_id in unique_entity_id:
        # Mask for this entity id
        entity_id_mask = sym_entity_id == entity_id  # [L]
        # ASU that corresponds to this transform only
        entity_asu_mask = is_sym_asu & entity_id_mask
        if entity_asu_mask.sum() == 0:
            continue
        asu_xyz = X_L[:, entity_asu_mask, :]  # [B, Lasu, 3]
        # Transforms
        unique_transform_id = torch.unique(sym_transform_id[entity_id_mask]).tolist()
        for (
            target_id
        ) in unique_transform_id:  # Open to suggestions for making this more efficient
            # Get a mask that corresponds to this specific transform in the entire atom array
            this_subunit = entity_id_mask & (sym_transform_id == target_id)
            # Apply this subunit's symmetry transform to its corresponding ASU
            sym_X_L[:, this_subunit, :] = torch.einsum(
                "blc,cd->bld", asu_xyz, sym_transforms[target_id][0].to(asu_xyz.dtype)
            ) + sym_transforms[target_id][1].to(asu_xyz.dtype)

    return sym_X_L
