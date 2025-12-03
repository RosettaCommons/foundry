import numpy as np
from rfd3.inference.symmetry.frames import (
    decompose_symmetry_frame,
    get_symmetry_frames_from_symmetry_id,
)

from foundry.utils.ddp import RankedLogger

FIXED_TRANSFORM_ID = -1
FIXED_ENTITY_ID = -1
ranked_logger = RankedLogger(__name__, rank_zero_only=True)


########################################################
# Symmetry annotations
########################################################


def add_sym_annotations(atom_array, sym_conf):
    """
    Add symmetry base annotations to an atom array.
    Arguments:
        atom_array: atom array of symmetry subunit
        sym_conf: symmetry configuration (dict, "id" key is required)
    """
    n = atom_array.shape[0]
    # which is the asymmetric unit? At this point, we annotate everything as the asu
    is_asu = np.full(n, True, dtype=np.bool_)
    atom_array.set_annotation("is_sym_asu", is_asu)
    # symmetry_id
    symmetry_ids = np.full(n, sym_conf.get("id"), dtype="U6")
    atom_array.set_annotation("symmetry_id", symmetry_ids)
    return atom_array


def add_sym_annotations_to_fixed_motif(atom_array):
    """
    Add symmetry annotations to a motif atom array.
    Arguments:
        atom_array: atom array of symmetry subunit
    """
    n = atom_array.shape[0]

    # setting the identity transform
    Ori, X, Y = decompose_symmetry_frame((np.eye(3), np.zeros(3)))
    Oris = np.full(n, Ori)
    Xs = np.full(n, X)
    Ys = np.full(n, Y)
    atom_array.set_annotation("sym_transform_Ori", Oris)
    atom_array.set_annotation("sym_transform_X", Xs)
    atom_array.set_annotation("sym_transform_Y", Ys)

    transform_ids = np.full(n, FIXED_TRANSFORM_ID, dtype=np.int32)
    atom_array.set_annotation("sym_transform_id", transform_ids)
    entity_ids = np.full(n, FIXED_ENTITY_ID, dtype=np.int32)
    atom_array.set_annotation("sym_entity_id", entity_ids)
    # make sure that the motif is not the asu
    is_sym_asu = np.full(n, False, dtype=np.bool_)
    atom_array.set_annotation("is_sym_asu", is_sym_asu)
    return atom_array


def add_src_sym_component_annotations(atom_array):
    """
    Add src_sym_component annotations to an atom array.
    This is used to correctly map the original motif id to diffused unindexed motifs.
    Arguments:
        atom_array: atom array with src_component annotated
    """
    if "src_component" not in atom_array.get_annotation_categories():
        return atom_array

    src_sym_component = atom_array.src_component.copy()
    src_tokens = np.unique(atom_array.src_component)

    for src_token in src_tokens:
        # Skip non-alphabetic tokens
        if len(src_token) == 0:
            continue
        if not src_token[0].isalpha():
            continue

        # Get block of atoms with this src token
        src_block_mask = atom_array.src_component == src_token
        src_block = atom_array[src_block_mask]

        # Skip if not all unindexed motif atoms
        if not src_block.is_motif_atom_unindexed.all():
            continue

        # Update src component with chain ID prefix
        for chain_id in np.unique(src_block.chain_id):
            chain_mask = src_block.chain_id == chain_id
            src_block.src_component[chain_mask] = chain_id + src_token[1:]

        src_sym_component[src_block_mask] = src_block.src_component

    atom_array.set_annotation("src_sym_component", src_sym_component)
    return atom_array


def fix_3D_sym_motif_annotations(atom_array):
    """
    Add fixed motif annotations to the 3D NON-indexed motifs (only unindexed and ligands).
    since indexed motifs are contiguously connected to generative residues,
    it should NOT be fixed, instead get symmetrized at each step
    Arguments:
        atom_array: atom array
    """
    # fixed_motif_mask = atom_array.is_motif_atom_with_fixed_coord == 1
    fixed_motif_mask = atom_array._is_motif & ~atom_array._is_indexed_motif
    fixed_motif_array = atom_array[fixed_motif_mask].copy()
    fixed_motif_array = add_sym_annotations_to_fixed_motif(fixed_motif_array)
    atom_array[fixed_motif_mask] = fixed_motif_array
    return atom_array


def add_sym_transform_annotations(atom_array, transform_id, frame, is_asu=False):
    """
    Add symmetry annotations to an atom array.
    Arguments:
        atom_array: atom array of symmetry subunit
        transform_id: index of the transform frame
        is_asu: whether this is the asymmetric unit
    Returns:
        atom_array: atom array with symmetry annotations
    """
    Ori, X, Y = decompose_symmetry_frame(frame)
    n = atom_array.shape[0]

    # symmetry transform (decomposed into Ori, X, Y)
    Oris = np.full(n, Ori)
    Xs = np.full(n, X)
    Ys = np.full(n, Y)
    atom_array.set_annotation("sym_transform_Ori", Oris)
    atom_array.set_annotation("sym_transform_X", Xs)
    atom_array.set_annotation("sym_transform_Y", Ys)

    # symmetry transform id
    transform_ids = np.full(n, transform_id, dtype=np.int32)
    atom_array.set_annotation("sym_transform_id", transform_ids)

    # entity ids - this will help keep track of different multiplicities
    # if there are sm, they will have different entity ids from the prot atoms
    unique_chain_ids = np.unique(atom_array.chain_id).tolist()
    unique_chain_ids.sort()
    entity_ids = np.array([unique_chain_ids.index(id) for id in atom_array.chain_id])
    atom_array.set_annotation("sym_entity_id", entity_ids)

    is_sym_asu = np.full(n, is_asu, dtype=np.bool_)
    atom_array.set_annotation("is_sym_asu", is_sym_asu)

    return atom_array


def apply_symmetry_to_atomarray_coord(atom_array, frame):
    """
    Apply symmetry to the atom array coordinates.
    Arguments:
        atom_array: atom array
        frame: symmetry frame (R, T)
    """
    R, T = frame
    atom_array.coord = atom_array.coord @ R.T
    atom_array.coord += T  # T should be 0 for most symmetry cases
    return atom_array


########################################################
# Motif functions
########################################################


def annotate_unsym_atom_array(atom_array):
    """
    Annotate the unsym motif and return it.
    Arguments:
        atom_array: atom array
        unsym_motif_mask: mask of unsym motifs
    """
    unsym_atom_array = atom_array.copy()
    unsym_atom_array._is_asu = np.full(unsym_atom_array.shape[0], False, dtype=np.bool_)
    unsym_atom_array.is_sym_asu = unsym_atom_array._is_asu
    unsym_atom_array = reset_chain_ids(
        unsym_atom_array, start_id="a"
    )  # give it a lowercase chain id to avoid confusion with symmetry units
    unsym_atom_array = add_sym_annotations_to_fixed_motif(unsym_atom_array)
    return unsym_atom_array


########################################################
# 2D conditioning functions
########################################################


def add_2d_entity_annotations(atom_array):
    entity_ids = np.zeros(atom_array.shape[0], dtype=np.int32)
    categories = get_2d_annotation_categories(atom_array)
    entity_id = 1
    for i, anno in enumerate(categories):
        entity_id = i + 1
        entity_ids[atom_array.get_annotation(anno) == 1] = entity_id
    atom_array.set_annotation("_2d_entity_id", entity_ids)
    return atom_array


def reannotate_2d_entity_ids(atom_array, transform_id):
    if "_2d_entity_id" not in atom_array.get_annotation_categories():
        return atom_array
    _2d_annos = get_2d_annotation_categories(atom_array)
    frames = get_symmetry_frames_from_symmetry_id(atom_array.symmetry_id[0])
    # NOTE: assuming its either 2d cond is within a subunit was specified or all active sites were explicity specified
    max_entity_id = max(len(_2d_annos), len(frames))
    mask = atom_array.get_annotation("_2d_entity_id") != 0
    atom_array._2d_entity_id[mask] = (
        (atom_array._2d_entity_id[mask] + transform_id - 1) % max_entity_id
    ) + 1
    return atom_array


def get_2d_annotation_categories(atom_array):
    categories = []
    for anno in atom_array.get_annotation_categories():
        if "2d_condition" in anno:
            categories.append(anno)
    categories.sort()  # sort to make sure that categories are in ascending order
    return categories


def reannotate_2d_conditions(atom_array):
    entity_ids_anno = atom_array.get_annotation("_2d_entity_id")
    entity_ids = [d for d in np.unique(entity_ids_anno) if d != 0]
    categories = get_2d_annotation_categories(atom_array)
    diff = len(entity_ids) - len(categories)
    if diff > 0:
        for i in range(len(categories), len(categories) + diff):
            categories.append(f"{categories[0]}_{i}")
    for d, anno in zip(entity_ids, categories):
        atom_array.set_annotation(anno, entity_ids_anno == d)
    atom_array.del_annotation("_2d_entity_id")
    return atom_array


########################################################
# Utility functions
########################################################


def reset_chain_ids(atom_array, start_id):
    """
    Reset the chain ids and pn_unit_iids of an atom array to start from the given id.
    Arguments:
        atom_array: atom array with chain_ids and pn_unit_iids annotated
    """
    chain_ids = np.unique(atom_array.chain_id)
    new_chain_range = range(ord(start_id), ord(start_id) + len(chain_ids))
    for new_id, old_id in zip(new_chain_range, chain_ids):
        atom_array.chain_id[atom_array.chain_id == old_id] = chr(new_id)
    atom_array.pn_unit_iid = atom_array.chain_id
    return atom_array


def reannotate_chain_ids(atom_array, offset, multiplier=0):
    """
    Reannotate the chain ids and pn_unit_iids of an atom array.
    Arguments:
        atom_array: protein atom array with chain_ids and pn_unit_iids annotated
        offset: offset to add to the chain ids
        multiplier: multiplier to add to the chain ids
    """
    chain_ids_int = (
        np.array([ord(c) for c in atom_array.chain_id]) + offset * multiplier
    )
    chain_ids = np.array([chr(id) for id in chain_ids_int], dtype=str)
    atom_array.chain_id = chain_ids
    atom_array.pn_unit_iid = chain_ids
    return atom_array


def get_symmetry_unit(asu_atom_array, transform_id, frame):
    """
    Annotate the ASU protein atom array and return it for each symmetry unit.
    Arguments:
        asu_atom_array: atom array of the asymmetric unit, annotated with symmetry_id
        transform_id: index of the symmetry unit
        frame: symmetry frame
    """
    num_prot_chains = len(np.unique(asu_atom_array.chain_id))
    symmetry_unit = asu_atom_array.copy()
    symmetry_unit = reannotate_chain_ids(symmetry_unit, num_prot_chains, transform_id)
    symmetry_unit = reannotate_2d_entity_ids(symmetry_unit, transform_id)
    symmetry_unit = add_sym_transform_annotations(
        symmetry_unit, transform_id, frame, is_asu=(transform_id == 0)
    )
    # apply symmetry to indexed motifs
    # at this point, the diffused coordinates are at the origin/ have no xyz
    symmetry_unit = apply_symmetry_to_atomarray_coord(symmetry_unit, frame)
    return symmetry_unit
