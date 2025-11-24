"""
Class-based motif masking system
"""

import logging
from abc import ABC, abstractmethod

import networkx as nx
import numpy as np
from atomworks.ml.utils.token import (
    apply_token_wise,
    get_token_starts,
    spread_token_wise,
)
from biotite.structure import AtomArray, get_residue_starts
from rfd3.transforms.conditioning_utils import (
    random_condition,
    sample_island_tokens,
    sample_subgraph_atoms,
)

nx.from_numpy_matrix = nx.from_numpy_array
logger = logging.getLogger(__name__)


#################################################################################
# Transform for creating training conditions
#################################################################################


class TrainingCondition(ABC):
    """
    Base class for applying conditioning during training
    """

    name = None

    def __init__(self, frequency):
        self.frequency = frequency

    @abstractmethod
    def is_valid_for_example(self, data) -> bool:
        """
        Returns true whether this mask can be applied to the data instance

        E.g. only use this transform if data metadata contains key or if data contains type
        """

    @abstractmethod
    def sample(self, data) -> AtomArray:
        """
        Set which atoms should be made into tokens
        """


class IslandCondition(TrainingCondition):
    """
    Select islands as motif and assign conditioning strategies.
    """

    def __init__(
        self,
        *,
        name,
        frequency,
        island_sampling_kwargs,
        p_diffuse_motif_sidechains,
        p_diffuse_subgraph_atoms,
        subgraph_sampling_kwargs,
        p_fix_motif_coordinates,
        p_fix_motif_sequence,
        p_unindex_motif_tokens,
    ):
        self.name = name
        self.frequency = frequency

        # Token selection
        self.island_sampling_kwargs = island_sampling_kwargs

        # Atom selection
        self.p_diffuse_motif_sidechains = p_diffuse_motif_sidechains
        self.p_include_oxygen_in_backbone_mask = 0.95
        self.p_diffuse_subgraph_atoms = p_diffuse_subgraph_atoms
        self.subgraph_sampling_kwargs = subgraph_sampling_kwargs

        # Additional conditioning selection
        self.p_fix_motif_coordinates = p_fix_motif_coordinates
        self.p_fix_motif_sequence = p_fix_motif_sequence
        self.p_unindex_motif_tokens = p_unindex_motif_tokens

    def is_valid_for_example(self, data) -> bool:
        is_protein = data["atom_array"].is_protein
        if not np.any(is_protein):
            return False
        return True

    def sample_motif_tokens(self, atom_array):
        """
        Samples what tokens should be considered motif.
        """
        token_level_array = atom_array[get_token_starts(atom_array)]

        # initialize motif tokens as all non-protein tokens
        is_motif_token = np.asarray(~token_level_array.is_protein, dtype=bool).copy()
        n_protein_tokens = np.sum(token_level_array.is_protein)
        islands_mask = sample_island_tokens(
            n_protein_tokens,
            **self.island_sampling_kwargs,
        )
        is_motif_token[token_level_array.is_protein] = islands_mask

        # TODO: Atoms with covalent bonds should be motif, needs FlagAndReassignCovalentModifications transform prior to this
        # atom_with_coval_bond = token_level_array.covale  # (n_atoms, )
        # is_motif_token[atom_with_coval_bond] = True

        return spread_token_wise(atom_array, is_motif_token)

    def sample_motif_atoms(self, atom_array):
        """
        Samples which atoms in motif tokens should be masked.
        This handles the case where you want the sidechain of a residue to not be motif.

        Argument attrs:
            - is_motif_token
            - is_motif_atom_with_fixed_seq
        """
        is_motif_atom = np.asarray(atom_array.is_motif_token, dtype=bool).copy()

        if random_condition(self.p_diffuse_motif_sidechains):
            backbone_atoms = ["N", "C", "CA"]
            if random_condition(self.p_include_oxygen_in_backbone_mask):
                backbone_atoms.append("O")
            is_motif_atom = is_motif_atom & np.isin(
                atom_array.atom_name, backbone_atoms
            )
        elif random_condition(self.p_diffuse_subgraph_atoms):
            is_motif_atom = sample_motif_subgraphs(
                atom_array=atom_array,
                **self.subgraph_sampling_kwargs,
            )

        # We also only want resolved atoms to be motif
        is_motif_atom = (is_motif_atom) & (atom_array.occupancy > 0.0)

        return is_motif_atom

    def sample(self, data):
        atom_array = data["atom_array"]

        atom_array.set_annotation(
            "is_motif_token", self.sample_motif_tokens(atom_array)
        )
        atom_array.set_annotation("is_motif_atom", self.sample_motif_atoms(atom_array))

        # After selecting the motif, we need to decide what conditioning strategy to apply
        atom_array = sample_conditioning_strategy(
            atom_array,
            p_fix_motif_sequence=self.p_fix_motif_sequence,
            p_fix_motif_coordinates=self.p_fix_motif_coordinates,
            p_unindex_motif_tokens=self.p_unindex_motif_tokens,
        )

        atom_array.set_annotation(
            "is_motif_atom_unindexed_motif_breakpoint",
            sample_unindexed_breaks(
                atom_array,
                remove_random_break=data["conditions"]["unindex_remove_random_break"],
                insert_random_break=data["conditions"]["unindex_insert_random_break"],
                leak_global_index=data["conditions"]["unindex_leak_global_index"],
            ),
        )

        return atom_array


class PPICondition(TrainingCondition):
    """Get condition indicating what is motif and what is to be diffused for protein-protein interaction training."""

    name = "ppi"

    def is_valid_for_example(self, data):
        # Extract relevant data
        atom_array = data["atom_array"]
        self.query_pn_unit_iids = data.get("query_pn_unit_iids")

        # Compute protein pn_unit_iids
        protein_pn_unit_iids = []
        for pn_unit_iid in np.unique(atom_array.pn_unit_iid):
            pn_unit_atom_array = atom_array[atom_array.pn_unit_iid == pn_unit_iid]
            pn_unit_is_protein = np.unique(pn_unit_atom_array.is_protein)

            if all(pn_unit_is_protein):  # Exclude cases of chimeric ligands
                protein_pn_unit_iids.append(pn_unit_iid)

        # This mask is intended to operate on binary protein-protein interfaces
        if (
            self.query_pn_unit_iids is None
            or len(self.query_pn_unit_iids) != 2
            or len(np.unique(self.query_pn_unit_iids)) != 2
        ):
            return False

        elif not all(
            [pn_unit in protein_pn_unit_iids for pn_unit in self.query_pn_unit_iids]
        ):
            return False

        else:
            # Randomly select one of the two query pn_unit_iids to be the binder
            # NOTE: Could also do this based on if only one will work uncropped, but since that
            # strategy will not always be applied, enforcing it here would bias the training data.
            binder_pn_unit = np.random.choice(self.query_pn_unit_iids)
            data["binder_pn_unit"] = binder_pn_unit
            atom_array.set_annotation(
                "is_binder_pn_unit", atom_array.pn_unit_iid == binder_pn_unit
            )
            return True

    # TODO: If I want to have multiple possible strategies for motif assignment (e.g. motif scaffolding for the binder)
    # should probably just have this function sample between them with a set of probabilities specified in the config.
    # Anything that makes it this far will have to be a valid PPI example with an assigned binder chain.
    def sample(self, data):
        atom_array = data["atom_array"]

        # Set `is_motif_token`
        # NOTE: In the future, we may want to diffuse part of the target or fix part of the binder
        is_motif_token = atom_array.pn_unit_iid != data["binder_pn_unit"]
        atom_array.set_annotation("is_motif_token", is_motif_token)

        # Set `is_motif_atom_with_fixed_seq`
        is_motif_atom_with_fixed_seq = (
            is_motif_token.copy()
        )  # We fix the target sequence in binder design
        atom_array.set_annotation(
            "is_motif_atom_with_fixed_seq", is_motif_atom_with_fixed_seq
        )

        # Set `is_motif_atom`
        is_motif_atom = (
            is_motif_token.copy()
        )  # The PPI mask should apply to all or no atoms of a token
        atom_array.set_annotation("is_motif_atom", is_motif_atom)

        # Set `is_motif_atom_with_fixed_pos`
        is_motif_atom_with_fixed_coord = (
            is_motif_token.copy()
        )  # We fully fix the target atom positions (at least for now)
        atom_array.set_annotation(
            "is_motif_atom_with_fixed_coord", is_motif_atom_with_fixed_coord
        )

        # Set `is_motif_atom_without_index`
        is_motif_atom_unindexed = np.zeros_like(
            is_motif_token
        )  # We want fixed indices for the target
        atom_array.set_annotation("is_motif_atom_unindexed", is_motif_atom_unindexed)

        # Set `is_motif_atom_unindexed_motif_breakpoint`
        is_motif_atom_unindexed_motif_breakpoint = np.zeros_like(is_motif_token)
        atom_array.set_annotation(
            "is_motif_atom_unindexed_motif_breakpoint",
            is_motif_atom_unindexed_motif_breakpoint,
        )
        return atom_array


##############################################################################################
# Additional conditioning classes
##############################################################################################


class SubtypeCondition(TrainingCondition):
    """
    Selects specific subtypes of atoms as motif and assigns conditioning strategies.
    """

    name = "subtype"

    def __init__(self, frequency: float, subtype: list[str], fix_pos: bool = False):
        self.frequency = frequency
        self.subtype = subtype
        self.fix_pos = fix_pos

    def is_valid_for_example(self, data):
        """
        For subtype conditioning, the example must contain the specified subtype
        """
        is_subtypes = [
            data["atom_array"].get_annotation(subtype).sum() for subtype in self.subtype
        ]
        if not np.any(is_subtypes):
            return False
        return True

    def sample(self, data):
        atom_array = data["atom_array"]

        is_motif = generate_subtype_mask(atom_array, self.subtype)
        is_motif = prune_unresolved_motif(atom_array, is_motif)
        atom_array.set_annotation("is_motif_token", is_motif)
        atom_array.set_annotation("is_motif_atom", is_motif)
        atom_array.set_annotation("is_motif_atom_with_fixed_seq", is_motif)

        if self.fix_pos:
            atom_array.set_annotation("is_motif_atom_with_fixed_coord", is_motif)
        else:
            atom_array.set_annotation(
                "is_motif_atom_with_fixed_coord", np.zeros(len(atom_array), dtype=bool)
            )
        atom_array.set_annotation(
            "is_motif_atom_unindexed", np.zeros(len(atom_array), dtype=bool)
        )
        atom_array.set_annotation(
            "is_motif_atom_unindexed_motif_breakpoint",
            np.zeros(len(atom_array), dtype=bool),
        )

        return atom_array


################# need mask -> condition refactor
def prune_unresolved_motif(atom_array, mask):
    """
    Prune the mask to only include resolved atoms.
    and for any residue that have unresolved atoms, set the whole residue to be False.
    """
    # Get the indices of the atoms that are resolved
    resolved_indices = np.where(atom_array.occupancy > 0.0)[0]

    # Create a mask for the resolved atoms
    resolved_mask = np.zeros_like(mask, dtype=bool)
    resolved_mask[resolved_indices] = True

    # Combine the original mask with the resolved mask
    combined_mask = mask & resolved_mask

    # Set the whole residue to be False if any atom in the residue is unresolved
    token_ids = np.unique(atom_array.token_id)
    for token_id in token_ids:
        if np.any(~combined_mask[atom_array.token_id == token_id]):
            combined_mask[atom_array.token_id == token_id] = False
    return combined_mask


def generate_subtype_mask(atom_array, subtypes):
    """
    Generate a mask for a specific subtype list of atoms.
    E.g. is_protein, is_ligand, is_dna etc.
    """
    all_masks = []
    for subtype in subtypes:
        if subtype not in atom_array.get_annotation_categories():
            raise ValueError(f"Subtype {subtype} not found in atom array annotations.")
        mask = atom_array.get_annotation(subtype)
        all_masks.append(mask)
    # Combine all masks using logical OR
    combined_mask = np.logical_or.reduce(all_masks)
    return combined_mask


##############################################################################################
# Shared assignment functions
##############################################################################################


def sample_motif_subgraphs(
    atom_array,
    residue_p_seed_furthest_from_o,
    residue_n_bond_expectation,
    hetatom_n_bond_expectation,
    residue_p_fix_all,
    hetatom_p_fix_all,
):
    """
    Returns a boolean mask over atoms, indicating which atoms are part of the sampled motif.
    Sampling is performed per residue, with sidechains optionally excluded based on bond-based neighborhood sampling.

    Handles both protein residues and heteroatoms (e.g., ligands).

    Args:
        atom_array: AtomArray with annotations is_motif_token, is_protein, occupancy, res_id.

    Returns:
        is_motif_atom: np.ndarray of shape (n_atoms,) with True for sampled motif atoms.
    """
    is_motif_token = atom_array.is_motif_token.copy()
    is_motif_atom = is_motif_token.copy()
    idxs = np.arange(atom_array.array_length(), dtype=int)
    starts = get_residue_starts(atom_array, add_exclusive_stop=True)

    for i, (start, end) in enumerate(zip(starts[:-1], starts[1:])):
        if not is_motif_token[start]:
            continue

        # Get the atoms of the current residue
        subset_mask = np.isin(idxs, idxs[start:end])
        atom_array_subset = atom_array[subset_mask]
        assert atom_array_subset.array_length() > 0

        args = {
            "p_seed_furthest_from_o": residue_p_seed_furthest_from_o,
            "n_bond_expectation": residue_n_bond_expectation,
            "p_fix_all": residue_p_fix_all,
        }
        if not atom_array_subset.is_protein.all():
            args.update(
                {
                    "p_seed_furthest_from_o": 0.0,
                    "n_bond_expectation": hetatom_n_bond_expectation,
                    "p_fix_all": hetatom_p_fix_all,
                }
            )
        try:
            mask = sample_subgraph_atoms(atom_array_subset, **args)
        except Exception as e:
            logger.warning(
                f"Failed to sample subgraph motif atoms for {atom_array_subset.res_name[0]}. Error: {e}"
            )
            mask = np.ones(atom_array_subset.array_length(), dtype=bool)

        is_motif_atom[subset_mask] = mask

    # We also only want resolved atoms to be motif
    is_motif_atom = (is_motif_atom) & (atom_array.occupancy > 0.0)

    return is_motif_atom


def sample_conditioning_strategy(
    atom_array,
    p_fix_motif_sequence,
    p_fix_motif_coordinates,
    p_unindex_motif_tokens,
):
    atom_array.set_annotation(
        "is_motif_atom_with_fixed_seq",
        sample_is_motif_atom_with_fixed_seq(
            atom_array, p_fix_motif_sequence=p_fix_motif_sequence
        ),
    )

    atom_array.set_annotation(
        "is_motif_atom_with_fixed_coord",
        sample_fix_motif_coordinates(
            atom_array, p_fix_motif_coordinates=p_fix_motif_coordinates
        ),
    )

    atom_array.set_annotation(
        "is_motif_atom_unindexed",
        sample_unindexed_atoms(
            atom_array, p_unindex_motif_tokens=p_unindex_motif_tokens
        ),
    )

    return atom_array


def sample_is_motif_atom_with_fixed_seq(atom_array, p_fix_motif_sequence):
    """
    Samples what kind of conditioning to apply to motif tokens.

    Argument attrs:
        - is_motif_token
    """
    if random_condition(p_fix_motif_sequence):
        is_motif_atom_with_fixed_seq = atom_array.is_motif_token.copy()
    else:
        is_motif_atom_with_fixed_seq = np.zeros(atom_array.array_length(), dtype=bool)

    # By default reveal sequence for non-protein
    is_motif_atom_with_fixed_seq = is_motif_atom_with_fixed_seq | ~atom_array.is_protein
    return is_motif_atom_with_fixed_seq


def sample_fix_motif_coordinates(atom_array, p_fix_motif_coordinates):
    """
    Universal function to decide if atoms' coords are fixed in the point cloud for conditioning.

    Argument attrs:
        - is_motif_atom_with_fixed_coord
    """
    if random_condition(p_fix_motif_coordinates):
        is_motif_atom_with_fixed_coord = atom_array.is_motif_atom.copy()
    else:
        is_motif_atom_with_fixed_coord = np.zeros(atom_array.array_length(), dtype=bool)
    return is_motif_atom_with_fixed_coord


def sample_unindexed_atoms(atom_array, p_unindex_motif_tokens):
    """
    Samples which atoms in motif tokens should be flagged for unindexing.

    Argument attrs:
        - is_motif_atom_unindexed
    """
    if random_condition(p_unindex_motif_tokens):
        is_motif_atom_unindexed = atom_array.is_motif_atom.copy()
    else:
        is_motif_atom_unindexed = np.zeros(atom_array.array_length(), dtype=bool)

    # ensure non-residue atoms are not already flagged
    is_motif_atom_unindexed = np.logical_and(
        is_motif_atom_unindexed, atom_array.is_residue
    )

    return is_motif_atom_unindexed


def sample_unindexed_breaks(
    atom_array,
    remove_random_break=False,
    insert_random_break=False,
    leak_global_index=False,
):
    is_unindexed_token = apply_token_wise(
        atom_array,
        atom_array.is_motif_atom_unindexed.copy(),
        function=lambda x: np.any(x),
    )
    starts = get_token_starts(atom_array)
    token_idxs = np.arange(len(starts))
    breaks_all = np.zeros(len(starts), dtype=bool)

    if is_unindexed_token.sum() == 1:
        breaks_all = is_unindexed_token
    elif np.any(is_unindexed_token):
        # ... Subset to unindexed tokens
        unindexed_token_starts = starts[is_unindexed_token]
        unindexed_token_resid = atom_array[unindexed_token_starts].res_id
        breaks = np.diff(unindexed_token_resid) != 1  # (M-1,)

        # ... Connect discontiguous regions
        if remove_random_break and np.any(breaks):
            break_idx = np.random.choice(np.flatnonzero(breaks), size=1, replace=False)
            breaks[break_idx] = False

        # ... Disconnect contiguous regions
        if insert_random_break:
            break_idx = np.random.choice(np.arange(len(breaks)), size=1, replace=False)
            breaks[break_idx] = True

        breaks[0] = True
        breaks = np.concatenate([np.array([False], dtype=bool), breaks])

        # ... Remove all breaks to leak global indices:
        if leak_global_index:
            breaks = False

        breaks_all[token_idxs[is_unindexed_token]] = breaks

    return spread_token_wise(atom_array, breaks_all)
