import numpy as np
from atomworks.enums import ChainType
from atomworks.ml.transforms._checks import (
    check_atom_array_annotation,
    check_contains_keys,
)
from atomworks.ml.transforms.base import Transform
from atomworks.ml.transforms.crop import resize_crop_info_if_too_many_atoms
from atomworks.ml.utils.token import (
    get_token_count,
    spread_token_wise,
)
from biotite.structure.basepairs import (
    _check_dssr_criteria,
    _get_proximate_residues,
    get_residue_masks,
    get_residue_starts_for,
)
from scipy.spatial import distance_matrix


def protein_dna_contact_contiguous_crop_mask(
    atom_array,
    protein_contact_atoms,
    dna_contact_atoms,
    contact_dist_cutoff,
    protein_expand_min,
    protein_expand_max,
    dna_expand_min,
    dna_expand_max,
):
    dna_contact, prot_contact = identify_and_sample_protein_dna_contact(
        atom_array, protein_contact_atoms, dna_contact_atoms, contact_dist_cutoff
    )

    # total_protein_expand = np.random.randint(protein_expand_min, protein_expand_max)
    left = np.random.randint(protein_expand_min, protein_expand_max)
    right = np.random.randint(protein_expand_min, protein_expand_max)
    protein_keep_mask = expand_connected_component_mask(
        atom_array, prot_contact, left, right
    )

    # total_dna_expand = np.random.randint(dna_expand_min, dna_expand_max)
    left = np.random.randint(dna_expand_min, dna_expand_max)
    right = np.random.randint(dna_expand_min, dna_expand_max)
    dna_keep_mask = get_dna_mask(atom_array, dna_contact, left, right)
    # count keep protein token num and dna token num

    mask = np.logical_or(protein_keep_mask, dna_keep_mask)

    requires_crop = np.any(mask)
    crop_atom_idxs = np.where(mask)[0]

    token_id = np.arange(get_token_count(atom_array), dtype=np.uint32)
    crop_token_idxs = spread_token_wise(atom_array, token_id)[mask]

    if get_token_count(atom_array[mask]) > 300:
        raise ValueError(
            "Noncanonical DNAs are causing token count explosion, skipping..."
        )

    return {
        "type": "ProteinDNAContactContiguousCrop",
        "requires_crop": requires_crop,
        "crop_atom_idxs": crop_atom_idxs,
        "crop_token_idxs": crop_token_idxs,
        "atom_array": atom_array,
    }


def atom_array_from_contact_dict(atom_array, contact_atoms):
    mask = []
    for row in atom_array:
        if (
            row.res_name in contact_atoms.keys()
            and row.atom_name in contact_atoms[row.res_name]
        ):
            mask.append(True)
        else:
            mask.append(False)

    return atom_array[mask]


def identify_and_sample_protein_dna_contact(
    atom_array, protein_contact_atoms, dna_contact_atoms, contact_dist=4
):
    if isinstance(protein_contact_atoms, dict):
        protein = atom_array_from_contact_dict(atom_array, protein_contact_atoms)
    elif isinstance(protein_contact_atoms, list):
        protein = atom_array[
            (atom_array.chain_type == ChainType.POLYPEPTIDE_L)
            & np.isin(atom_array.atom_name, protein_contact_atoms)
        ]
    elif isinstance(protein_contact_atoms, str):
        if protein_contact_atoms == "all":
            protein = atom_array[(atom_array.chain_type == ChainType.POLYPEPTIDE_L)]
        else:
            raise ValueError
    else:
        raise ValueError

    if isinstance(dna_contact_atoms, dict):
        atom_array = atom_array[atom_array.chain_type == ChainType.DNA]
        dna = atom_array_from_contact_dict(atom_array, dna_contact_atoms)
    elif isinstance(dna_contact_atoms, list):
        dna = atom_array[
            (atom_array.chain_type == ChainType.DNA)
            & (np.isin(atom_array.atom_name, dna_contact_atoms))
        ]
    elif isinstance(dna_contact_atoms, str):
        if dna_contact_atoms == "all":
            dna = atom_array[(atom_array.chain_type == ChainType.DNA)]
        else:
            raise ValueError
    else:
        raise ValueError
    pdist = distance_matrix(dna.coord, protein.coord)

    contacts = np.stack(np.where(pdist < contact_dist), axis=1)

    try:
        sample = contacts[np.random.choice(range(len(contacts)))]
    except Exception:
        raise ValueError("No protein-DNA contacts found")

    dna_contact = dna[sample[0]]
    prot_contact = protein[sample[1]]

    return dna_contact, prot_contact


def create_residue_mask(atom_array, first_atom_indices):
    """
    Creates a boolean mask for entire residues based on indices of their first atoms.
    Uses efficient broadcasting for better performance.

    Parameters
    ----------
    atom_array : biotite.structure.atom_array
        The atom array to create the mask for
    first_atom_indices : array-like
        Indices of the first atoms of the residues to select

    Returns
    -------
    numpy.ndarray
        Boolean mask that can be used to select all atoms of the specified residues
    """
    # Get target residue IDs and chain IDs as 2D arrays
    target_res_ids = atom_array.res_id[first_atom_indices][:, np.newaxis]
    target_chain_ids = atom_array.chain_id[first_atom_indices][:, np.newaxis]

    # Use broadcasting to create masks for all residues at once
    res_match = atom_array.res_id == target_res_ids
    chain_match = atom_array.chain_id == target_chain_ids

    # Combine the matches
    mask = (res_match & chain_match).any(axis=0)

    return mask


def expand_connected_component_mask(atom_array, origin, left_expand, right_expand):
    center = origin.within_poly_res_idx
    left = center - left_expand
    right = center + right_expand
    candidates = list(range(left, right))
    keep_mask = (atom_array.chain_id == origin.chain_id) & np.isin(
        atom_array.within_poly_res_idx, candidates
    )
    return keep_mask


def get_dna_mask(atom_array, origin, left_expand, right_expand):
    one_chain_mask = expand_connected_component_mask(
        atom_array, origin, left_expand, right_expand
    )

    pairs = base_pairs(atom_array)

    other_chain_first_atom_indices = []
    one_chain_first_atom_tags = np.zeros(len(atom_array), dtype=bool)
    for pair in pairs:
        if one_chain_mask[pair[0]]:
            other_chain_first_atom_indices.append(pair[1])
            one_chain_first_atom_tags[pair[0]] = True

        elif one_chain_mask[pair[1]]:
            other_chain_first_atom_indices.append(pair[0])
            one_chain_first_atom_tags[pair[1]] = True

    other_chain_mask = create_residue_mask(atom_array, other_chain_first_atom_indices)

    return np.logical_or(one_chain_mask, other_chain_mask)


class ProteinDNAContactContiguousCrop(Transform):
    """
    A transform the crops the DNA-protein contact region according to the continous region of contact.

    Args:
        protein_contact_type (str): The type of protein contact atoms to consider. Can be 'backbone', 'sidechain', 'all', or 'from_dict'
        dna_contact_type (str): The type of DNA contact atoms to consider. Can be 'backbone', 'base', 'all', or 'from_dict'
        contact_distance_cutoff (float): The distance cutoff for considering two atoms to be in contact
    """

    def __init__(
        self,
        protein_contact_type,
        dna_contact_type,
        contact_distance_cutoff=10.0,
        protein_expand_min=15,
        protein_expand_max=40,
        dna_expand_min=3,
        dna_expand_max=10,
        keep_uncropped_atom_array: bool = False,
        max_atoms_in_crop=None,
        protein_contact_atom_dict=None,
        dna_contact_atom_dict=None,
    ):
        if protein_contact_type == "backbone":
            self.protein_contact_atoms = ["N", "CA", "C"]
        elif protein_contact_type == "all":
            self.protein_contact_atoms = "all"

        if dna_contact_type == "backbone":
            self.dna_contact_atoms = ["P", "OP1", "OP2"]
        elif dna_contact_type == "base":
            self.dna_contact_atoms = {
                "DA": ["N7", "N6"],
                "DC": ["N4"],
                "DG": ["N7", "O6"],
                "DT": ["O4"],
            }
        else:
            self.dna_contact_atoms = "all"

        self.protein_contact_type = protein_contact_type
        self.dna_contact_type = dna_contact_type

        self.protein_expand_min = protein_expand_min
        self.protein_expand_max = protein_expand_max
        self.dna_expand_min = dna_expand_min
        self.dna_expand_max = dna_expand_max
        self.contact_distance_cutoff = contact_distance_cutoff

        self.keep_uncropped_atom_array = keep_uncropped_atom_array
        self.max_atoms_in_crop = max_atoms_in_crop

    def check_input(self, data: dict):
        check_contains_keys(data, ["atom_array"])
        check_atom_array_annotation(data, ["res_name"])

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]

        crop_info = protein_dna_contact_contiguous_crop_mask(
            atom_array,
            self.protein_contact_atoms,
            self.dna_contact_atoms,
            self.contact_distance_cutoff,
            self.protein_expand_min,
            self.protein_expand_max,
            self.dna_expand_min,
            self.dna_expand_max,
        )

        crop_info = resize_crop_info_if_too_many_atoms(
            crop_info=crop_info,
            atom_array=atom_array,
            max_atoms=self.max_atoms_in_crop,
        )

        if self.keep_uncropped_atom_array:
            data["uncropped_atom_array"] = atom_array

        if crop_info["requires_crop"]:
            data["atom_array"] = atom_array[crop_info["crop_atom_idxs"]]
            data["crop_info"] = crop_info
        else:
            data["atom_array"] = atom_array
        return data


def fill_nan_coords_with_random(atoms, min_val=-50, max_val=50, seed=None):
    """
    Fill NaN coordinates in a biotite AtomArray with random values.

    Parameters
    ----------
    atoms : biotite.structure.AtomArray
        The atom array containing coordinates to be filled
    min_val : float, optional
        Minimum value for random coordinates (default: -50)
    max_val : float, optional
        Maximum value for random coordinates (default: 50)
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    biotite.structure.AtomArray
        A new AtomArray with NaN coordinates filled
    """
    # Create a copy to avoid modifying the original
    filled_atoms = atoms.copy()

    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Get the coordinate array
    coords = filled_atoms.coord

    # Find indices of NaN values
    nan_mask = np.isnan(coords)

    # Generate random values for NaN positions
    random_coords = np.random.uniform(
        low=min_val, high=max_val, size=coords[nan_mask].shape
    )

    # Fill NaN values with random coordinates
    coords[nan_mask] = random_coords

    return filled_atoms


def base_pairs(atom_array, min_atoms_per_base=3, unique=True):
    """
    Use DSSR criteria to find the base pairs in an :class:`atom_array`.

    The algorithm is able to identify canonical and non-canonical
    base pairs. between the 5 common bases Adenine, Guanine, Thymine,
    Cytosine, and Uracil bound to Deoxyribose and Ribose.
    Each Base is mapped to the 5 common bases Adenine, Guanine, Thymine,
    Cytosine, and Uracil in a standard reference frame described in
    :footcite:`Olson2001` using :func:`map_nucleotide()`.

    The DSSR Criteria are as follows :footcite:`Lu2015`:

    (i) Distance between base origins <=15 Å

    (ii) Vertical separation between the base planes <=2.5 Å

    (iii) Angle between the base normal vectors <=65°

    (iv) Absence of stacking between the two bases

    (v) Presence of at least one hydrogen bond involving a base atom

    Parameters
    ----------
    atom_array : atom_array
        The :class:`atom_array` to find base pairs in.
    min_atoms_per_base : integer, optional (default: 3)
        The number of atoms a nucleotides' base must have to be
        considered a candidate for a base pair.
    unique : bool, optional (default: True)
        If ``True``, each base is assumed to be only paired with one
        other base. If multiple pairings are plausible, the pairing with
        the most hydrogen bonds is selected.

    Returns
    -------
    basepairs : ndarray, dtype=int, shape=(n,2)
        Each row is equivalent to one base pair and contains the first
        indices of the residues corresponding to each base.

    Notes
    -----
    The bases from the standard reference frame described in
    :footcite:`Olson2001` were modified such that only the base atoms
    are implemented.
    Sugar atoms (specifically C1') were disregarded, as nucleosides such
    as PSU do not posess the usual N-glycosidic linkage, thus leading to
    inaccurate results.

    The vertical separation is implemented as the scalar
    projection of the distance vectors between the base origins
    according to :footcite:`Lu1997` onto the averaged base normal
    vectors.

    The presence of base stacking is assumed if the following criteria
    are met :footcite:`Gabb1996`:

    (i) Distance between aromatic ring centers <=4.5 Å

    (ii) Angle between the ring normal vectors <=23°

    (iii) Angle between normalized distance vector between two ring
          centers and both bases' normal vectors <=40°

    Please note that ring normal vectors are assumed to be equal to the
    base normal vectors.

    For structures without hydrogens the accuracy of the algorithm is
    limited as the hydrogen bonds can be only checked be checked for
    plausibility.
    A hydrogen bond is considered as plausible if a cutoff of 3.6 Å
    between N/O atom pairs is met. 3.6Å was chosen as hydrogen bonds are
    typically 1.5-2.5Å in length. N-H and O-H bonds have a length of
    1.00Å and 0.96Å respectively. Thus, including some buffer, a 3.6Å
    cutoff should cover all hydrogen bonds.

    Examples
    --------
    Compute the base pairs for the structure with the PDB ID 1QXB:

    >>> from os.path import join
    >>> dna_helix = load_structure(join(path_to_structures, "base_pairs", "1qxb.cif"))
    >>> basepairs = base_pairs(dna_helix)
    >>> print(dna_helix[basepairs].res_name)
    [['DC' 'DG']
     ['DG' 'DC']
     ['DC' 'DG']
     ['DG' 'DC']
     ['DA' 'DT']
     ['DA' 'DT']
     ['DT' 'DA']
     ['DT' 'DA']
     ['DC' 'DG']
     ['DG' 'DC']
     ['DC' 'DG']
     ['DG' 'DC']]

    References
    ----------

    .. footbibliography::
    """
    dna_boolean = np.logical_and(
        atom_array.chain_type == ChainType.DNA,
        np.isin(atom_array.res_name, ["DA", "DG", "DT", "DC"]),
    )

    # Get the nucleotides for the given atom_array
    # Disregard the phosphate-backbone
    non_phosphate_boolean = ~np.isin(
        atom_array.atom_name, ["O5'", "P", "OP1", "OP2", "OP3", "HOP2", "HOP3"]
    )

    # Combine the two boolean masks
    boolean_mask = np.logical_and(non_phosphate_boolean, dna_boolean)

    # Get only nucleosides
    nucleosides = atom_array[boolean_mask]

    # Get the base pair candidates according to a N/O cutoff distance,
    # where each base is identified as the first index of its respective
    # residue
    n_o_mask = np.isin(nucleosides.element, ["N", "O"])

    nucleosides = fill_nan_coords_with_random(nucleosides)
    basepair_candidates, n_o_matches = _get_proximate_residues(
        nucleosides, n_o_mask, 3.6
    )

    # Contains the plausible base pairs
    basepairs = []
    # Contains the number of hydrogens for each plausible base pair
    basepairs_hbonds = []

    # Get the residue masks for each residue
    base_masks = get_residue_masks(nucleosides, basepair_candidates.flatten())

    # Group every two masks together for easy iteration (each 'row' is
    # respective to a row in ``basepair_candidates``)
    base_masks = base_masks.reshape(
        (basepair_candidates.shape[0], 2, nucleosides.shape[0])
    )

    for (base1_index, base2_index), (base1_mask, base2_mask), n_o_pairs in zip(
        basepair_candidates, base_masks, n_o_matches
    ):
        base1 = nucleosides[base1_mask]
        base2 = nucleosides[base2_mask]

        hbonds = _check_dssr_criteria((base1, base2), min_atoms_per_base, unique)

        # If no hydrogens are present use the number N/O pairs to
        # decide between multiple pairing possibilities.

        if hbonds is None:
            # Each N/O-pair is detected twice. Thus, the number of
            # matches must be divided by two.
            hbonds = n_o_pairs / 2
        if hbonds != -1:
            basepairs.append((base1_index, base2_index))
            if unique:
                basepairs_hbonds.append(hbonds)

    basepair_array = np.array(basepairs)

    if unique:
        # Contains all non-unique base pairs that are flagged to be
        # removed
        to_remove = []

        # Get all bases that have non-unique pairing interactions
        base_indices, occurrences = np.unique(basepairs, return_counts=True)
        for base_index, occurrence in zip(base_indices, occurrences):
            if occurrence > 1:
                # Write the non-unique base pairs to a dictionary as
                # 'index: number of hydrogen bonds'
                remove_candidates = {}
                for i, row in enumerate(np.asarray(basepair_array == base_index)):
                    if np.any(row):
                        remove_candidates[i] = basepairs_hbonds[i]
                # Flag all non-unique base pairs for removal except the
                # one that has the most hydrogen bonds
                del remove_candidates[max(remove_candidates, key=remove_candidates.get)]
                to_remove += list(remove_candidates.keys())
        # Remove all flagged base pairs from the output `ndarray`
        basepair_array = np.delete(basepair_array, to_remove, axis=0)

    # Remap values to original atom array
    if len(basepair_array) > 0:
        basepair_array = np.where(boolean_mask)[0][basepair_array]
        for i, row in enumerate(basepair_array):
            basepair_array[i] = get_residue_starts_for(atom_array, row)
    return basepair_array
