import numpy as np
from biotite.structure import Atom, array, concatenate


def add_oxt_each_chain(atom_array):
    """
    Add a missing OXT atom to the C-terminus of each protein chain in the array.

    Args:
        atom_array (AtomArray): Input structure containing one or more chains.

    Returns:
        AtomArray: Structure with OXT atoms added to protein chains where absent.
    """
    updated_chains = []
    chain_ids = np.unique(atom_array.chain_id)

    for chain_id in chain_ids:
        chain_mask = atom_array.chain_id == chain_id
        chain_array = atom_array[chain_mask]
        updated_chain = add_oxt_to_chain(chain_array)
        updated_chains.append(updated_chain)

    # Reassemble the processed chains into a single AtomArray
    return concatenate(updated_chains)


def add_oxt_to_chain(chain_array):
    """
    Add an OXT atom to the C-terminal residue of a single protein chain if missing.

    Args:
        chain_array (AtomArray): AtomArray corresponding to one chain.

    Returns:
        AtomArray: Chain with OXT added if needed; otherwise unchanged.
    """
    # Require the is_protein annotation to avoid modifying non-protein chains
    is_protein_ann = getattr(chain_array, "is_protein", None)
    if is_protein_ann is None:
        raise ValueError("atom_array is missing required 'is_protein' annotation")
    if not bool(np.all(is_protein_ann)):
        return chain_array  # Skip chains that are not fully protein

    # Identify the C-terminal residue for this chain
    c_terminal_res_id = np.max(chain_array.res_id)
    c_terminal_mask = chain_array.res_id == c_terminal_res_id
    c_terminal_atoms = chain_array[c_terminal_mask]

    # If OXT already exists, return unchanged
    if "OXT" in c_terminal_atoms.atom_name.tolist():
        return chain_array

    # Extract coordinates for the required atoms; skip if any are missing
    try:
        c_coord = c_terminal_atoms.coord[c_terminal_atoms.atom_name == "C"][0]
        ca_coord = c_terminal_atoms.coord[c_terminal_atoms.atom_name == "CA"][0]
        o_coord = c_terminal_atoms.coord[c_terminal_atoms.atom_name == "O"][0]
    except IndexError:
        return chain_array  # Cannot place OXT without C, CA, and O

    # Calculate OXT coordinates via reflection about the C–CA vector
    oxt_coord = calculate_oxt_coord(o_coord, ca_coord, c_coord)

    # Construct the new OXT atom
    oxt_atom = Atom(
        coord=oxt_coord,
        atom_name="OXT",
        res_id=c_terminal_atoms.res_id[0],
        res_name=c_terminal_atoms.res_name[0],
        chain_id=c_terminal_atoms.chain_id[0],
        element="O",
        hetero=False,
    )

    # Append OXT to the chain and return
    return concatenate([chain_array, array([oxt_atom])])


def calculate_oxt_coord(o_coord, ca_coord, c_coord):
    """
    Compute the OXT coordinates by reflecting the O atom across the C–CA vector.

    Args:
        o_coord (numpy.ndarray): Coordinates of the carbonyl oxygen (O).
        ca_coord (numpy.ndarray): Coordinates of the alpha carbon (CA).
        c_coord (numpy.ndarray): Coordinates of the carbonyl carbon (C).

    Returns:
        numpy.ndarray: Coordinates of the synthesized OXT atom.
    """
    # Vectors from C to CA and from O toward C
    c_ca_vector = ca_coord - c_coord
    o_c_vector = c_coord - o_coord

    # Reflect O about the C–CA vector to position OXT
    oxt_vector = (
        o_c_vector
        - 2
        * np.dot(o_c_vector, c_ca_vector)
        / np.linalg.norm(c_ca_vector) ** 2
        * c_ca_vector
    )
    oxt_coord = c_coord + oxt_vector

    return oxt_coord
