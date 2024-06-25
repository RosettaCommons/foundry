"""
Transforms and helper functions to convert from `AtomArray` objects representing polymers
to various polymer encoding schemes and back. Polymer encodings are specified via:
 - a tuple (or 1D array) of sequence tokens (e.g. amino acid names, nucleotide names,
    unknown token names)
 - a tuple of tuples (or 2D array) of atom names for each token (e.g. atom names for
    each amino acid)

The order of the tokens in the sequence determines the integer encoding of the token.
The order of the atom names in the tuple determines the integer encoding of the atom name
within the token.

During encoding, sequences of tokens are converted to sequences of integers, and the
AtomArray of coordinates is converted to a (N_res, N_atoms_per_token, 3) tensor.

Example encodings are:
    - RF2AA's atom36 encoding
    - AF2's atom14 encoding
    - AF2's atom37 encoding
"""

from functools import cache
from typing import Any, Literal

import biotite.structure as struc
import numpy as np
import torch
from assertpy import assert_that
from biotite.structure import AtomArray

from rf2aa.chemical import ChemicalData, initialize_chemdata
from rf2aa.data_new.transforms._checks import (
    check_atom_array_annotation,
    check_contains_keys,
    check_is_instance,
    check_nonzero_length,
)
from rf2aa.data_new.transforms.base import Transform

initialize_chemdata()
chemdata = ChemicalData()


def get_encoding_dict(seq_tokens: np.ndarray, token_atoms: np.ndarray):
    """Get dictionary based representation of the encoding."""
    seq_tokens = tuple(seq_tokens)
    token_atoms = tuple([tuple(row) for row in token_atoms])
    return _get_cached_encoding_dict(seq_tokens, token_atoms)


@cache
def _get_cached_encoding_dict(seq_tokens: tuple[str, ...], token_atoms: tuple[tuple[str | None, ...], ...]):
    """Cached conversion of list-based encoding to dictionary-based encoding.

    Used for encoding atom arrays to various encodings defined via a list-based format as used in `ChemicalData`.

    NOTE: This function is seperate from `get_encoding_dict`, because that API uses numpy arrays in the
     function signature for convenience, which cannot be hashed and therefore do not allow the @cache decorator.
    """
    token_to_idx = {token: i for i, token in enumerate(seq_tokens)}

    token_and_atom_to_idx = {}
    for token, token_idx in token_to_idx.items():
        for atom_idx, atom_name in enumerate(token_atoms[token_idx]):
            if atom_name:
                # Atom name exists in this token (otherwise it will be `None`)
                atom_name = atom_name.strip()
                token_and_atom_to_idx[token, atom_name] = atom_idx

    return token_to_idx, token_and_atom_to_idx


def atom_array_to_encoding(
    atom_array: AtomArray,
    encoding_seq_tokens: list[str] = np.array(chemdata.num2aa[: chemdata.NPROTAAS]),
    encoding_token_atoms: list[list[str]] = np.array(chemdata.aa2long)[:, : chemdata.NHEAVYPROT],
    default_coord: np.ndarray | float = float("nan"),
    unknown_token: str | dict[str | int, str] = "UNK",
):
    """
    Convert an atom array containing polymer information into an arbitrary encoding specified by
    `encoding_seq_tokens` and `encoding_token_atoms`.

    Args:
        atom_array (AtomArray): The atom array containing polymer information.
        encoding_seq_tokens (list[str], optional): List of sequence tokens for encoding.
        encoding_token_atoms (list[list[str]], optional): List of token atoms for encoding.
        default_coord (np.ndarray | float, optional): Default coordinate value. Defaults to float("nan").
        unknown_token (str | dict[str | int, str], optional): Token to use for unknown residues. Defaults to "UNK".

    Returns:
        tuple: A tuple containing:
            - encoded_coord (np.ndarray): Encoded coordinates of shape [n_res, n_atoms_per_token, 3].
            - encoded_mask (np.ndarray): Encoded mask of shape [n_res, n_atoms_per_token]. Holds information
                about which atoms are resolved in the encoded sequence.
            - encoded_seq (np.ndarray): Encoded sequence of shape [n_res].

    WARNING:
        - This function is only intended for `AtomArray` objects containing polymer information
            (i.e. proteins & nucleic acids).
    """
    # Ensure all encoding information is given as numpy arrays
    token_to_idx, token_and_atom_to_idx = get_encoding_dict(encoding_seq_tokens, encoding_token_atoms)
    n_tokens, n_atoms_per_token = encoding_token_atoms.shape

    # Extract atom array information
    n_res = struc.get_residue_count(atom_array)

    # Init encoded arrays
    encoded_coord = np.full(
        (n_res, n_atoms_per_token, 3), fill_value=default_coord, dtype=np.float32
    )  # [n_res, n_atoms_per_token, 3] (float)

    encoded_mask = np.zeros((n_res, n_atoms_per_token), dtype=bool)  # [n_res, n_atoms_per_token] (bool)
    encoded_seq = np.empty((n_res), dtype=int)  # [n_res] (int)

    # Iterate over residues (# TODO: Speed up by vectorizing if necessary)
    has_chain_type_annotation = "chain_type" in atom_array.get_annotation_categories()
    for i, res in enumerate(struc.residue_iter(atom_array)):
        res_name = res.res_name[0]

        # Deal with unknown tokens
        if res_name not in token_to_idx:
            if isinstance(unknown_token, str) or not has_chain_type_annotation:
                res_name = unknown_token
            else:
                chain_type = res.chain_type[0]
                res_name = unknown_token[chain_type]

        # Encode sequence
        encoded_seq[i] = token_to_idx[res_name]

        # Encode coords
        for atom in res:
            if (res_name, atom.atom_name) in token_and_atom_to_idx and atom.occupancy > 0:
                to_idx = token_and_atom_to_idx[(res_name, atom.atom_name)]
                encoded_coord[i, to_idx, :] = atom.coord
                encoded_mask[i, to_idx] = True

    return encoded_coord, encoded_mask, encoded_seq


def atom_array_from_encoding(
    encoded_coord: torch.Tensor | np.ndarray,
    encoded_mask: torch.Tensor | np.ndarray,
    encoded_seq: torch.Tensor | np.ndarray,
    encoding_seq_tokens: list[str] = np.array(chemdata.num2aa),
    encoding_token_atoms: list[list[str]] = np.array(chemdata.aa2long)[:, : chemdata.NHEAVYPROT],
    encoding_token_elements: list[list[str]] = np.array(chemdata.aa2elt)[:, : chemdata.NHEAVYPROT],
    chain_id: str = "A",  # TODO: Allow passing a numpy array of chain ids
    # TODO: Allow passing a res_id
):
    """
    Create an AtomArray from an encoded polymer coordinate tensor, atom mask, and sequence.

    Args:
        encoded_coord (torch.Tensor | np.ndarray): Encoded coordinates tensor.
        encoded_mask (torch.Tensor | np.ndarray): Encoded mask tensor.
        encoded_seq (torch.Tensor | np.ndarray): Encoded sequence tensor.
        encoding_seq_tokens (list[str], optional): List of sequence tokens. Defaults to np.array(chemdata.num2aa).
        encoding_token_atoms (list[list[str]], optional): List of token atoms. Defaults to np.array(chemdata.aa2long)[:, : chemdata.NHEAVYPROT].
        encoding_token_elements (list[list[str]], optional): List of token elements. Defaults to np.array(chemdata.aa2elt)[:, : chemdata.NHEAVYPROT].
        chain_id (str, optional): Chain ID. Defaults to "A".

    Returns:
        AtomArray: The created AtomArray.
    """
    # Turn tensors into numpy arrays if necessary
    for tensor in [encoded_coord, encoded_mask, encoded_seq]:
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.cpu().numpy()

    # Decode sequence:
    seq = np.array(encoding_seq_tokens)[encoded_seq]  # [n_res] (str)

    # Extract element and atom name information via the encoding
    element = np.array(encoding_token_elements)[encoded_seq]  # [n_res, n_atoms_per_token] (str)
    atom_name = np.array(encoding_token_atoms)[encoded_seq]  # [n_res, n_atoms_per_token] (str)

    # Determine which atoms should exist in each token, and how many atoms are in each token
    atom_should_exist = atom_name != None  # noqa  # [n_res, n_atoms_per_token] (bool)
    atoms_per_res = np.sum(atom_should_exist, axis=1)  # [n_res] (int)

    # Set up atom array
    n_res = len(seq)
    n_atom = np.sum(atoms_per_res)
    atom_array = AtomArray(length=n_atom)

    # ... flatten occupancy & validate that masking did not miss any existing atoms
    atom_array.set_annotation("occupancy", encoded_mask[atom_should_exist])
    assert_that(np.sum(encoded_mask)).is_equal_to(np.sum(atom_array.occupancy))

    # ... flatten and annotate coordinates
    atom_array.coord = encoded_coord[atom_should_exist]

    # ... flatten atom names and strip whitespace in atom names
    _strip_whitespace = np.vectorize(lambda x: x.strip())
    atom_array.atom_name = _strip_whitespace(atom_name[atom_should_exist])

    # ... flatten element info
    atom_array.element = element[atom_should_exist]

    # ... repeat residue name and id for each atom in the residue
    atom_array.res_name = np.repeat(seq, atoms_per_res)
    atom_array.res_id = np.repeat(np.arange(1, n_res + 1), atoms_per_res)
    atom_array.atom_id = np.arange(n_atom)

    # ... repeat chain id for each atom in the residue
    atom_array.chain_id = np.repeat(np.array(chain_id), n_atom)

    return atom_array


# Convenience functions for common encodings
def atom_array_to_rf2aa_atom36(atom_array: AtomArray, **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RF2 encoding for all atoms of amino acids and nucleic acids (including hydrogens)"""
    return atom_array_to_encoding(
        atom_array,
        encoding_seq_tokens=np.array(chemdata.num2aa[: chemdata.NNAPROTAAS]),  # [n_tokens] (n_tokens = 32)
        encoding_token_atoms=np.array(chemdata.aa2long),  # [n_tokens, n_atoms_per_token] (n_atoms_per_token = 36)
        **kwargs,
    )


def atom_array_to_rf2aa_atom22(atom_array: AtomArray, **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RF2 encoding for heavy atoms of amino acids and nucleic acids"""
    return atom_array_to_encoding(
        atom_array,
        encoding_seq_tokens=np.array(chemdata.num2aa[: chemdata.NNAPROTAAS]),  # [n_tokens] (n_tokens = 32)
        encoding_token_atoms=np.array(chemdata.aa2long)[
            :, : chemdata.NHEAVY
        ],  # [n_tokens, n_atoms_per_token] (n_atoms_per_token = 22)
        **kwargs,
    )


def atom_array_to_rf2aa_atom14(atom_array: AtomArray, **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RF2 encoding for heavy atoms of amino acids"""
    return atom_array_to_encoding(
        atom_array,
        encoding_seq_tokens=np.array(chemdata.num2aa[: chemdata.NPROTAAS]),  # [n_tokens] (n_tokens = 22)
        encoding_token_atoms=np.array(chemdata.aa2long)[
            :, : chemdata.NHEAVYPROT
        ],  # [n_tokens, n_atoms_per_token] (n_atoms_per_token = 14)
        **kwargs,
    )


class PolymersToRF2Atom36Encoding(Transform):
    """
    Convert atom array coordinates of all polymers to an RF2AA atom36 encoding.
    """

    def __init__(
        self,
        default_coord: float | np.ndarray = float("nan"),
        unknown_token: str | dict[str | int, str] = "UNK",
        rf2aa_encoding: Literal["atom36", "atom22", "atom14"] = "atom36",
    ):
        if rf2aa_encoding == "atom22":
            self.to_encoding = atom_array_to_rf2aa_atom22
        elif rf2aa_encoding == "atom14":
            self.to_encoding = atom_array_to_rf2aa_atom14
        elif rf2aa_encoding == "atom36":
            self.to_encoding = atom_array_to_rf2aa_atom36
        else:
            raise ValueError(f"Unknown RF2AA encoding: {rf2aa_encoding}")

        self.default_coord = default_coord
        self.unknown_token = unknown_token

    def check_input(self, data: dict[str, Any]):
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_nonzero_length(data, "atom_array")
        check_atom_array_annotation(data, ["is_polymer", "occupancy"])

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array = data["atom_array"]
        polymers = atom_array[atom_array.is_polymer]

        # TODO: possibly carry over chain type, etc.
        xyz, mask, seq = self.to_encoding(
            polymers,
            default_coord=self.default_coord,
            unknown_token=self.unknown_token,
        )

        data["encoded"] = dict(xyz=xyz, mask=mask, seq=seq)
        return data
