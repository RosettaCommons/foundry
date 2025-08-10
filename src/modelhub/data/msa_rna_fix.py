"""Transforms on MSAs"""

from __future__ import annotations

import logging
from os import PathLike
from pathlib import Path

import numpy as np
from biotite.structure import AtomArray
from cifutils.enums import ChainType
from datahub.transforms._checks import (
    check_atom_array_annotation,
    check_contains_keys,
    check_is_instance,
)
from datahub.transforms.base import Transform
from datahub.transforms.msa._msa_loading_utils import (
    get_msa_path,
    load_msa_data_from_path,
)
from datahub.utils.io import cache_to_disk_as_pickle

logger = logging.getLogger(__name__)


def load_polymer_msas_fixed_rna(
    atom_array: AtomArray,
    chain_info: dict,
    protein_msa_dirs: list[dict[str, str]],
    rna_msa_dirs: list[dict[str, str]],
    max_msa_sequences: int = 10_000,
    msa_cache_dir: PathLike | None = None,
    use_paths_in_chain_info: bool = True,
    raise_if_missing_msa_for_protein_of_length_n: int | None = None,
) -> dict[str, np.array]:
    """
    Load MSAs for all polymer chains in the AtomArray and store them in a dictionary. See the LoadPolymerMSAs transform for more information
    Args:
        atom_array (AtomArray): The AtomArray for the full structure
        chain_info (dict): A dictionary containing chain information, including:
            - processed_entity_non_canonical_sequence: The non-canonical sequence for the chain
            - processed_entity_canonical_sequence: The canonical sequence for the chain
            - chain_type: The type of the chain (e.g., protein, RNA)
            - msa_path (optional): The path to the MSA file for the chain, if available
        protein_msa_dirs (list[dict[str, str]]): The directories containing the protein MSAs and their associated file types.
        rna_msa_dirs (list[dict[str, str]]): The directories containing the RNA MSAs and their associated file types.
        max_msa_sequences (int): The maximum number of sequences to load from the MSA files. Defaults to 10_000.
        msa_cache_dir (PathLike | None): The directory to cache the parsed MSA data (since loading from text files is slow). If None, caching is turned off.
        use_paths_in_chain_info (bool): Whether to use the MSA paths provided in the chain_info dictionary. If True, we will first check the chain_info dictionary for MSA paths.
        raise_if_missing_msa_for_protein_of_length_n (int | None): If provided, raises an error if a protein of length >= n is missing an MSA file.
    Returns:
        dict[str, np.array]: A dictionary mapping chain IDs to their corresponding MSA data
    """
    msas_by_chain_id = {}

    # NOTE: If `msa_cache_dir` is `None`, the cache decorator will be a no-op
    cached_load_msa_data_from_path = cache_to_disk_as_pickle(msa_cache_dir)(
        load_msa_data_from_path
    )

    for chain_id in np.unique(
        atom_array.chain_id[np.isin(atom_array.chain_type, ChainType.get_polymers())]
    ):
        non_canonical_sequence = chain_info[chain_id][
            "processed_entity_non_canonical_sequence"
        ]
        canonical_sequence = chain_info[chain_id]["processed_entity_canonical_sequence"]
        chain_type = chain_info[chain_id]["chain_type"]

        # Set the query chain tax_id to "query" to avoid pairing issues downstream (we force all query sequences to be paired with themselves)
        # Subsequent occurrences of the query sequence will not have the "query" tax ID, and will be paired appropriately
        query_chain_msa_tax_id = "query"

        # ... find the path
        msa_file_path = None
        if (
            use_paths_in_chain_info
            and "msa_path" in chain_info[chain_id]
            and chain_info[chain_id]["msa_path"] is not None
        ):
            # Use provided path
            msa_file_path = Path(chain_info[chain_id]["msa_path"])
        else:
            # Check both canonical and non-canonical sequences
            for sequence in [non_canonical_sequence, canonical_sequence]:
                if chain_type.is_protein() and protein_msa_dirs:
                    msa_file_path = get_msa_path(sequence, protein_msa_dirs)
                elif chain_type == ChainType.RNA and rna_msa_dirs:
                    msa_file_path = get_msa_path(sequence, rna_msa_dirs)
                    if not msa_file_path:
                        msa_file_path = get_msa_path(
                            sequence.replace("U", "T"), rna_msa_dirs
                        )
                if msa_file_path:
                    break

        if msa_file_path is None:
            # If no MSA file path is found, we skip this chain
            if raise_if_missing_msa_for_protein_of_length_n is not None:
                if (
                    chain_type.is_protein()
                    and len(canonical_sequence)
                    >= raise_if_missing_msa_for_protein_of_length_n
                ):
                    raise ValueError(
                        f"MSA file not found for protein of length {len(canonical_sequence)}"
                    )
            continue

        assert (
            msa_file_path.exists()
        ), f"MSA file not found at given path: {msa_file_path}"

        # ... load the MSA data from the specified path
        msa_data = cached_load_msa_data_from_path(
            msa_file_path=msa_file_path,
            chain_type=chain_type,
            max_msa_sequences=max_msa_sequences,
            query_tax_id=query_chain_msa_tax_id,
        )

        if msa_data["msa"] is not None:
            msas_by_chain_id[chain_id] = {
                **msa_data,
                "msa_is_padded_mask": np.zeros(
                    msa_data["msa"].shape, dtype=bool
                ),  # 1 = padded, 0 = not padded
            }

    return msas_by_chain_id


class LoadPolymerMSAsFixedRNA(Transform):
    """Load MSAs for all polymer chains in the AtomArray.

    For the MSAs that are found, store the MSA (as a np.array of integers), insertions,
    tax IDs, and pre-computed sequence similarities in `polymer_msas_by_chain_id`
    indexed by chain_id (e.g., "A").

    Note that MSAs may be found in two ways:
        (1) By loading from the MSA files on disk based on the sequence hash(e.g., for training data).
        (2) By using specific MSA paths provided in the chain_info dictionary (e.g., for inference).

    We check both the canonical and non-canonical sequences for MSAs, preferring the canonical sequence if both are present.

    Args:
        protein_msa_dirs (list[dict]): The directories containing the protein MSAs and
            their associated file types, as a list of dictionaries. If multiple
            directories are provided, all of them will be searched. Keys in the dictionary
            are:
                - dir (str): The directory where the MSA files are stored.
                - extension (str): The file extension of the MSA files (e.g., ".a3m.gz" or ".fasta").
                - directory_depth (int, optional): The directory nesting depth, i.e., the MSA file
                  might be stored at `dir/d8/07/d8074f77ba.a3m.gz`. Must be sharded
                  by the first two characters of the sequence hash. Defaults to 0 (flat directory).
            Note:
                (a) The files must be named using the SHA-256 hash of the sequence (see `hash_sequence` in
                    `utils/misc`).
                (b) Order matters - directories will be searched in the order provided, and the first match will be returned.
        rna_msa_dirs (list[dict]): The directories containing the RNA MSAs and their
            associated file types, as a list of dictionaries. See `protein_msa_dirs`
            for directory structure details.
        use_paths_in_chain_info (bool): Whether to use the MSA paths provided in the chain_info dictionary.
            E.g., for inference mode. If True, we will first check the chain_info dictionary for MSA paths.
        max_msa_sequences (int, optional): The maximum number of sequences to load from
            the MSA files. Defaults to 10000. Only applies when loading; further
            sub-sampling of the MSA occurs downstream (e.g., for the standard or extra MSA stack).
            AF-3 used a large value (~16K), but our MSAs on disk are already pre-filtered to 10K.
        msa_cache_dir (PathLike, optional): The directory to cache the parsed MSA data
            (since loading from text files is slow). If None, caching is turned off.
        raise_if_missing_msa_for_protein_of_length_n (int | None): If provided, raises an error if a protein of length >= n is missing an MSA file.

    The `polymer_msas_by_chain_id` dictionary which is added contains the following keys:
        - msa: The MSA as a 2D np.array of integers, using the encoding specified in
          `_msa_constants.py`. Note that this encoding is transitory and will be
          converted to model-specific token indices later.
        - ins: The insertion array for the MSA, indicating the number of insertions to
          the LEFT of a given index, stored as a 2D np.array of integers.
        - tax_ids: The taxonomic IDs for each sequence in the MSA, stored as a 1D
          np.array of strings.
        - sequence_similarity: The sequence similarity to the query sequence for each
          row in the MSA.
        - msa_is_padded_mask: A mask indicating whether a given position in the MSA is
          padded (0) or not (1); defaults to 1 for all positions. Used downstream when
          filling the full MSA from the encoded MSA.
    """

    max_msa_sequences: int
    protein_msa_dirs: list[dict]
    rna_msa_dirs: list[dict]

    def __init__(
        self,
        protein_msa_dirs: list[
            dict
        ] = [],  # Example: [{"dir": "/path/to/protein/msas", "extension": ".a3m.gz", "directory_depth": 2}]
        rna_msa_dirs: list[dict] = [],
        max_msa_sequences: int = 10000,
        msa_cache_dir: PathLike | None = None,
        use_paths_in_chain_info: bool = True,
        raise_if_missing_msa_for_protein_of_length_n: int | None = None,
    ):
        self.max_msa_sequences = max_msa_sequences
        self.protein_msa_dirs = protein_msa_dirs
        self.rna_msa_dirs = rna_msa_dirs
        self.msa_cache_dir = msa_cache_dir
        self.use_paths_in_chain_info = use_paths_in_chain_info
        self.raise_if_missing_msa_for_protein_of_length_n = (
            raise_if_missing_msa_for_protein_of_length_n
        )

    def check_input(self, data: dict):
        check_contains_keys(data, ["atom_array", "chain_info"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["chain_type", "chain_id"])

    def forward(self, data: dict) -> dict:
        polymer_msas_by_chain_id = load_polymer_msas_fixed_rna(
            atom_array=data["atom_array"],
            chain_info=data["chain_info"],
            protein_msa_dirs=self.protein_msa_dirs,
            rna_msa_dirs=self.rna_msa_dirs,
            max_msa_sequences=self.max_msa_sequences,
            msa_cache_dir=self.msa_cache_dir,
            use_paths_in_chain_info=self.use_paths_in_chain_info,
            raise_if_missing_msa_for_protein_of_length_n=self.raise_if_missing_msa_for_protein_of_length_n,
        )
        data["polymer_msas_by_chain_id"] = polymer_msas_by_chain_id

        return data
