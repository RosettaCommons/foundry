"""Transforms on MSAs"""

from __future__ import annotations

from biotite.structure import AtomArray

from rf2aa.data_new.transforms._checks import (
    check_atom_array_annotation,
    check_contains_keys,
    check_is_instance,
)
from rf2aa.data_new.transforms.base import Transform


class LoadPolymerMSAs(Transform):
    """
    Load polymer MSAs for the given AtomArray.
    Store the MSAs as a dictionary indexed by chain_id (e.g., "A")
    """

    def check_input(self, data: dict):
        check_contains_keys(data, ["atom_array", "chain_info"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["chain_info", "chain_type", "chain_id"])

    def forward(self, data: dict) -> dict:
        pass

        # NOTE: Function in progress; committing for simplicity

        # atom_array = data["atom_array"]
        # chain_info = data["chain_info"]
        # polymer_chain_ids = np.unique(atom_array.chain_id[atom_array.chain_type == ChainType.POLYPEPTIDE_L])

        # msas = {}
        # for chain_id in polymer_chain_ids:
        #     # Get MSA lookup identifier
        #     msa_lookup_id = get_template_msa_lookup_id(data["pdb_id"], chain_id)

        #     # Lookup MSA information from disk

        #     atom_array = data["atom_array"]
        #     data["polymer_msas"] = load_polymer_msas(atom_array)
        # return data
