"""Transforms on atom arrays."""

from __future__ import annotations

import numpy as np
from biotite.structure import AtomArray

from data.data_constants import ChainType
from data.data_preprocessor import DataPreprocessor
from rf2aa.chemical import ChemicalData
from rf2aa.data_new.transforms._checks import (
    check_atom_array_annotation,
    check_contains_keys,
    check_is_instance,
)
from rf2aa.data_new.transforms.base import AddData, Compose, LogData, RemoveKeys, SubsetToKeys, Transform


# Convenience utils
def get_atom(atom_array: AtomArray, chain_id: str, res_id: int, atom_name: str) -> AtomArray:
    """Select an atom from an atom array."""
    return atom_array[
        (atom_array.chain_id == chain_id) & (atom_array.res_id == res_id) & (atom_array.atom_name == atom_name)
    ]


def get_residue(residue_array: AtomArray, chain_id: str, res_id: int) -> AtomArray:
    """Select a residue from an atom array."""
    return residue_array[(residue_array.chain_id == chain_id) & (residue_array.res_id == res_id)]


def get_chain(chain_array: AtomArray, chain_id: str) -> AtomArray:
    """Select a chain from an atom array."""
    return chain_array[chain_array.chain_id == chain_id]


def get_molecule(atom_array: AtomArray, molecule_full_id: str) -> AtomArray:
    """Select a molecule from an atom array."""
    molecule_id = molecule_full_id.split(":")[0]
    return atom_array[atom_array.molecule_id == molecule_id]


# Transforms


class FilterAndAnnotateMolecules(Transform):
    """
    Filters the given AtomArray to only include atoms that belong to the specified molecules,
    and adds annotations for the full molecule ID and the molecule ID.
    """

    def check_input(self, data: dict):
        check_contains_keys(data, ["atom_array", "all_molecule_full_ids"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["chain_full_id", "chain_id"])

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        all_molecule_full_ids = data["all_molecule_full_ids"]

        # Count the maximum number of charatcers in a molecule id
        max_molecule_id_length = max([len(molecule_full_id) for molecule_full_id in all_molecule_full_ids])

        # Add molecule annotations
        atom_array.add_annotation("molecule_full_id", dtype=f"<U{max_molecule_id_length}")
        atom_array.add_annotation("molecule_id", dtype=f"<U{max_molecule_id_length}")
        for molecule_full_id in all_molecule_full_ids:
            # Molecule full ID
            chain_full_ids = molecule_full_id.split(",")
            atom_array.molecule_full_id[np.isin(atom_array.chain_full_id, chain_full_ids)] = molecule_full_id

            # Molecule ID
            molecule_id = ",".join([chain_full_id.split("_")[0] for chain_full_id in chain_full_ids])
            atom_array.molecule_id[np.isin(atom_array.chain_id, chain_full_ids)] = molecule_id

        # Filter to only the molecules of interest
        data["atom_array"] = atom_array[atom_array.molecule_full_id != ""]
        return data


class AddChainTypeAnnotation(Transform):
    """
    Adds the ChainType annotation (according to the ChainType enum) to the AtomArray.
    """

    def check_input(self, data: dict):
        check_contains_keys(data, ["atom_array", "chain_info"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["chain_id"])

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        chain_info_dict = data["chain_info"]
        data["atom_array"] = DataPreprocessor.add_chain_type_annotation(atom_array, chain_info_dict)
        return data


class RemoveUnsupportedChainTypes(Transform):
    """
    Filter out chains with unsupported chain types from the AtomArray.
    Additionally, asserts that none of the query molecules are of an unsupported chain type (in which case they should have been filtered out upstream, otherwise our example is not valid).
    NOTE: This transform should be used after the FilterToMoleculesAndAnnotate transform as well as the AddChainTypeAnnotation transform.
    """

    # At the time of writing, the supported chain types are: NON_POLYMER, POLYPEPTIDE_L, DNA, RNA
    SUPPORTED_CHAIN_TYPES = [ChainType.NON_POLYMER, ChainType.POLYPEPTIDE_L, ChainType.DNA, ChainType.RNA]

    def check_input(self, data: dict):
        check_contains_keys(data, ["atom_array", "query_molecule_full_ids", "pdb_id"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["chain_type", "molecule_full_id"])

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        # We first assert that none of the query molecules are of an unsupported chain type, which means the example should have been filtered out upstream
        query_molecule_chain_types = np.unique(
            atom_array.chain_type[np.isin(atom_array.molecule_full_id, data["query_molecule_full_ids"])]
        )
        assert np.all(
            np.isin(query_molecule_chain_types, self.SUPPORTED_CHAIN_TYPES)
        ), f"{data['pdb_id']}: Query molecules has an unsupported chain type: {query_molecule_chain_types}"

        # Then, we filter out chains with unsupported chain types
        data["atom_array"] = atom_array[np.isin(atom_array.chain_type, self.SUPPORTED_CHAIN_TYPES)]
        return data


class RemoveHydrogens(Transform):
    """
    Remove hydrogens from the atom array.
    """

    def __init__(self, hydrogen_names: tuple | list = ("1", 1, "H", "D", "T")):
        self.hydrogen_names = hydrogen_names

    def check_input(self, data: dict):
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        is_heavy = ~np.isin(atom_array.element, self.hydrogen_names)
        data["atom_array"] = atom_array[is_heavy]
        return data


if __name__ == "__main__":
    #
    # TEMPORARY CODE BELOW FOR DEMONSTRATION
    #
    import os

    from cifutils.cifutils_biotite.cifutils_biotite import CIFParser

    from rf2aa.chemical import ChemicalData, initialize_chemdata

    initialize_chemdata()
    chemdata = ChemicalData()

    def get_digs_path(pdbid: str) -> str:
        pdbid = pdbid.lower()
        filename = f"/databases/rcsb/cif/{pdbid[1:3]}/{pdbid}.cif.gz"
        if not os.path.exists(filename):
            raise ValueError(f"File {filename} does not exist")
        return filename

    parser = CIFParser()
    data = parser.parse(
        get_digs_path("5ocm"),
        convert_mse_to_met=True,
        remove_waters=True,
        remove_crystallization_aids=True,
        build_assembly="first",
    )

    # fmt: off
    pipeline = Compose([
        RemoveHydrogens(), 
        LogData(depth=1),
        ToRF2Atom36Encoding(chemdata, default_coord=float("nan")),  # can also e.g. take default_coord=np.zeros(3)
        LogData(depth=1),
        RemoveKeys(["atom_array"], require_keys_exist=False),  # just for demo
        LogData(depth=1),
        SubsetToKeys(["xyz"]),
        LogData(depth=1),
        AddData({"mock": "mock"}, allow_overwrite=False),
        LogData(depth=1),
    ])
    # fmt: on

    data = pipeline(data)
