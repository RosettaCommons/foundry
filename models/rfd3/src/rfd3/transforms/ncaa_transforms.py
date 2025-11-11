import biotite.structure as struct
import numpy as np
import torch
from atomworks.ml.preprocessing.constants import ChainType
from atomworks.ml.transforms._checks import check_contains_keys
from atomworks.ml.transforms.base import Transform
from rfd3.transforms.conditioning_base import (
    convert_existing_annotations_to_bool,
)

MIRROR_IMAGE_MAPPING = {
    "ALA": "DAL",
    "SER": "DSN",
    "CYS": "DCY",
    "PRO": "DPR",
    "VAL": "DVA",
    "THR": "DTH",
    "LEU": "DLE",
    "ILE": "DIL",
    "ASN": "DSG",
    "ASP": "DAS",
    "MET": "MED",
    "GLN": "DGN",
    "GLU": "DGL",
    "LYS": "DLY",
    "HIS": "DHI",
    "PHE": "DPN",
    "ARG": "DAR",
    "TYR": "DTY",
    "TRP": "DTR",
    "GLY": "GLY",
}

D_TO_L_MAPPING = {v: k for k, v in MIRROR_IMAGE_MAPPING.items() if k != "GLY"}

TWO_WAY_MIRROR_IMAGE_MAPPING = {**MIRROR_IMAGE_MAPPING, **D_TO_L_MAPPING}

D_AA = [aa for aa in MIRROR_IMAGE_MAPPING.values() if aa != "GLY"]


class RandomlyMirrorInputs(Transform):
    """
    This component reflects inputs with a user-provided probability.

    Only protein and ligand comonents are reflected, nucleic acids are not.
    """

    def forward(self, data: dict) -> dict:
        assert not data.get("is_inference", False)
        mirror_input = data["conditions"].get("mirror_input", False)
        atom_array = data["atom_array"]

        if (
            (atom_array.chain_type == ChainType.DNA).any()
            or (atom_array.chain_type == ChainType.RNA).any()
            or (atom_array.chain_type == ChainType.DNA_RNA_HYBRID).any()
        ):
            return data

        if not mirror_input:
            return data

        renamed_map = {}
        res_starts = struct.get_residue_starts(atom_array)
        for i, r_i in enumerate(res_starts):
            if i == len(res_starts) - 1:
                r_j = len(atom_array)
            else:
                r_j = res_starts[i + 1]

            # case 1: standard AA
            resname = atom_array.res_name[r_i]
            if resname in TWO_WAY_MIRROR_IMAGE_MAPPING:
                atom_array.res_name[r_i:r_j] = TWO_WAY_MIRROR_IMAGE_MAPPING[resname]
            # case 2: non-standard AA or ligand with >=4 atoms
            elif r_j - r_i >= 3:
                if resname in renamed_map:
                    newname = renamed_map[resname]
                else:
                    newname = "L:" + str(len(renamed_map))
                    renamed_map[resname] = newname
                atom_array.res_name[r_i:r_j] = newname

        # flip coords about Z
        atom_array.coord = atom_array.coord * np.array([1, 1, -1.0])

        xyz = data.get("coord_atom_lvl_to_be_noised", None)
        if xyz is not None:
            # flip coords about Z
            data["coord_atom_lvl_to_be_noised"] = xyz * torch.tensor(
                [1, 1, -1], dtype=xyz.dtype, device=xyz.device
            )
        ground_truth_coord = (
            data["ground_truth"].get("coord_atom_lvl", None)
            if "ground_truth" in data
            else None
        )
        if ground_truth_coord is not None:
            # flip coords about Z
            data["ground_truth"]["coord_atom_lvl"] = ground_truth_coord * torch.tensor(
                [1, 1, -1],
                dtype=ground_truth_coord.dtype,
                device=ground_truth_coord.device,
            )

        return data


class AddIsDAminoAcidFeat(Transform):
    """
    Adds an annotation to the atom array indicating whether each residue is a D-amino acid.
    """

    def check_input(self, data) -> None:
        check_contains_keys(data, ["atom_array", "feats"])

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        # Check if there is already an annotation for D-amino acids
        if "is_d_amino_acid" not in atom_array.get_annotation_categories():
            # Check if the res_name is in the D-amino acid set
            is_d_aa = np.isin(atom_array.res_name, D_AA)
            # Create a new annotation for D-amino acids

            glycines = atom_array.res_name == "GLY"
            # half the time, we will set glycine to be D-glycine
            is_d_aa = np.logical_or(
                is_d_aa, np.logical_and(glycines, np.random.rand(len(glycines)) < 0.5)
            )

            atom_array.set_annotation(
                "is_d_amino_acid",
                is_d_aa,
            )

        # Add feature for is_d_amino_acid
        if "is_d_amino_acid" not in data["feats"]:
            is_d_amino_acid = atom_array.get_annotation("is_d_amino_acid")
            data["feats"]["is_d_amino_acid"] = is_d_amino_acid

        data["atom_array"] = atom_array

        return data


class StrtoBoolforIsDAminoAcidFeature(Transform):
    def forward(self, data):
        atom_array = data["atom_array"]
        convert_existing_annotations_to_bool(
            atom_array, annotations=["is_d_amino_acid"]
        )
        data["atom_array"] = atom_array
        return data
