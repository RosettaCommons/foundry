import biotite.structure as struct
import numpy as np
from datahub.preprocessing.constants import ChainType
from datahub.transforms.base import Transform

MIROR_IMAGE_MAPPING = {
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


class RandomlyMirrorInputs(Transform):
    """
    This component reflects inputs with a user-provided probability.

    Only protein and ligand comonents are reflected, nucleic acids are not.
    """

    def __init__(
        self,
        mirror_prob=0.0,
    ):
        self.mirror_prob = mirror_prob

    def forward(self, data: dict) -> dict:
        assert not data.get("is_inference", False)
        atom_array = data["atom_array"]

        if (
            (atom_array.chain_type == ChainType.DNA).any()
            or (atom_array.chain_type == ChainType.RNA).any()
            or (atom_array.chain_type == ChainType.DNA_RNA_HYBRID).any()
        ):
            return data

        if np.random.rand() > self.mirror_prob:
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
            if resname in MIROR_IMAGE_MAPPING:
                atom_array.res_name[r_i:r_j] = MIROR_IMAGE_MAPPING[resname]
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

        return data
