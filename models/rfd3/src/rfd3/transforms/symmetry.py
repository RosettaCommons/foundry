import numpy as np
import torch
from atomworks.ml.transforms.base import Transform
from rfd3.inference.symmetry.frames import (
    framecoords_to_RTs,
    unpack_vector,
)


class AddSymmetryFeats(Transform):
    """
    Add atom_array symmetry features to the data features.
    Arguments:
        symmetry_features: The atom_array symmetry features to add to the data features.
    Returns:
        data: The data with the atom_array symmetry features added to the data features.
    """

    def __init__(
        self,
        symmetry_features=[
            "sym_transform_id",
            "sym_entity_id",
            "is_sym_asu",
        ],
    ):
        self.symmetry_feats = symmetry_features

    def forward(self, data):
        atom_array = data["atom_array"]
        # Get frames from atom_array
        transforms_dict = self.make_transforms_dict(atom_array)
        data["feats"]["sym_transform"] = transforms_dict  # {str(id): tuple (R,T)}
        # Else, add symmetry features atomwise
        for feature_name in self.symmetry_feats:
            feature_array = atom_array.get_annotation(feature_name)
            data["feats"][feature_name] = feature_array
        return data

    def make_transforms_dict(self, atom_array):
        transforms_dict = {}
        # get decomposed frames from atom array (unpacking the vectorized frames)
        Oris = torch.tensor(
            [
                np.asarray(unpack_vector(Ori)).tolist()
                for Ori in atom_array.get_annotation("sym_transform_Ori")
            ]
        )
        Xs = torch.tensor(
            [
                np.asarray(unpack_vector(X)).tolist()
                for X in atom_array.get_annotation("sym_transform_X")
            ]
        )
        Ys = torch.tensor(
            [
                np.asarray(unpack_vector(Y)).tolist()
                for Y in atom_array.get_annotation("sym_transform_Y")
            ]
        )
        TIDs = torch.from_numpy(atom_array.get_annotation("sym_transform_id"))

        Oris = torch.unique_consecutive(Oris, dim=0)
        Xs = torch.unique_consecutive(Xs, dim=0)
        Ys = torch.unique_consecutive(Ys, dim=0)
        TIDs = torch.unique_consecutive(TIDs, dim=0)
        # the case in which there is only rotation (no translation), Ori = [0,0,0]
        if len(Oris) == 1 and (Oris == 0).all():
            Oris = Oris.repeat(len(Xs), 1)
        Rs, Ts = framecoords_to_RTs(Oris, Xs, Ys)

        for R, T, transform_id in zip(Rs, Ts, TIDs):
            if transform_id.item() == -1:
                continue
            transforms_dict[str(transform_id.item())] = (R, T)
        return transforms_dict
