import numpy as np
from atomworks.common import sum_string_arrays
from atomworks.io.utils.io_utils import to_cif_file
from atomworks.ml.transforms.center_random_augmentation import CenterRandomAugmentation
from biotite.structure import AtomArrayStack
from rfd3.trainer.rfd3 import _reassign_unindexed_token_chains
from rfd3.transforms.design_transforms import (
    MotifCenterRandomAugmentation,
)


def pipe_out_to_file(output, save=True):
    atom_array = output["atom_array"]

    xyz = output["coord_atom_lvl_to_be_noised"]
    idxs = np.argsort(output["t"].numpy())
    eps = output["noise"].numpy()[idxs]
    eps[0] = eps[0] * 0
    x = AtomArrayStack(xyz.shape[0], xyz.shape[1])
    x.coord = xyz[idxs].numpy() + eps

    x.set_annotation("chain_id", ["A"] * xyz.shape[1])
    x.set_annotation("atom_name", [f"C{i}" for i in range(x.shape[-1])])
    x.set_annotation("res_id", output["feats"]["atom_to_token_map"])
    x.set_annotation("element", ["C"] * x.shape[-1])
    x.set_annotation("res_name", [atom_array.res_name[i] for i in range(x.shape[-1])])

    if save:
        f = f"{output.get('example_id', 'example')}_debug_out.cif"
        to_cif_file(
            x,
            f,
            id="x",
        )
        print("Saved cif file to:", f)
    else:
        return x


def save_pipe_out(atom_array):
    atom_array = _reassign_unindexed_token_chains(atom_array)

    f = "debug_out.cif"
    to_cif_file(
        atom_array,
        f,
        id="x",
    )
    print("Saved cif file to:", f)


def to_debug_pipe(pipe):
    pipe.transforms = [
        t
        for t in pipe.transforms
        if not isinstance(t, (CenterRandomAugmentation, MotifCenterRandomAugmentation))
    ]
    return pipe


# Allows to use atom-array whenever debugging by removing friction in atoms having the same identifiers
def save_debug_cif(atom_array, filepath, name="debug_out.cif"):
    dummy_array = atom_array.copy()
    dummy_array.chain_id = sum_string_arrays(
        dummy_array.chain_id, "-", dummy_array.transformation_id.astype(str)
    )

    f = filepath + name
    to_cif_file(
        dummy_array,
        f,
    )
    print("Saved cif file to:", f)
