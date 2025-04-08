import os

import torch

from modelhub.pymol import cmd
from modelhub.util import writepdb


def clear():
    cmd.reinitialize("everything")
    cmd.delete("all")
    # cmd.do(f'cd {REPO_DIR}/pymol_config')
    cmd.do("@./pymolrc")


# def pseudoatom(
#         pos: list = [0,0,0],
#         label='origin',
#         ):
#     cmd.pseudoatom(label,'', 'PS1','PSD', '1', 'P',
#         'PSDO', 'PS', -1.0, 1, 0.0, 0.0, '',
#         '', pos)
#     # cmd.do(f'label {label}, "{label}"')
#     return label


def pseudoatom(
    cmd,
    pos: list = [0, 0, 0],
    label="origin",
):
    cmd.pseudoatom(
        label, "", "PS1", "PSD", "1", "P", "PSDO", "PS", -1.0, 1, 0.0, 0.0, "", "", pos
    )
    # cmd.do(f'label {label}, "{label}"')
    return label


def show_origin():
    pa = pseudoatom(cmd, label="the_origin")
    cmd.center(pa)
    cmd.color("red", pa)
    cmd.set("grid_slot", -2, pa)


def show_pymol(true_crds, seq, bond_feats, label="unlabeled"):
    pdb_path = "tmp/true_0.pdb"
    writepdb(
        pdb_path,
        true_crds,
        seq.long(),
        bond_feats=bond_feats[None],
    )
    cmd.load(os.path.abspath(pdb_path), label)
    show_origin()
    return label


def to_atom37(X_L, atom_mask):
    assert X_L.shape[-1] == 3
    assert X_L.numel() / 3 == atom_mask.sum(), (
        f"{X_L.numel()/3=} != {atom_mask.sum()=}.  {X_L.shape=} {atom_mask.shape=}"
    )
    L, _ = atom_mask.shape[-2:]
    X_I = (
        torch.zeros(atom_mask.shape + (3,), dtype=torch.float, device=atom_mask.device)
        - 10
    )
    X_I[atom_mask] = X_L
    return X_I
