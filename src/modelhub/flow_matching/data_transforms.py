import torch
from opt_einsum import contract as einsum

from modelhub.chemical import ChemicalData as ChemData
from modelhub.util import get_prot_sm_mask, rigid_from_3_points

"""
the flow matching code in frame flow uses openfold primitives which are slightly 
different from their rosettafold counterparts
"""


def convert_dataloader_inputs_to_rigids(inputs, device):
    (
        seq,
        msa,
        msa_masked,
        msa_full,
        mask_msa,
        true_crds,
        mask_crds,
        idx_pdb,
        xyz_t,
        t1d,
        mask_t,
        xyz_prev,
        mask_prev,
        same_chain,
        unclamp,
        negative,
        atom_frames,
        bond_feats,
        dist_matrix,
        chirals,
        ch_label,
        symmgp,
        task,
        item,
    ) = inputs
    if len(true_crds.shape) == 4:
        true_crds = true_crds[None]
    if len(mask_crds.shape) == 3:
        mask_crds = mask_crds[None]

    rotmats, trans = xyz_to_rigids(true_crds, mask_crds)
    seq_unmasked = msa[:, 0, 0][0]
    res_mask = get_prot_sm_mask(mask_crds, seq_unmasked)[0][
        0
    ].long()  # reduce dimension to (L)
    diffuse_mask = torch.ones_like(res_mask).long()
    diffuse_mask_seq = torch.zeros_like(res_mask).long()
    batch = {
        "aatypes_1": seq_unmasked[None],
        "rotmats_1": rotmats[None],
        "trans_1": trans[None],
        "res_mask": res_mask[None],
        "diffuse_mask": diffuse_mask[None],
        "diffuse_mask_seq": diffuse_mask_seq[None],
    }
    return to_device(batch, device)


def xyz_to_rigids(xyz, mask):
    """
    convert xyz to rigid transforms (ru.Rigid)
    """
    # remove symmetry dimension
    xyz = xyz[0][0]
    mask = mask[0][0]

    xyz = center_chain_backbone(
        xyz[..., :3, :], mask[..., :3]
    )  # center backbone at origin
    N, Ca, C = xyz[..., 0, :], xyz[..., 1, :], xyz[..., 2, :]
    rotmats, trans = rigid_from_3_points(N, Ca, C)
    return rotmats, trans


def rigids_to_xyz(rotmats, trans):
    """
    convert rigid transforms to backbone xyz
    """
    L = rotmats.shape[1]
    init_coords = (
        ChemData()
        .INIT_CRDS[None, None]
        .repeat(1, L, 1, 1)[..., :3, :]
        .to(rotmats.device)
    )
    xyz = einsum("blij,blaj->blai", rotmats, init_coords) + trans[:, :, None]
    return xyz


def center_chain_backbone(xyz, mask):
    """
    center N,Ca,C at origin
    """
    assert len(xyz.shape) == 3
    assert len(mask.shape) == 2
    assert xyz.shape[-2] == 3
    assert mask.shape[-1] == 3

    xyz_allatom = xyz[mask]
    xyz = xyz - xyz_allatom.mean(0)
    assert len(xyz.shape) == 3
    return xyz


def get_unbatched_backbone_coords(xyz):
    """
    get unbatched backbone coords
    """
    if len(xyz.shape) == 5:
        xyz = xyz[0][0]
    else:
        xyz = xyz[0]
    return xyz[..., :3, :]


def to_device(batch, device):
    for k, v in batch.items():
        batch[k] = v.to(device, non_blocking=True)
    return batch
