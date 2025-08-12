import logging

import numpy as np
import torch

from modelhub.chemical import ChemicalData as ChemData
from modelhub.kinematics import get_ang, get_dih
from modelhub.scoring import HbHybType

logger = logging.getLogger(__name__)

# Loss functions for the training
# 1. BB rmsd loss
# 2. distance loss (or 6D loss?)
# 3. bond geometry loss
# 4. predicted lddt loss


# resolve "equivalent" natives
def resolve_equiv_natives(xs, natstack, maskstack):
    if len(natstack.shape) == 4:
        return natstack, maskstack
    if natstack.shape[1] == 1:
        return natstack[:, 0, ...], maskstack[:, 0, ...]
    dx = torch.norm(xs[:, None, :, None, 1, :] - xs[:, None, None, :, 1, :], dim=-1)
    dnat = torch.norm(
        natstack[:, :, :, None, 1, :] - natstack[:, :, None, :, 1, :], dim=-1
    )
    delta = torch.sum(torch.abs(dnat - dx), dim=(-2, -1))
    return natstack[:, torch.argmin(delta), ...], maskstack[:, torch.argmin(delta), ...]


def resolve_equiv_natives_asmb(xyz_pred, xyz_true, mask, ch_label, Ls_prot, Ls_sm):
    """Resolves multiple chain and atom permutations of a protein-ligand assembly to a
    single set of true coordinates with the lowest C-alpha distance error to predicted
    coordinates. Protein chains are assigned to the chain permutation with the lowest
    distance error. Ligand chains are assigned using a greedy search to minimize the
    distance error within that ligand chain and between it and the already-assigned
    protein chains.

    Parameters
    ----------
    xyz_pred: tensor (B, L, N_atoms, 3) with predicted coordinates for all chains, where
        the total number of residues is L = sum(Ls_prot)+sum(Ls_sm)
    xyz_true: tensor (B, N_perm, L, N_atoms, 3) with true coordinates, with chain and
        atom permutations in dimension 1. For protein chains, all chain permutations are
        enumerated. For ligand chains, only atom permutations within each chain are
        enumerated. Ligand chain swaps will be handled by a greedy search below.
    mask: tensor (B, N_perm, L, N_atoms) with boolean mask for whether atoms exist in
        `xyz_true`. Some ligand chains have fewer sets of atom permutations than `N_perm`.
        This is indicated by all False values in the relevant entries in `mask`.
    ch_label: tensor (B, L) with integer labels for each unique chain, assigned to each
        residue. Used to determine which ligand chains are equivalent and can be
        considered for alternate ligand chain assignments.
    Ls_prot: list of lengths of protein chains
    Ls_sm: list lengths of ligand chains. All ligand chains come after all protein chains

    Returns
    -------
    xyz_out: tensor (B, L, N_atoms, 3) with best true coordinates for the prediction.
    mask_out: tensor (B, L, N_atoms) with corresponding atom mask for best true
        coordinates.
    """
    if len(xyz_true.shape) == 4:
        return xyz_true, mask

    batch_size = xyz_pred.shape[0]
    assert batch_size == 1, "this function does not work if B!=1"
    total_protein_length = sum(Ls_prot)
    xyz_out = torch.full(
        xyz_pred.shape, torch.nan, device=xyz_pred.device, dtype=xyz_pred.dtype
    )
    mask_out = torch.full(
        xyz_pred.shape[:3], False, device=xyz_pred.device, dtype=torch.bool
    )

    # Step 1: choose true protein chain permutation
    # with lowest distance error to prediction via CA-CA distances.
    # This assumes that all possible protein pairs are enumerated
    # in all possible combinations. For example, if we have three
    # identical chains A1, A2 and A3 and another chain B,
    # this assumes that the tensor enumerate chains like so:
    # [A1, A2, A3, B]
    # [A1, A3, A2, B]
    # [A2, A1, A3, B]
    # [A2, A3, A1, B]
    # [A3, A1, A2, B]
    # [A3, A2, A1, B]

    pred_ca_ca_distances = torch.norm(
        xyz_pred[:, None, :total_protein_length, None, 1, :]
        - xyz_pred[:, None, None, :total_protein_length, 1, :],
        dim=-1,
    )

    valid_protein_permutations = torch.any(
        torch.any(mask[:, :, :total_protein_length], dim=2), dim=2
    )  # only take distances over valid protein permutations
    xyz_true_valid_prot = xyz_true[
        :, valid_protein_permutations[0], :total_protein_length
    ]  ## rk assumes B==1
    mask_valid_prot = mask[
        :, valid_protein_permutations[0], :total_protein_length
    ]  ## rk assumes B==1
    true_ca_ca_distances = torch.norm(
        xyz_true_valid_prot[:, :, :, None, 1, :]
        - xyz_true_valid_prot[:, :, None, :, 1, :],
        dim=-1,
    )

    # Note: we need to account for different masking patterns in this function. Basically,
    # every copy of the SAME protein chain may have a different number of resolved CA atoms,
    # so we need to separately index each copy by its respective atom mask.
    mask_valid_prot_ca = mask_valid_prot[:, :, :, 1]
    mask_valid_prot_ca_ca = (
        mask_valid_prot_ca[:, :, :, None] * mask_valid_prot_ca[:, :, None, :]
    )
    num_valid_ca_ca_distances = torch.sum(mask_valid_prot_ca_ca, dim=(-2, -1))
    # Another note: if there are 0 valid ca atoms for a given symmetry instance,
    # torch handles 0 division by returning torch.inf, which should give the correct behavior.
    # We never want to pick a chain with 0 valid ca atoms.

    pred_true_ca_ca_dist_total = (
        torch.abs(true_ca_ca_distances - pred_ca_ca_distances) * mask_valid_prot_ca_ca
    )
    pred_true_ca_ca_dist_diff = torch.sum(pred_true_ca_ca_dist_total, dim=(-2, -1))
    pred_true_ca_ca_dist_diff = torch.nan_to_num(
        pred_true_ca_ca_dist_diff, nan=torch.inf
    )
    pred_true_ca_ca_dist_diff = pred_true_ca_ca_dist_diff / num_valid_ca_ca_distances
    best_protein_perm_index_per_batch = torch.argmin(
        pred_true_ca_ca_dist_diff, axis=-1
    )  # indices over indices of xyz_true_valid_prot

    best_protein_coords = xyz_true_valid_prot[
        torch.arange(batch_size),
        best_protein_perm_index_per_batch,
    ]
    matching_protein_mask = mask_valid_prot[
        torch.arange(batch_size),
        best_protein_perm_index_per_batch,
    ]
    xyz_out[:, :total_protein_length] = best_protein_coords
    mask_out[:, :total_protein_length] = matching_protein_mask

    # Step 2: match ligands greedily to protein chains.
    # Each of the following dictionaries takes in a ligand
    # "label", e.g. chain label from ch_label, and maps it to
    # all of the possible coordinates that belong to that label.
    # ligand_offsets refers to the length offset of each coordinate
    # in the original xyz tensor, while symmetry offsets
    # refers to the permutation index of the coordinate in the original tensor.
    label_to_ligand_coordinates = {}  # unrolled coordinates for each unique ligand label
    label_to_ligand_offsets = {}  # indexing for to get coords for this chosen sm in length dimension
    label_to_symmetry_offsets = {}  # indexing for to get coords for this chosen sm in symm dimension
    label_to_position_index = {}  # holds information on alternative automorphs for same ligand
    label_to_ligand_mask = {}  # unrolled mask for each of the atoms in label_to_ligand_coordinates
    label_to_num_copies_same = {}  # number of copies of the same ligand

    running_offset = total_protein_length
    # This first for loop groups ligand coordinates by
    # "chain" label, e.g. groups by identical ligands. It assumes
    # that identical ligands are of the same length.
    for position_index, ligand_length in enumerate(Ls_sm):
        # Note: this functionality assumes a batch size of 1.
        ligand_label = ch_label[0, running_offset].item()
        ligand_offsets = [running_offset, running_offset + ligand_length]
        ligand_coordinates = xyz_true[:, :, ligand_offsets[0] : ligand_offsets[1]]
        ligand_mask = mask[:, :, ligand_offsets[0] : ligand_offsets[1], 1]
        # any across coordinates (x, y, z) and length
        ligand_mask_any_atoms = ligand_mask.any(dim=-1)  # B, NSymm
        ligand_coordinates_valid = ligand_coordinates[ligand_mask_any_atoms][
            None
        ]  # assumes B=1
        ligand_mask_valid = ligand_mask[ligand_mask_any_atoms][None]

        # NOTE: assumes that the first N dimensions are real coordinates and the rest are buffer
        num_valid_ligand_perms = torch.sum(ligand_mask_any_atoms)
        symmetry_offsets = torch.arange(
            num_valid_ligand_perms, device=ligand_coordinates_valid.device
        )
        ligand_offsets_tensor = (
            torch.tensor(ligand_offsets, device=ligand_coordinates_valid.device)
            .reshape(1, 2)
            .repeat(num_valid_ligand_perms, 1)
        )
        position_index_tensor = torch.full(
            (num_valid_ligand_perms,),
            position_index,
            device=ligand_coordinates_valid.device,
        )
        if ligand_label in label_to_ligand_coordinates:
            label_to_ligand_coordinates[ligand_label].append(ligand_coordinates_valid)
            label_to_ligand_offsets[ligand_label].append(ligand_offsets_tensor)
            label_to_symmetry_offsets[ligand_label].append(symmetry_offsets)
            label_to_position_index[ligand_label].append(position_index_tensor)
            label_to_ligand_mask[ligand_label].append(ligand_mask_valid)
            label_to_num_copies_same[ligand_label] += 1
        else:
            label_to_ligand_coordinates[ligand_label] = [ligand_coordinates_valid]
            label_to_ligand_offsets[ligand_label] = [ligand_offsets_tensor]
            label_to_symmetry_offsets[ligand_label] = [symmetry_offsets]
            label_to_position_index[ligand_label] = [position_index_tensor]
            label_to_ligand_mask[ligand_label] = [ligand_mask_valid]
            label_to_num_copies_same[ligand_label] = 1

        running_offset += ligand_length
    # This second for loop just stacks the accumulated tensors from the last
    # for loop into a single tensor
    for ligand_label in torch.unique(ch_label[0, total_protein_length:]):
        ligand_label = ligand_label.item()
        label_to_ligand_coordinates[ligand_label] = torch.cat(
            label_to_ligand_coordinates[ligand_label], dim=1
        )
        label_to_ligand_offsets[ligand_label] = torch.cat(
            label_to_ligand_offsets[ligand_label], dim=0
        )
        label_to_symmetry_offsets[ligand_label] = torch.cat(
            label_to_symmetry_offsets[ligand_label], dim=0
        )
        label_to_position_index[ligand_label] = torch.cat(
            label_to_position_index[ligand_label], dim=0
        )
        label_to_ligand_mask[ligand_label] = torch.cat(
            label_to_ligand_mask[ligand_label], dim=1
        )

    # This third for loop is super niche: if there
    # are two copies of the same ligand, and FURTHER
    # loaded copies of that ligand from the additional
    # cif loading code that Pascal wrote, those will be
    # stacked in the symmetry dimension of the FIRST ligand.
    # Every other copy of the same ligand will have only
    # n_symm copies, but the first will have n_symm * n_additional_copies.
    # The position index for each of these copies should only pertain
    # to automorphs, so we need to adjust the position index in this case.
    # So if position_index_tensor is something like [0, 0, 0, 0, 0, 0, 1, 1]
    # we actually want it to be [0, 0, 1, 1, 2, 2, 3, 3].
    for ligand_label, position_index_tensor in label_to_position_index.items():
        if label_to_num_copies_same[ligand_label] == 1:
            continue

        unique_positions, counts = torch.unique(
            position_index_tensor, return_counts=True
        )
        if torch.unique(counts).shape[0] == 1:
            continue

        min_count = torch.min(counts).item()
        max_count = torch.max(counts).item()
        if position_index_tensor.shape[0] % min_count != 0:
            continue
        if max_count % min_count != 0:
            continue

        num_count_copies = int(position_index_tensor.shape[0] / min_count)
        new_position_index_tensor = torch.arange(
            num_count_copies,
            dtype=position_index_tensor.dtype,
            device=position_index_tensor.device,
        )
        new_position_index_tensor = new_position_index_tensor.repeat_interleave(
            min_count
        )
        label_to_position_index[ligand_label] = new_position_index_tensor
    # This final for loop does the bulk of the computation:
    # for each ligand, it computes the ca-prot-lig distance
    # and all possible positions of the ligand as gathered
    # in the previous for loop. It picks the best possible position
    # for that ligand, and then removes that ligand position and
    # all of its associated isomorphs (but not other identical ligand positions)
    # from the batch of possible positions.
    xyz_true_prot_ca = xyz_out[:, :total_protein_length, 1]
    xyz_pred_prot_ca = xyz_pred[:, :total_protein_length, 1]
    running_offset = total_protein_length
    for ligand_length in Ls_sm:
        ligand_label = ch_label[0, running_offset].item()
        xyz_pred_query_lig_ca = xyz_pred[
            :, running_offset : running_offset + ligand_length, 1
        ]

        xyz_pred_prot_and_select_lig_ca = torch.cat(
            [xyz_pred_prot_ca, xyz_pred_query_lig_ca], dim=1
        )  # (B, L_prot + L_query_lig, 3)
        pred_prot_lig_ca_distance = torch.cdist(
            xyz_pred_prot_and_select_lig_ca,
            xyz_pred_query_lig_ca,
            compute_mode="donot_use_mm_for_euclid_dist",
        )  # (B, L_prot+ L_query_lig, L_query_lig)

        xyz_true_lig_ca = label_to_ligand_coordinates[ligand_label][:, :, :, 1]
        xyz_true_prot_and_select_lig_ca = torch.cat(
            [
                xyz_true_prot_ca[None].repeat((1, xyz_true_lig_ca.shape[1], 1, 1)),
                xyz_true_lig_ca,
            ],
            dim=2,
        )  # (B, Nsymm*Nrepeats, L_prot+ L_query_lig, 3)
        all_true_prot_lig_ca_distance = torch.cdist(
            xyz_true_prot_and_select_lig_ca,
            xyz_true_lig_ca,
            compute_mode="donot_use_mm_for_euclid_dist",
        )  # (B, Nsymm*Nrepeats, L_prot+ L_query_lig, L_query_lig)
        # Note: not all ligand atoms may be resolved, and the pattern of resolution
        # may be DIFFERENT for different copies of the same ligand. We need
        # to apply a per symmetry mask to account for resolved ligand atoms.
        ligand_mask = label_to_ligand_mask[ligand_label]
        expanded_ligand_mask = ligand_mask[:, :, None].repeat(
            1, 1, all_true_prot_lig_ca_distance.shape[2], 1
        )

        # The only valid distances are those for which both the ligand atoms are resolved
        # AND the corresponding protein CA coordinates are also resolved, so we have
        # to account for the protein mask as well here.
        expanded_ligand_mask[:, :, :total_protein_length, :] = (
            expanded_ligand_mask[:, :, :total_protein_length, :]
            * matching_protein_mask[:, None, :, 1, None]
        )

        num_valid_distances_resolved = torch.sum(expanded_ligand_mask, dim=(-2, -1))
        pred_true_lig_prot_dist_diff = (
            torch.abs(pred_prot_lig_ca_distance - all_true_prot_lig_ca_distance)
            * expanded_ligand_mask
        )
        pred_true_lig_prot_dist_diff = torch.sum(
            pred_true_lig_prot_dist_diff, dim=(-2, -1)
        )
        pred_true_lig_prot_dist_diff = torch.nan_to_num(
            pred_true_lig_prot_dist_diff, nan=torch.inf
        )
        pred_true_lig_prot_dist_diff = (
            pred_true_lig_prot_dist_diff / num_valid_distances_resolved
        )
        best_ligand_dist_index = torch.argmin(pred_true_lig_prot_dist_diff)
        true_selected_lig_offsets = label_to_ligand_offsets[ligand_label][
            best_ligand_dist_index
        ]
        true_selected_symmetry_offset = label_to_symmetry_offsets[ligand_label][
            best_ligand_dist_index
        ]

        xyz_out[:, running_offset : running_offset + ligand_length] = xyz_true[
            :,
            true_selected_symmetry_offset,
            true_selected_lig_offsets[0] : true_selected_lig_offsets[1],
        ]
        mask_out[:, running_offset : running_offset + ligand_length] = mask[
            :,
            true_selected_symmetry_offset,
            true_selected_lig_offsets[0] : true_selected_lig_offsets[1],
        ]

        position_index_tensor = label_to_position_index[ligand_label]
        chosen_position = position_index_tensor[best_ligand_dist_index]
        remove_chosen_mask = position_index_tensor != chosen_position

        label_to_ligand_coordinates[ligand_label] = label_to_ligand_coordinates[
            ligand_label
        ][:, remove_chosen_mask]
        label_to_ligand_offsets[ligand_label] = label_to_ligand_offsets[ligand_label][
            remove_chosen_mask
        ]
        label_to_symmetry_offsets[ligand_label] = label_to_symmetry_offsets[
            ligand_label
        ][remove_chosen_mask]
        label_to_position_index[ligand_label] = label_to_position_index[ligand_label][
            remove_chosen_mask
        ]
        label_to_ligand_mask[ligand_label] = label_to_ligand_mask[ligand_label][
            :, remove_chosen_mask
        ]
        running_offset += ligand_length
    return xyz_out, mask_out


def calc_rmsd(pred, true, mask):
    # pred (N,B,Lasu,natom,3)
    # true (B,Lasu,natom,3)
    # mask (B,Lasu,natom)
    def rmsd(V, W, eps=1e-4):
        L = V.shape[1]
        return torch.sqrt(torch.sum((V - W) * (V - W), dim=(1, 2)) / L + eps)

    def centroid(X):
        return X.mean(dim=-2, keepdim=True)

    N, B, L, Natm = pred.shape[:4]
    resmask = mask[0, :, 1]
    pred = pred[:, :, resmask, 1].squeeze(1)
    true = true[:, resmask, 1]
    cP = centroid(pred)
    cT = centroid(true)
    pred = pred - cP
    true = true - cT
    C = torch.einsum("bji,njk->bik", pred, true)
    V, S, W = torch.svd(C)
    d = torch.ones([N, 3, 3], device=pred.device)
    d[:, :, -1] = torch.sign(torch.det(V) * torch.det(W)).unsqueeze(1)
    U = torch.matmul(d * V, W.permute(0, 2, 1))  # (IB, 3, 3)
    rpred = torch.matmul(pred, U)  # (IB, L*3, 3)
    rms = rmsd(rpred, true).reshape(N)
    return rms, U, cP, cT


# torsion angle predictor loss
def torsionAngleLoss(alpha, alphanat, alphanat_alt, tors_mask, tors_planar, eps=1e-4):
    I = alpha.shape[0]
    alpha = alpha.float()
    lnat = torch.sqrt(torch.sum(torch.square(alpha), dim=-1) + eps)
    anorm = alpha / (lnat[..., None])

    l_tors_ij = torch.min(
        torch.sum(torch.square(anorm - alphanat[None]), dim=-1),
        torch.sum(torch.square(anorm - alphanat_alt[None]), dim=-1),
    )
    l_tors = torch.sum(l_tors_ij * tors_mask[None]) / (torch.sum(tors_mask) * I + eps)
    l_norm = torch.sum(torch.abs(lnat - 1.0) * tors_mask[None]) / (
        torch.sum(tors_mask) * I + eps
    )
    l_planar = torch.sum(torch.abs(alpha[..., 0]) * tors_planar[None]) / (
        torch.sum(tors_planar) * I + eps
    )

    return l_tors + 0.02 * l_norm + 0.02 * l_planar


def compute_FAPE(Rs, Ts, xs, Rsnat, Tsnat, xsnat, Z=10.0, dclamp=10.0, eps=1e-4):
    xij = torch.einsum("rji,rsj->rsi", Rs, xs[None, ...] - Ts[:, None, ...])
    xij_t = torch.einsum("rji,rsj->rsi", Rsnat, xsnat[None, ...] - Tsnat[:, None, ...])

    # torch.norm(xij-xij_t,dim=-1)
    diff = torch.sqrt(torch.sum(torch.square(xij - xij_t), dim=-1) + eps)

    loss = (1.0 / Z) * (torch.clamp(diff, max=dclamp)).mean()

    return loss


def compute_pae_loss(
    X,
    X_y,
    uX,
    Y,
    Y_y,
    uY,
    logit_pae,
    frame_mask,
    atom_mask,
    pae_bin_step=0.5,
    eps=1e-4,
    frame_atom_mask_2d=None,
):
    """Predicted Aligned Error: C-alpha (or sm. mol atom) distances in backbone frames from final layer"""
    frame_mask_bb = frame_mask[0, :, 0]  # valid backbone frames (L,)
    atom_mask_ca = atom_mask[0, :, 1]  # valid CA atoms (L,)

    xij_ca = torch.einsum(
        "fji,faj->fai",
        uX[-1, frame_mask_bb, 0],
        X[-1, None, atom_mask_ca, 1] - X_y[-1, frame_mask_bb, None, 0],
    )  # (N_valid_frames, N_valid_ca, 3)

    xij_ca_t = torch.einsum(
        "fji,faj->fai",
        uY[-1, frame_mask_bb, 0],
        Y[-1, None, atom_mask_ca, 1] - Y_y[-1, frame_mask_bb, None, 0],
    )  # (N_valid_frames, N_valid_ca, 3)

    eij_label = (
        torch.sqrt(torch.square(xij_ca - xij_ca_t).sum(dim=-1) + eps).clone().detach()
    )

    nbin = logit_pae.shape[1]
    pae_bins = torch.linspace(
        pae_bin_step,
        pae_bin_step * (nbin - 1),
        nbin - 1,
        dtype=logit_pae.dtype,
        device=logit_pae.device,
    )
    true_pae_label = torch.bucketize(eij_label, pae_bins, right=True).long()

    logit_pae_masked = logit_pae[:, :, frame_mask_bb][
        ..., atom_mask_ca
    ]  # (1, nbins, N_valid_frames, N_valid_ca)

    cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="none")(
        logit_pae_masked, true_pae_label[None]
    )  # assumes B=1

    if frame_atom_mask_2d is None:
        return torch.mean(cross_entropy_loss)
    else:
        # The following tensor should be 1 x num_valid_frames x num_valid_ca
        frame_atom_ca_only_mask = frame_atom_mask_2d[:, frame_mask_bb, :, :, :]
        frame_atom_ca_only_mask = frame_atom_ca_only_mask[:, :, :, atom_mask_ca, :]
        frame_atom_ca_only_mask = frame_atom_ca_only_mask[:, :, 0, :, 1]
        # We unsqueeze once in the number of bins dimension to do appropriate broadcasting
        frame_atom_ca_only_mask = frame_atom_ca_only_mask.unsqueeze(1)
        return torch.mean(frame_atom_ca_only_mask * cross_entropy_loss)


def compute_pde_loss(
    X, Y, logit_pde, atom_mask, pde_bin_step=0.3, frame_atom_mask_2d=None
):
    """Predicted Distance Error: C-alpha (or sm. mol atom) pairwise distances"""
    atom_mask_ca = atom_mask[0, :, 1]  # valid CA atoms (L,)

    dX = torch.cdist(
        X[-1, atom_mask_ca, 1],
        X[-1, atom_mask_ca, 1],
        compute_mode="donot_use_mm_for_euclid_dist",
    )
    dY = torch.cdist(
        Y[0, atom_mask_ca, 1],
        Y[0, atom_mask_ca, 1],
        compute_mode="donot_use_mm_for_euclid_dist",
    )
    dist_err = torch.abs(dX - dY).clone().detach()

    nbin = logit_pde.shape[1]
    pde_bins = torch.linspace(
        pde_bin_step,
        pde_bin_step * (nbin - 1),
        nbin - 1,
        dtype=logit_pde.dtype,
        device=logit_pde.device,
    )
    true_pde_label = torch.bucketize(dist_err, pde_bins, right=True).long()
    logit_pde_masked = logit_pde[:, :, atom_mask_ca][
        ..., atom_mask_ca
    ]  # (1, nbins, N_valid_ca, N_valid_ca)

    cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="none")(
        logit_pde_masked, true_pde_label[None]
    )  # assumes B=1
    if frame_atom_mask_2d is None:
        return torch.mean(cross_entropy_loss)
    else:
        # NOTE: This is probably not correct. What should happen is that you pass in "same_chain"
        # into this function, and then mask the PDE error with same chain. Instead,
        # I'm basically making the assumption that a (any frame -> CA) mask is
        # fairly close to a CA x CA mask. It's a subset of the true mask, so it
        # will only reduce what loss is computed over.
        frame_atom_ca_only_mask = frame_atom_mask_2d[:, atom_mask_ca, :, :, :]
        frame_atom_ca_only_mask = frame_atom_ca_only_mask[:, :, :, atom_mask_ca, :]
        frame_atom_ca_only_mask = frame_atom_ca_only_mask[:, :, :, :, 1].any(dim=2)
        frame_atom_ca_only_mask = frame_atom_ca_only_mask.unsqueeze(1)
        return torch.mean(frame_atom_ca_only_mask * cross_entropy_loss)


def mask_unresolved_frames(frames, frame_mask, atom_mask):
    """
    reindex frames tensor from relative indices to absolute indices and masks out frames with atoms that are unresolved
    in the structure
    Input:
        - frames: relative indices for frames (B, L, nframes, 3)
        - frame_mask: mask for which frames are valid to compute FAPE/losses (B, L, nframes)
        - atom_mask: mask for seen coordinates (B, L, natoms)
    Output:
        - frames_reindex: absolute indices for frames
        - frame_mask_update: updated frame mask with frames with unresolved atoms removed
    """
    B, L, natoms = atom_mask.shape

    # reindex frames for flat X
    frames_reindex = (
        torch.arange(L, device=frames.device)[None, :, None, None] + frames[..., 0]
    ) * natoms + frames[..., 1]

    masked_atom_frames = torch.any(
        frames_reindex > L * natoms, dim=-1
    )  # find frames with atoms that aren't resolved
    masked_atom_frames *= torch.any(frames_reindex < 0, dim=-1)
    # There are currently indices for frames that aren't in the coordinates bc they arent resolved, reset these indices to 0 to avoid
    # indexing errors
    frames_reindex[masked_atom_frames, :] = 0

    frame_mask_update = frame_mask.clone()
    frame_mask_update *= ~masked_atom_frames
    frame_mask_update *= torch.all(
        torch.gather(
            atom_mask.reshape(1, L * natoms),
            1,
            frames_reindex.reshape(1, L * ChemData().NFRAMES * 3),
        ).reshape(1, L, -1, 3),
        axis=-1,
    )

    return frames_reindex, frame_mask_update


def mask_unresolved_frames_batched(frames, frame_mask, atom_mask):
    """
    reindex frames tensor from relative indices to absolute indices and masks out frames with atoms that are unresolved
    in the structure
    Input:
        - frames: relative indices for frames (B, L, nframes, 3)
        - frame_mask: mask for which frames are valid to compute FAPE/losses (B, L, nframes)
        - atom_mask: mask for seen coordinates (B, L, natoms)
    Output:
        - frames_reindex: absolute indices for frames
        - frame_mask_update: updated frame mask with frames with unresolved atoms removed
    """
    B, L, natoms = atom_mask.shape

    # reindex frames for flat X
    frames_reindex = (
        torch.arange(L, device=frames.device)[None, :, None, None] + frames[..., 0]
    ) * natoms + frames[..., 1]

    masked_atom_frames = torch.any(
        frames_reindex > L * natoms, dim=-1
    )  # find frames with atoms that aren't resolved
    masked_atom_frames *= torch.any(frames_reindex < 0, dim=-1)
    # There are currently indices for frames that aren't in the coordinates bc they arent resolved, reset these indices to 0 to avoid
    # indexing errors
    frames_reindex[masked_atom_frames, :] = 0

    frame_mask_update = frame_mask.clone()
    frame_mask_update *= ~masked_atom_frames
    frame_mask_update *= torch.all(
        torch.gather(
            atom_mask.reshape(B, L * natoms),
            1,
            frames_reindex.reshape(B, L * ChemData().NFRAMES * 3),
        ).reshape(B, L, -1, 3),
        axis=-1,
    )

    return frames_reindex, frame_mask_update


def calc_crd_rmsd(pred, true, atom_mask, rmsd_mask=None, alignment_radius=None):
    """
    Calculate coordinate RMSD
    Input:
        - pred: predicted coordinates (B, L, natoms, 3)
        - true: true coordinates (B, L, natoms, 3)
        - atom_mask: mask for coordinates used for alignment (B, L, natoms)
        - rmsd_mask: mask for coordinates used for rmsd calculation
        - alignment_radius: radius around the rmsd mask that will be used for alignment (float)
    Output: RMSD after superposition
    """

    def rmsd(V, W, eps=1e-4):
        L = V.shape[1]
        return torch.sqrt(torch.sum((V - W) * (V - W), dim=(1, 2)) / L + eps)

    def centroid(X):
        return X.mean(dim=-2, keepdim=True)

    if rmsd_mask is None:
        rmsd_mask = atom_mask.clone()
    if alignment_radius is None:
        alignment_radius = torch.inf
    dist = torch.cdist(true[atom_mask][None], true[rmsd_mask][None])
    in_radius = (dist < alignment_radius).any(dim=-1)[
        0
    ]  # shape: (num seen atoms in atom_mask)

    B, L, natoms = pred.shape[:3]

    # center to centroid
    pred_allatom = pred[atom_mask][in_radius][None]
    true_allatom = true[atom_mask][in_radius][None]

    pred_allatom_origin = pred_allatom - centroid(pred_allatom)
    true_allatom_origin = true_allatom - centroid(true_allatom)

    # reshape true crds to match the shape to pred crds
    # true = true.unsqueeze(0).expand(I,-1,-1,-1,-1)
    # pred = pred.view(B, L*natoms, 3)
    # true = true.view(I*B, L*natoms, 3)

    # Computation of the covariance matrix
    C = torch.matmul(pred_allatom_origin.permute(0, 2, 1), true_allatom_origin)

    # Compute optimal rotation matrix using SVD
    V, S, W = torch.svd(C)

    # get sign to ensure right-handedness
    d = torch.ones([B, 3, 3], device=pred.device)
    d[:, :, -1] = torch.sign(torch.det(V) * torch.det(W)).unsqueeze(1)

    # Rotation matrix U
    U = torch.matmul(d * V, W.permute(0, 2, 1))  # (IB, 3, 3)

    pred_rms = pred[rmsd_mask][None] - centroid(pred_allatom)
    true_rms = true[rmsd_mask][None] - centroid(true_allatom)
    # Rotate pred
    rP = torch.matmul(pred_rms, U)  # (IB, L*3, 3)

    # get RMS
    rms = rmsd(rP, true_rms).reshape(B)
    return rms


def angle(a, b, c, eps=1e-4):
    """
    Calculate cos/sin angle between ab and cb
    a,b,c have shape of (B, L, 3)
    """
    B, L = a.shape[:2]

    u1 = a - b
    u2 = c - b

    u1_norm = torch.norm(u1, dim=-1, keepdim=True) + eps
    u2_norm = torch.norm(u2, dim=-1, keepdim=True) + eps

    # normalize u1 & u2 --> make unit vector
    u1 = u1 / u1_norm
    u2 = u2 / u2_norm
    u1 = u1.reshape(B * L, 3)
    u2 = u2.reshape(B * L, 3)

    # sin_theta = norm(a cross b)/(norm(a)*norm(b))
    # cos_theta = norm(a dot b) / (norm(a)*norm(b))
    sin_theta = torch.norm(torch.cross(u1, u2, dim=1), dim=1, keepdim=True).reshape(
        B, L, 1
    )  # (B,L,1)
    cos_theta = torch.matmul(u1[:, None, :], u2[:, :, None]).reshape(B, L, 1)

    return torch.cat([cos_theta, sin_theta], axis=-1)  # (B, L, 2)


def length(a, b):
    return torch.norm(a - b, dim=-1)


def torsion(a, b, c, d, eps=1e-4):
    # A function that takes in 4 atom coordinates:
    # a - [B,L,3]
    # b - [B,L,3]
    # c - [B,L,3]
    # d - [B,L,3]
    # and returns cos and sin of the dihedral angle between those 4 points in order a, b, c, d
    # output - [B,L,2]
    u1 = b - a
    u1 = u1 / (torch.norm(u1, dim=-1, keepdim=True) + eps)
    u2 = c - b
    u2 = u2 / (torch.norm(u2, dim=-1, keepdim=True) + eps)
    u3 = d - c
    u3 = u3 / (torch.norm(u3, dim=-1, keepdim=True) + eps)
    #
    t1 = torch.cross(u1, u2, dim=-1)  # [B, L, 3]
    t2 = torch.cross(u2, u3, dim=-1)
    t1_norm = torch.norm(t1, dim=-1, keepdim=True)
    t2_norm = torch.norm(t2, dim=-1, keepdim=True)

    cos_angle = torch.matmul(t1[:, :, None, :], t2[:, :, :, None])[:, :, 0]
    sin_angle = (
        torch.norm(u2, dim=-1, keepdim=True)
        * (torch.matmul(u1[:, :, None, :], t2[:, :, :, None])[:, :, 0])
    )

    cos_sin = torch.cat([cos_angle, sin_angle], axis=-1) / (
        t1_norm * t2_norm + eps
    )  # [B,L,2]
    return cos_sin


def cosangle(A, B, C, eps=1e-4):
    AB = A - B
    BC = C - B
    ABn = torch.sqrt(torch.sum(torch.square(AB), dim=-1) + eps)
    BCn = torch.sqrt(torch.sum(torch.square(BC), dim=-1) + eps)
    return torch.clamp(torch.sum(AB * BC, dim=-1) / (ABn * BCn), -0.999, 0.999)


def calc_cart_bonded(seq, pred, idx, len_param, ang_param, tor_param, eps=1e-4):
    # pred: N x L x 27 x 3
    # idx: 1 x L
    # seq: 1 x L
    def gen_ang(A, B, C):
        AB = A - B
        BC = C - B
        ABn = torch.sqrt(torch.sum(torch.square(AB), dim=-1) + eps)
        BCn = torch.sqrt(torch.sum(torch.square(BC), dim=-1) + eps)
        return torch.acos(
            torch.clamp(torch.sum(AB * BC, dim=-1) / (ABn * BCn), -0.999, 0.999)
        )

    # quadratic from [-1,1], linear elsewhere
    def boundfunc(X):
        Y = torch.abs(X)
        Y[Y < 1.0] = torch.square(Y[Y < 1.0])
        # Y = torch.square(X)
        return Y

    N, L = pred.shape[:2]
    cb_loss = torch.zeros(N, device=pred.device)

    ## intra-res
    cblens = len_param[seq]
    len_idx = cblens[..., :2].to(torch.long).reshape(1, L, -1, 1).repeat(N, 1, 1, 3)
    len_all = torch.gather(pred, 2, len_idx).reshape(N, L, -1, 2, 3)
    len_mask = cblens[..., 0] != cblens[..., 1]
    E_cb_len = (
        len_mask[None, ...]
        * cblens[None, ..., 3]
        * boundfunc(length(len_all[..., 0, :], len_all[..., 1, :]) - cblens[..., 2])
    ).sum(dim=(0, 3)) / len_mask.sum()

    # figure out which his are his_d
    cblens[seq == 8] = len_param[-1]
    len_idx = cblens[..., :2].to(torch.long).reshape(1, L, -1, 1).repeat(N, 1, 1, 3)
    len_all_a = torch.gather(pred, 2, len_idx).reshape(N, L, -1, 2, 3)
    len_mask_a = cblens[..., 0] != cblens[..., 1]
    E_cb_len_a = (
        len_mask_a[None, ...]
        * cblens[None, ..., 3]
        * boundfunc(length(len_all_a[..., 0, :], len_all_a[..., 1, :]) - cblens[..., 2])
    ).sum(dim=(0, 3)) / len_mask.sum()  # N,L
    is_his_d = (seq == 8) * (E_cb_len_a < E_cb_len)

    cb_loss += torch.min(E_cb_len_a, E_cb_len).sum(dim=1)

    cbangs = ang_param[seq].repeat(N, 1, 1, 1)
    cbangs[is_his_d] = ang_param[-1]
    ang_idx = cbangs[..., :3].to(torch.long).reshape(N, L, -1, 1).repeat(1, 1, 1, 3)
    ang_all = torch.gather(pred, 2, ang_idx).reshape(N, L, -1, 3, 3)
    ang_mask = cbangs[..., 0] != cbangs[..., 1]
    E_cb_ang = (
        ang_mask[None, ...]
        * cbangs[None, ..., 4]
        * boundfunc(
            get_ang(ang_all[..., 0, :], ang_all[..., 1, :], ang_all[..., 2, :])
            - cbangs[None, ..., 3]
        )
    ).sum(dim=(0, 2, 3)) / ang_mask.sum()
    cb_loss += E_cb_ang

    cbtors = tor_param[seq].repeat(N, 1, 1, 1)
    cbtors[is_his_d] = tor_param[-1]
    tor_idx = cbtors[..., :4].to(torch.long).reshape(N, L, -1, 1).repeat(1, 1, 1, 3)
    tor_all = torch.gather(pred, 2, tor_idx).reshape(N, L, -1, 4, 3)
    tor_mask = cbtors[..., 0] != cbtors[..., 1]
    offset = 2 * np.pi / cbtors[None, ..., 6]
    tor_deltas = (
        get_dih(
            tor_all[..., 0, :],
            tor_all[..., 1, :],
            tor_all[..., 2, :],
            tor_all[..., 3, :],
        )
        - cbtors[None, ..., 4]
        + 0.5 * offset
    ) % offset - 0.5 * offset

    # dihs = get_dih(
    #     tor_all[..., 0, :], tor_all[..., 1, :], tor_all[..., 2, :], tor_all[..., 3, :]
    # )

    E_cb_tor = (tor_mask[None, ...] * cbtors[None, ..., 5] * boundfunc(tor_deltas)).sum(
        dim=(0, 2, 3)
    ) / tor_mask.sum()
    cb_loss += E_cb_tor

    # inter-res
    # bond length: C-N
    bonded = (idx[:, 1:] - idx[:, :-1]) == 1
    blen_CN_pred = length(pred[:, :-1, 2], pred[:, 1:, 0]).reshape(N, L - 1)  # (B, L-1)
    CN_loss = ChemData().cb_lengths_CN[1] * boundfunc(
        blen_CN_pred - ChemData().cb_lengths_CN[0]
    )
    cb_loss += (bonded * CN_loss).sum(dim=1) / (bonded.sum())

    # bond angle: CA-C-N, C-N-CA
    bang_CACN_pred = get_ang(pred[:, :-1, 2], pred[:, 1:, 0], pred[:, 1:, 1]).reshape(
        N, L - 1
    )
    CACN_loss = ChemData().cb_angles_CACN[1] * boundfunc(
        bang_CACN_pred - ChemData().cb_angles_CACN[0]
    )
    cb_loss += (bonded * CACN_loss).sum(dim=1) / (bonded.sum())

    bang_CNCA_pred = get_ang(pred[:, :-1, 2], pred[:, 1:, 0], pred[:, 1:, 1]).reshape(
        N, L - 1
    )
    CNCA_loss = ChemData().cb_angles_CNCA[1] * boundfunc(
        bang_CNCA_pred - ChemData().cb_angles_CNCA[0]
    )
    cb_loss += (bonded * CNCA_loss).sum(dim=1) / (bonded.sum())

    # improper torsions CA-C-N-H (CD-C-N-CA), CA-N-C-O
    # planarity around N (H for non-pro, CD for pro)
    atom4idx = torch.full_like(seq, 14)
    atom4idx[seq == 14] = 6  # set to CD for proline
    atom4 = torch.gather(pred, 2, atom4idx[:, :, None, None].repeat(1, 1, 1, 3))
    btor_CACNH_delta = (
        get_dih(pred[:, :-1, 1], pred[:, :-1, 2], pred[:, 1:, 0], atom4[:, 1:, 0])
        - ChemData().cb_torsions_CACNH[0]
        + np.pi / 2
    ) % np.pi - np.pi / 2
    CACNH_loss = ChemData().cb_torsions_CACNH[1] * boundfunc(btor_CACNH_delta)
    cb_loss += (bonded * CACNH_loss).sum(dim=1) / (bonded.sum())

    # planarity around C
    btor_CANCO_delta = (
        get_dih(pred[:, :-1, 1], pred[:, 1:, 0], pred[:, :-1, 2], pred[:, :-1, 3])
        - ChemData().cb_torsions_CANCO[0]
        + np.pi / 2
    ) % np.pi - np.pi / 2
    CANCO_loss = ChemData().cb_torsions_CANCO[1] * boundfunc(btor_CANCO_delta)
    cb_loss += (bonded * CANCO_loss).sum(dim=1) / (bonded.sum())

    return cb_loss


# AF2-like version of clash score
def calc_clash(xs, mask):
    DISTCUT = 2.0  # (d_lit - tau) from AF2 MS
    L = xs.shape[0]
    dij = torch.sqrt(
        torch.sum(
            torch.square(xs[:, :, None, None, :] - xs[None, None, :, :, :]), dim=-1
        )
        + 1e-4
    )

    allmask = mask[:, :, None, None] * mask[None, None, :, :]
    allmask[torch.arange(L), :, torch.arange(L), :] = False  # ignore res-self
    allmask[torch.arange(1, L), 0, torch.arange(L - 1), 2] = False  # ignore N->C
    allmask[torch.arange(L - 1), 2, torch.arange(1, L), 0] = False  # ignore N->C

    clash = torch.sum(torch.clamp(DISTCUT - dij[allmask], 0.0)) / torch.sum(mask)
    return clash


# fd more efficient LJ loss
class LJLoss(torch.autograd.Function):
    @staticmethod
    def ljVdV(deltas, sigma, epsilon, lj_lin, eps):
        # deltas - (N,natompair,3)
        N = deltas.shape[0]

        dist = torch.sqrt(torch.sum(torch.square(deltas), dim=-1) + eps)
        linpart = dist < lj_lin * sigma[None]
        deff = dist.clone()
        deff[linpart] = lj_lin * sigma.repeat(N, 1)[linpart]
        sd = sigma / deff
        sd2 = sd * sd
        sd6 = sd2 * sd2 * sd2
        sd12 = sd6 * sd6
        ljE = epsilon * (sd12 - 2 * sd6)

        ljE[linpart] += (
            epsilon.repeat(N, 1)[linpart]
            * (-12 * sd12[linpart] / deff[linpart] + 12 * sd6[linpart] / deff[linpart])
            * (dist[linpart] - deff[linpart])
        )

        # works for linpart too
        dljEdd_over_r = epsilon * (-12 * sd12 / deff + 12 * sd6 / deff) / (dist)

        return ljE.sum(dim=-1), dljEdd_over_r

    @staticmethod
    def forward(
        ctx,
        xs,
        seq,
        aamask,
        bond_feats,
        dist_matrix,
        ljparams,
        ljcorr,
        num_bonds,
        lj_lin=0.75,
        lj_hb_dis=3.0,
        lj_OHdon_dis=2.6,
        lj_hbond_hdis=1.75,
        eps=1e-4,
        normNviolations=True,
        useH=False,
        norm_by_atoms_twice=False,  # this exists purely for backwards compatibility
        training=True,
    ):
        N, L, A = xs.shape[:3]
        assert N == 1  # see comment below

        # ds_res = torch.sqrt(
        #     torch.sum(
        #         torch.square(
        #             xs.detach()[:, :, None, 1, :] - xs.detach()[:, None, :, 1, :]
        #         ),
        #         dim=-1,
        #     )
        # )
        rs = torch.triu_indices(L, L, 0, device=xs.device)
        ri, rj = rs[0], rs[1]

        # batch during inference for huge systems
        BATCHSIZE = 65536 // N

        ljval = 0
        dljEdx = torch.zeros_like(xs, dtype=torch.float)

        for i_batch in range((len(ri) - 1) // BATCHSIZE + 1):
            idx = torch.arange(
                i_batch * BATCHSIZE,
                min((i_batch + 1) * BATCHSIZE, len(ri)),
                device=xs.device,
            )
            rii, rjj = ri[idx], rj[idx]  # residue pairs we consider

            if not useH:
                ridx, ai, aj = (
                    aamask[seq[rii], :14][:, :, None]
                    * aamask[seq[rjj], :14][:, None, :]
                ).nonzero(as_tuple=True)
            else:
                ridx, ai, aj = (
                    aamask[seq[rii]][:, :, None] * aamask[seq[rjj]][:, None, :]
                ).nonzero(as_tuple=True)

            deltas = (
                xs[:, rii, :, None, :] - xs[:, rjj, None, :, :]
            )  # N,BATCHSIZE,Natm,Natm,3
            seqi, seqj = seq[rii[ridx]], seq[rjj[ridx]]

            mask = torch.ones_like(ridx, dtype=torch.bool)  # are atoms defined?

            # mask out atom pairs from too-distant residues (C-alpha dist > 24A)
            ca_dist = torch.linalg.norm(deltas[:, :, 1, 1], dim=-1)
            mask *= (ca_dist[:, ridx] < 24).any(
                dim=0
            )  # will work for batch>1 but very inefficient

            intrares = rii[ridx] == rjj[ridx]
            mask[intrares * (ai < aj)] = False  # upper tri (atoms)

            ## count-pair
            # a) intra-protein
            mask[intrares] *= num_bonds[seqi[intrares], ai[intrares], aj[intrares]] >= 4
            pepbondres = ri[ridx] + 1 == rj[ridx]
            mask[pepbondres] *= (
                num_bonds[seqi[pepbondres], ai[pepbondres], 2]
                + num_bonds[seqj[pepbondres], 0, aj[pepbondres]]
                + 1
            ) >= 4

            # b) intra-ligand
            atommask = (ai == 1) * (aj == 1)
            dist_matrix = torch.nan_to_num(
                dist_matrix, posinf=4.0
            )  # NOTE: need to run nan_to_num to remove infinities
            resmask = dist_matrix[0, rii, rjj] >= 4  # * will only work for batch=1
            mask[atommask] *= resmask[ridx[atommask]]

            # c) protein/ligand
            ##fd NOTE1: changed 6->5 in masking (atom 5 is CG which should always be 4+ bonds away from connected atom)
            ##fd NOTE2: this does NOT work correctly for nucleic acids
            ##fd     for NAs atoms 0-4 are masked, but also 5,7,8 and 9 should be masked!
            bbatommask = (ai < 5) * (aj < 5)
            resmask = bond_feats[0, rii, rjj] != 6  # * will only work for batch=1
            mask[bbatommask] *= resmask[ridx[bbatommask]]

            # apply mask.  only interactions to be scored remain
            ai, aj, seqi, seqj, ridx = (
                ai[mask],
                aj[mask],
                seqi[mask],
                seqj[mask],
                ridx[mask],
            )
            deltas = deltas[:, ridx, ai, aj]

            # hbond correction
            use_hb_dis = (
                ljcorr[seqi, ai, 0] * ljcorr[seqj, aj, 1]
                + ljcorr[seqi, ai, 1] * ljcorr[seqj, aj, 0]
            ).nonzero()
            use_ohdon_dis = (  # OH are both donors & acceptors
                ljcorr[seqi, ai, 0] * ljcorr[seqi, ai, 1] * ljcorr[seqj, aj, 0]
                + ljcorr[seqi, ai, 0] * ljcorr[seqj, aj, 0] * ljcorr[seqj, aj, 1]
            ).nonzero()
            use_hb_hdis = (
                ljcorr[seqi, ai, 2] * ljcorr[seqj, aj, 1]
                + ljcorr[seqi, ai, 1] * ljcorr[seqj, aj, 2]
            ).nonzero()

            # disulfide correction
            potential_disulf = (ljcorr[seqi, ai, 3] * ljcorr[seqj, aj, 3]).nonzero()

            ljrs = ljparams[seqi, ai, 0] + ljparams[seqj, aj, 0]
            ljrs[use_hb_dis] = lj_hb_dis
            ljrs[use_ohdon_dis] = lj_OHdon_dis
            ljrs[use_hb_hdis] = lj_hbond_hdis

            ljss = torch.sqrt(ljparams[seqi, ai, 1] * ljparams[seqj, aj, 1] + eps)
            ljss[potential_disulf] = 0.0

            ljval_batch, dljEdd_i = LJLoss.ljVdV(deltas, ljrs, ljss, lj_lin, eps)
            natoms = 1.0  # LT resolve test_benchmark referenced before assignment - intialize natoms before alpha = 1.0/natoms
            if not normNviolations:
                if not useH:
                    natoms = torch.sum(aamask[seq, :14])
                else:
                    natoms = torch.sum(aamask[seq])
                ljval_batch = ljval_batch / natoms
                dljEdd_i = dljEdd_i / natoms
            else:  # mean over "clashing" atoms
                dist = torch.sqrt(torch.sum(torch.square(deltas), dim=-1) + eps)

                # count clashes (2 atoms closer than 90% of vdw radii)
                # normalize by (#clashes)+1
                # +1 ensures LJ is never "over-weighted"
                nclash = torch.sum(dist < 0.9 * ljrs) + 1.0

                ljval_batch = ljval_batch / nclash
                dljEdd_i = dljEdd_i / nclash

            ljval += ljval_batch

            # sum per-atom-pair grads into per-atom grads
            # note this is stochastic op on GPU
            idxI, idxJ = rii[ridx] * A + ai, rjj[ridx] * A + aj

            if norm_by_atoms_twice:
                alpha = 1.0 / natoms
            else:
                alpha = 1.0
            dljEdx.view(N, -1, 3).index_add_(
                1, idxI, dljEdd_i[..., None] * deltas, alpha=alpha
            )
            dljEdx.view(N, -1, 3).index_add_(
                1, idxJ, dljEdd_i[..., None] * deltas, alpha=-alpha
            )

        ctx.save_for_backward(dljEdx)

        return ljval

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        (dljEdx,) = ctx.saved_tensors
        return (
            grad_output * dljEdx,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


# Rosetta-like version of LJ (fa_atr+fa_rep)
#   lj_lin is switch from linear to 12-6.  Smaller values more sharply penalize clashes
def calc_lj(
    seq,
    xs,
    aamask,
    bond_feats,
    dist_matrix,
    ljparams,
    ljcorr,
    num_bonds,
    lj_lin=0.75,
    lj_hb_dis=3.0,
    lj_OHdon_dis=2.6,
    lj_hbond_hdis=1.75,
    lj_maxrad=-1.0,
    eps=1e-4,
    normNviolations=True,
    useH=False,
    training=True,
):
    lj = LJLoss.apply

    ljval = lj(
        xs,
        seq,
        aamask,
        bond_feats,
        dist_matrix,
        ljparams,
        ljcorr,
        num_bonds,
        lj_lin,
        lj_hb_dis,
        lj_OHdon_dis,
        lj_hbond_hdis,
        eps,
        normNviolations,
        useH,
        training,
    )

    return ljval


def calc_hb(
    seq,
    xs,
    aamask,
    hbtypes,
    hbbaseatoms,
    hbpolys,
    hb_sp2_range_span=1.6,
    hb_sp2_BAH180_rise=0.75,
    hb_sp2_outer_width=0.357,
    hb_sp3_softmax_fade=2.5,
    threshold_distance=6.0,
    eps=1e-4,
    normalize=True,
):
    def evalpoly(ds, xrange, yrange, coeffs):
        v = coeffs[..., 0]
        for i in range(1, 10):
            v = v * ds + coeffs[..., i]
        minmask = ds < xrange[..., 0]
        v[minmask] = yrange[minmask][..., 0]
        maxmask = ds > xrange[..., 1]
        v[maxmask] = yrange[maxmask][..., 1]
        return v

    def cosangle(A, B, C):
        AB = A - B
        BC = C - B
        ABn = torch.sqrt(torch.sum(torch.square(AB), dim=-1) + eps)
        BCn = torch.sqrt(torch.sum(torch.square(BC), dim=-1) + eps)
        return torch.clamp(torch.sum(AB * BC, dim=-1) / (ABn * BCn), -0.999, 0.999)

    hbts = hbtypes[seq]
    hbba = hbbaseatoms[seq]

    rh, ah = (hbts[..., 0] >= 0).nonzero(as_tuple=True)
    ra, aa = (hbts[..., 1] >= 0).nonzero(as_tuple=True)
    # D_xs = xs[rh, hbba[rh, ah, 0]][:, None, :]
    H_xs = xs[rh, ah][:, None, :]
    A_xs = xs[ra, aa][None, :, :]
    B_xs = xs[ra, hbba[ra, aa, 0]][None, :, :]
    B0_xs = xs[ra, hbba[ra, aa, 1]][None, :, :]
    hyb = hbts[ra, aa, 2]
    polys = hbpolys[hbts[rh, ah, 0][:, None], hbts[ra, aa, 1][None, :]]

    AH = torch.sqrt(torch.sum(torch.square(H_xs - A_xs), axis=-1) + eps)
    AHD = torch.acos(cosangle(B_xs, A_xs, H_xs))

    Es = polys[..., 0, 0] * evalpoly(
        AH, polys[..., 0, 1:3], polys[..., 0, 3:5], polys[..., 0, 5:]
    )
    Es += polys[..., 1, 0] * evalpoly(
        AHD, polys[..., 1, 1:3], polys[..., 1, 3:5], polys[..., 1, 5:]
    )

    Bm = 0.5 * (B0_xs[:, hyb == HbHybType.RING] + B_xs[:, hyb == HbHybType.RING])
    cosBAH = cosangle(Bm, A_xs[:, hyb == HbHybType.RING], H_xs)
    Es[:, hyb == HbHybType.RING] += polys[:, hyb == HbHybType.RING, 2, 0] * evalpoly(
        cosBAH,
        polys[:, hyb == HbHybType.RING, 2, 1:3],
        polys[:, hyb == HbHybType.RING, 2, 3:5],
        polys[:, hyb == HbHybType.RING, 2, 5:],
    )

    cosBAH1 = cosangle(
        B_xs[:, hyb == HbHybType.SP3], A_xs[:, hyb == HbHybType.SP3], H_xs
    )
    cosBAH2 = cosangle(
        B0_xs[:, hyb == HbHybType.SP3], A_xs[:, hyb == HbHybType.SP3], H_xs
    )
    Esp3_1 = polys[:, hyb == HbHybType.SP3, 2, 0] * evalpoly(
        cosBAH1,
        polys[:, hyb == HbHybType.SP3, 2, 1:3],
        polys[:, hyb == HbHybType.SP3, 2, 3:5],
        polys[:, hyb == HbHybType.SP3, 2, 5:],
    )
    Esp3_2 = polys[:, hyb == HbHybType.SP3, 2, 0] * evalpoly(
        cosBAH2,
        polys[:, hyb == HbHybType.SP3, 2, 1:3],
        polys[:, hyb == HbHybType.SP3, 2, 3:5],
        polys[:, hyb == HbHybType.SP3, 2, 5:],
    )
    Es[:, hyb == HbHybType.SP3] += (
        torch.log(
            torch.exp(Esp3_1 * hb_sp3_softmax_fade)
            + torch.exp(Esp3_2 * hb_sp3_softmax_fade)
        )
        / hb_sp3_softmax_fade
    )

    cosBAH = cosangle(
        B_xs[:, hyb == HbHybType.SP2], A_xs[:, hyb == HbHybType.SP2], H_xs
    )
    Es[:, hyb == HbHybType.SP2] += polys[:, hyb == HbHybType.SP2, 2, 0] * evalpoly(
        cosBAH,
        polys[:, hyb == HbHybType.SP2, 2, 1:3],
        polys[:, hyb == HbHybType.SP2, 2, 3:5],
        polys[:, hyb == HbHybType.SP2, 2, 5:],
    )

    BAH = torch.acos(cosBAH)
    B0BAH = get_dih(
        B0_xs[:, hyb == HbHybType.SP2],
        B_xs[:, hyb == HbHybType.SP2],
        A_xs[:, hyb == HbHybType.SP2],
        H_xs,
    )

    d, m, l = hb_sp2_BAH180_rise, hb_sp2_range_span, hb_sp2_outer_width
    Echi = torch.full_like(B0BAH, m - 0.5)

    mask1 = BAH > np.pi * 2.0 / 3.0
    H = 0.5 * (torch.cos(2 * B0BAH) + 1)
    F = d / 2 * torch.cos(3 * (np.pi - BAH[mask1])) + d / 2 - 0.5
    Echi[mask1] = H[mask1] * F + (1 - H[mask1]) * d - 0.5

    mask2 = BAH > np.pi * (2.0 / 3.0 - l)
    mask2 *= ~mask1
    outer_rise = torch.cos(np.pi - (np.pi * 2 / 3 - BAH[mask2]) / l)
    F = m / 2 * outer_rise + m / 2 - 0.5
    # G = (m - d) / 2 * outer_rise + (m - d) / 2 + d - 0.5
    Echi[mask2] = H[mask2] * F + (1 - H[mask2]) * d - 0.5

    Es[:, hyb == HbHybType.SP2] += polys[:, hyb == HbHybType.SP2, 2, 0] * Echi

    tosquish = torch.logical_and(Es > -0.1, Es < 0.1)
    Es[tosquish] = -0.025 + 0.5 * Es[tosquish] - 2.5 * torch.square(Es[tosquish])
    Es[Es > 0.1] = 0.0
    if normalize:
        return torch.sum(Es) / torch.sum(aamask[seq])
    else:
        return torch.sum(Es)


def calc_chiral_loss(pred, chirals):
    """
    calculate error in dihedral angles for chiral atoms
    Input:
     - pred: predicted coords (B, L, :, 3)
     - chirals: True coords (B, nchiral, 5), skip if 0 chiral sites, 5 dimension are indices for 4 atoms that make dihedral and the ideal angle they should form
    Output:
     - mean squared error of chiral angles
    """
    if chirals.shape[1] == 0:
        return torch.tensor(0.0, device=pred.device)
    chiral_dih = pred[:, chirals[..., :-1].long(), 1]
    pred_dih = get_dih(
        chiral_dih[..., 0, :],
        chiral_dih[..., 1, :],
        chiral_dih[..., 2, :],
        chiral_dih[..., 3, :],
    )  # n_symm, b, n, 36, 3
    l = torch.square(pred_dih - chirals[..., -1]).mean()
    return l


def calc_ddihedralmse_dxyz(a, b, c, d, true_dih, eps=1e-6):
    """
    Calculates the gradient of the dihedral angle with respect to the xyz coordinates using the closed form derivative.
    a, b, c, and d are atoms participating in the chiral center. true_dih is the true dihedral angle.

    Unlike the original implementation, this does NOT use autograd.
    """
    # I need to reshape this from n_symm, batch, n, 3 to n_symm * batch * n, 3)
    og_shape = a.shape
    # Expand the dihedral by the batch dimension to match n_atoms*batchs
    true_dih = true_dih.unsqueeze(0).repeat(a.shape[0], 1)
    a = a.view(-1, 3)
    b = b.view(-1, 3)
    c = c.view(-1, 3)
    d = d.view(-1, 3)
    true_dih = true_dih.view(-1)

    batch_size = a.shape[0]  # Support for batch size
    I = (
        torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(a.device)
    )  # Make batch-aware identity matrix

    # Compute b0, b1, b2
    b0 = a - b
    b1 = c - b
    b2 = d - c

    # Normalize b1
    b1_norm = torch.norm(b1, dim=-1, keepdim=True)
    b1n = b1 / (b1_norm + eps)

    # Compute orthogonal components v and w
    v = b0 - torch.sum(b0 * b1n, dim=-1, keepdim=True) * b1n
    w = b2 - torch.sum(b2 * b1n, dim=-1, keepdim=True) * b1n

    # Dihedral components x and y
    x = torch.sum(v * w, dim=-1)
    y = torch.sum(torch.cross(b1n, v, dim=-1) * w, dim=-1)

    # Dihedral angle
    dih = torch.atan2(y + eps, x + eps)

    # Compute MSE loss and manual gradients
    # mse_loss = torch.mean(torch.square(dih - true_dih))
    # mse_loss = torch.sum(torch.square(dih - true_dih))

    # Define matrices and gradients, adapted for batch
    db0_db = -I
    db1_db = -I
    db1_dc = I
    db2_dc = -I
    db0_da = I
    db2_dd = I
    # dmse_ddih = 2 * (dih - true_dih) / batch_size
    dmse_ddih = 2 * (dih - true_dih)
    ddih_dx = -y / (x**2 + y**2 + eps)
    ddih_dy = x / (x**2 + y**2 + eps)
    dy_dv = -torch.cross(b1n, w, dim=-1)
    dy_dw = torch.cross(b1n, v, dim=-1)
    dx_dv = w
    dx_dw = v

    dw_db1n = -torch.sum(b2 * b1n, dim=-1, keepdim=True).unsqueeze(-1) * I - torch.bmm(
        b2.unsqueeze(-1), b1n.unsqueeze(1)
    )

    db1n_db1 = (b1_norm + eps).unsqueeze(-1) * I / (b1_norm**2 + eps).unsqueeze(
        -1
    ) - torch.bmm(b1.unsqueeze(-1), b1.unsqueeze(1)) / (b1_norm**2 + eps).unsqueeze(-1)

    dv_db1n = -torch.sum(b0 * b1n, dim=-1, keepdim=True).unsqueeze(-1) * I - torch.bmm(
        b0.unsqueeze(-1), b1n.unsqueeze(1)
    )
    dv_db0 = I - torch.bmm(b1n.unsqueeze(-1), b1n.unsqueeze(1))
    dw_db2 = I - torch.bmm(b1n.unsqueeze(-1), b1n.unsqueeze(1))

    # Adjust sizes now for efficiency
    ddih_dx = ddih_dx.view(-1, 1, 1)
    ddih_dy = ddih_dy.view(-1, 1, 1)
    dmse_ddih = dmse_ddih.view(-1, 1, 1)
    dx_dv = dx_dv.unsqueeze(1)
    dx_dw = dx_dw.unsqueeze(1)
    dy_dv = dy_dv.unsqueeze(1)
    dy_dw = dy_dw.unsqueeze(1)

    # Gradient computations
    # wrt a
    dv_da = torch.matmul(dv_db0, db0_da)
    ddih_da = torch.bmm((ddih_dx * dx_dv), dv_da) + torch.bmm((ddih_dy * dy_dv), dv_da)
    dmse_da = torch.bmm(dmse_ddih, ddih_da)

    # wrt b
    db1n_db = torch.matmul(db1n_db1, db1_db)
    dv_db = torch.matmul(dv_db0, db0_db) + torch.matmul(
        dv_db1n.transpose(-1, -2), db1n_db
    )
    dw_db = torch.matmul(dw_db1n.transpose(-1, -2), db1n_db)
    dx_db = torch.bmm(dx_dv, dv_db) + torch.bmm(dx_dw, dw_db)
    dy_db = torch.bmm(dy_dv, dv_db) + torch.bmm(dy_dw, dw_db)
    ddih_db = torch.bmm(ddih_dx, dx_db) + torch.bmm(ddih_dy, dy_db)
    dmse_db = torch.bmm(dmse_ddih, ddih_db)

    # wrt c
    db1n_dc = torch.matmul(db1n_db1, db1_dc)
    dv_dc = torch.matmul(dv_db1n.transpose(-1, -2), db1n_dc)
    dw_dc = torch.matmul(dw_db2, db2_dc) + torch.matmul(
        dw_db1n.transpose(-1, -2), db1n_dc
    )
    dx_dc = torch.bmm(dx_dv, dv_dc) + torch.bmm(dx_dw, dw_dc)
    dy_dc = torch.bmm(dy_dv, dv_dc) + torch.bmm(dy_dw, dw_dc)
    ddih_dc = torch.bmm(ddih_dx, dx_dc) + torch.bmm(ddih_dy, dy_dc)
    dmse_dc = torch.bmm(dmse_ddih, ddih_dc)

    # wrt d
    dw_dd = torch.matmul(dw_db2, db2_dd)
    ddih_dd = torch.bmm((ddih_dx * dx_dw), dw_dd) + torch.bmm((ddih_dy * dy_dw), dw_dd)
    dmse_dd = torch.bmm(dmse_ddih, ddih_dd)

    # Reshape gradients back to original shape and prep for cat
    dmse_da = dmse_da.view(og_shape).unsqueeze(-2)
    dmse_db = dmse_db.view(og_shape).unsqueeze(-2)
    dmse_dc = dmse_dc.view(og_shape).unsqueeze(-2)
    dmse_dd = dmse_dd.view(og_shape).unsqueeze(-2)

    grads = torch.cat([dmse_da, dmse_db, dmse_dc, dmse_dd], dim=-2)
    return grads


def calc_chiral_grads_flat_impl(
    xyz, chiral_centers, chiral_center_dihedral_angles, no_grad_on_chiral_center
):
    """
    Calculates the gradient of the chiral centers with respect to the xyz coordinates using the closed form derivative.
    Args:
    xyz: torch.Tensor, shape (batch, n_atoms, 3)
    chiral_centers: torch.Tensor, shape (long) (n_centers, 4)
    chiral_center_dihedral_angles: torch.Tensor, shape (float) (n_centers, 1)

    Returns:
    grads: torch.Tensor, shape (batch, n_atoms, 3)
    """
    # (We want to track the gradient of the dihedral angle loss with respect to the xyz coordinates)
    xyz.requires_grad_(True)

    # Edge case: No chiral centers, return zero gradients
    if chiral_centers.shape[0] == 0:
        return torch.zeros(xyz.shape, device=xyz.device)

    # Get the coordinates of the four atoms that make up the chiral center
    chiral_dih = xyz[:, chiral_centers, :]

    # Calculate the gradient of the dihedral angle loss with respect to the xyz coordinates
    grads = torch.zeros_like(xyz).to(xyz.device)
    chiral_grads = calc_ddihedralmse_dxyz(
        chiral_dih[..., 0, :],
        chiral_dih[..., 1, :],
        chiral_dih[..., 2, :],
        chiral_dih[..., 3, :],
        chiral_center_dihedral_angles,
    )  # n_center, 4, 3

    if no_grad_on_chiral_center:
        chiral_grads[:, :, 0] = 0.0  # no gradient on chiral center

    # back to atom
    grads.index_add_(
        1,
        chiral_centers.flatten(),
        chiral_grads.flatten(start_dim=1, end_dim=2),
    )

    return grads


def calc_lddt(
    pred_ca,
    true_ca,
    mask_crds,
    mask_2d,
    same_chain,
    negative=False,
    interface=False,
    eps=1e-4,
):
    # Input
    # pred_ca: predicted CA coordinates (I, B, L, 3)
    # true_ca: true CA coordinates (B, L, 3)
    # pred_lddt: predicted lddt values (I-1, B, L)

    I, B, L = pred_ca.shape[:3]

    pred_dist = torch.cdist(pred_ca, pred_ca)  # (I, B, L, L)
    true_dist = torch.cdist(true_ca, true_ca).unsqueeze(0)  # (1, B, L, L)

    mask = torch.logical_and(true_dist > 0.0, true_dist < 15.0)  # (1, B, L, L)
    # update mask information
    mask *= mask_2d[None]
    if negative:
        mask *= same_chain.bool()[None]
    elif interface:
        # ignore atoms between the same chain
        mask *= ~same_chain.bool()[None]

    mask_crds = mask_crds * (mask[0].sum(dim=-1) != 0)

    delta = torch.abs(pred_dist - true_dist)  # (I, B, L, L)

    true_lddt = torch.zeros((I, B, L), device=pred_ca.device)
    for distbin in [0.5, 1.0, 2.0, 4.0]:
        true_lddt += (
            0.25
            * torch.sum((delta <= distbin) * mask, dim=-1)
            / (torch.sum(mask, dim=-1) + eps)
        )

    true_lddt = mask_crds * true_lddt
    true_lddt = true_lddt.sum(dim=(1, 2)) / (mask_crds.sum() + eps)
    return true_lddt


# fd allatom lddt
def calc_allatom_lddt(P, Q, idx, atm_mask, eps=1e-4):
    # P - N x L x 27 x 3
    # Q - L x 27 x 3
    N, L = P.shape[:2]

    # distance matrix
    Pij = torch.square(
        P[:, :, None, :, None, :] - P[:, None, :, None, :, :]
    )  # (N, L, L, 27, 27)
    Pij = torch.sqrt(Pij.sum(dim=-1) + eps)
    Qij = torch.square(
        Q[None, :, None, :, None, :] - Q[None, None, :, None, :, :]
    )  # (1, L, L, 27, 27)
    Qij = torch.sqrt(Qij.sum(dim=-1) + eps)

    # get valid pairs
    pair_mask = torch.logical_and(
        Qij > 0, Qij < 15
    ).float()  # only consider atom pairs within 15A
    # ignore missing atoms
    pair_mask *= (atm_mask[:, :, None, :, None] * atm_mask[:, None, :, None, :]).float()

    # ignore atoms within same residue
    pair_mask *= (
        idx[:, :, None, None, None] != idx[:, None, :, None, None]
    ).float()  # (1, L, L, 27, 27)

    delta_PQ = torch.abs(Pij - Qij + eps)  # (N, L, L, 14, 14)

    lddt = torch.zeros((N, L, 27), device=P.device)  # (N, L, 27)
    for distbin in (0.5, 1.0, 2.0, 4.0):
        lddt += (
            0.25
            * torch.sum((delta_PQ <= distbin) * pair_mask, dim=(2, 4))
            / (torch.sum(pair_mask, dim=(2, 4)) + 1e-4)
        )

    lddt = (lddt * atm_mask).sum(dim=(1, 2)) / (atm_mask.sum() + eps)
    return lddt


def calc_allatom_lddt_loss(
    P,
    Q,
    pred_lddt,
    idx,
    atm_mask,
    mask_2d,
    same_chain,
    negative=False,
    interface=False,
    bin_scaling=1,
    N_stripe=1,
    eps=1e-4,
):
    # P - N x L x natoms x 3
    # Q - L x natoms x 3
    # pred_lddt - 1 x nbucket x L
    # idx - 1 x L
    #
    N, L, Natm = P.shape[:3]

    # striped evaluation of L x L x N_atoms x N_atoms distances to save GPU mem
    L_stripe = int(np.ceil(L / N_stripe))  # how many residues in each stripe
    lddt_s = []
    pair_mask_accum = torch.zeros((N, L, Natm), device=P.device)
    for i1 in np.arange(0, L, L_stripe):
        i2 = min(i1 + L_stripe, L)

        # distance matrix
        Pij = torch.square(
            P[:, i1:i2, None, :, None, :] - P[:, None, :, None, :, :]
        )  # (N, L_stripe, L, 27, 27)
        Pij = torch.sqrt(Pij.sum(dim=-1) + eps)
        Qij = torch.square(
            Q[None, i1:i2, None, :, None, :] - Q[None, None, :, None, :, :]
        )  # (1, L_stripe, L, 27, 27)
        Qij = torch.sqrt(Qij.sum(dim=-1) + eps)

        # get valid pairs
        pair_mask = torch.logical_and(
            Qij > 0, Qij < 15
        ).float()  # only consider atom pairs within 15A
        # ignore missing atoms
        pair_mask *= (
            atm_mask[:, i1:i2, None, :, None] * atm_mask[:, None, :, None, :]
        ).float()

        # ignore atoms within same residue
        pair_mask *= (
            idx[:, i1:i2, None, None, None] != idx[:, None, :, None, None]
        ).float()  # (1, L_stripe, L, 27, 27)
        if negative:
            # ignore atoms between different chains
            pair_mask *= same_chain.bool()[:, i1:i2, :, None, None]
        elif interface:
            # ignore atoms between the same chain
            pair_mask *= ~same_chain.bool()[:, i1:i2, :, None, None]

        pair_mask *= mask_2d.bool()[:, i1:i2, :, None, None]

        delta_PQ = torch.abs(Pij - Qij + eps)  # (N, L_stripe, L, 14, 14)

        lddt_ = torch.zeros((N, i2 - i1, Natm), device=P.device)
        for distbin in (0.5, 1.0, 2.0, 4.0):
            lddt_ += (
                0.25
                * torch.sum((delta_PQ <= distbin * bin_scaling) * pair_mask, dim=(2, 4))
                / (torch.sum(pair_mask, dim=(2, 4)) + eps)
            )
        lddt_s.append(lddt_)
        pair_mask_accum += pair_mask.sum(dim=(1, 3))

    lddt = torch.cat(lddt_s, dim=1)  # (N, L, Natm)

    # per-res
    final_lddt_by_res = torch.clamp(
        (lddt[-1] * atm_mask[0]).sum(-1) / (atm_mask.sum(-1) + eps), min=0.0, max=1.0
    )

    # per-struct
    atm_mask = atm_mask * (pair_mask_accum != 0)
    lddt = (lddt * atm_mask).sum(dim=(1, 2)) / (atm_mask.sum() + eps)

    # calculate lddt prediction loss
    if pred_lddt is not None:
        nbin = pred_lddt.shape[1]
        bin_step = 1.0 / nbin
        lddt_bins = torch.linspace(
            bin_step, 1.0, nbin, dtype=pred_lddt.dtype, device=pred_lddt.device
        )
        true_lddt_label = torch.bucketize(
            final_lddt_by_res[None, ...], lddt_bins
        ).long()
        lddt_loss = torch.nn.CrossEntropyLoss(reduction="none")(
            pred_lddt, true_lddt_label[-1]
        )

        res_mask = atm_mask.any(dim=-1)
        lddt_loss = (lddt_loss * res_mask).sum() / (res_mask.sum() + eps)
    else:
        lddt_loss = None  # no pred lddt provided

    return lddt_loss, lddt


def rms_aln_tgt(predin, true, mask):
    def centroid(X):
        return X.mean(dim=-2, keepdim=True)

    pred = predin[mask]
    true = true[mask]
    pred = pred - centroid(pred)
    cT = centroid(true)
    true = true - cT
    C = torch.einsum("ji,jk->ik", pred, true)
    V, S, W = torch.svd(C)
    d = torch.ones([3, 3], device=pred.device)
    d[:, -1] = torch.sign(torch.det(V) * torch.det(W)).unsqueeze(0)
    U = torch.matmul(d * V, W.permute(1, 0))  # (3, 3)

    rpred = torch.matmul(predin, U) + cT
    return rpred


def translation_vector_field(pred_trans_1, noaln_gt_trans_1, mask, r3_t, params):
    gt_trans_1 = rms_aln_tgt(noaln_gt_trans_1, pred_trans_1.detach(), mask)

    return translation_vector_field_noaln(pred_trans_1, gt_trans_1, mask, r3_t, params)


def translation_vector_field_noaln(pred_trans_1, gt_trans_1, mask, r3_t, params):
    t_normalize_clip = params.t_normalize_clip
    trans_scale = params.trans_scale  # global scale

    t_dep_scale = 1 - torch.min(  # t-dependant scale
        r3_t[..., None], torch.tensor(t_normalize_clip)
    )  # (B, 1, 1)
    trans_error = trans_scale / t_dep_scale * (gt_trans_1 - pred_trans_1)

    loss_denom = 3 * torch.sum(mask)
    trans_loss = (
        torch.sum(trans_error * trans_error * mask[..., None], dim=(-1, -2))
        / loss_denom
    )

    return trans_loss
