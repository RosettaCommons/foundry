import torch
from icecream import ic
import numpy as np
import scipy
import pandas as pd
import networkx as nx

from rf2aa.util import (
    is_protein,
    rigid_from_3_points,
    cb_lengths_CN,
    cb_angles_CACN,
    cb_angles_CNCA,
    cb_torsions_CACNH,
    cb_torsions_CANCO,
    is_nucleic,
    find_all_paths_of_length_n,
    find_all_rigid_groups
)
from rf2aa.chemical import NFRAMES, NTOTAL

from rf2aa.kinematics import get_dih, get_ang
from rf2aa.scoring import HbHybType

from typing import List, Dict

# Loss functions for the training
# 1. BB rmsd loss
# 2. distance loss (or 6D loss?)
# 3. bond geometry loss
# 4. predicted lddt loss

#fd use improved coordinate frame generation
def get_t(N, Ca, C, eps=1e-5):
    I,B,L=N.shape[:3]
    Rs,Ts = rigid_from_3_points(N.view(I*B,L,3), Ca.view(I*B,L,3), C.view(I*B,L,3), eps=eps)
    Rs = Rs.view(I,B,L,3,3)
    Ts = Ts.view(I,B,L,3)
    t = Ts.unsqueeze(-2) - Ts.unsqueeze(-3)
    return torch.einsum('iblkj, iblmk -> iblmj', Rs, t) # (I,B,L,L,3) **fixed

def calc_str_loss(pred, true, mask_2d, same_chain, negative=False, d_clamp_intra=10.0, d_clamp_inter=30.0, A=10.0, gamma=0.99, eps=1e-6):
    '''
    Calculate Backbone FAPE loss
    Input:
        - pred: predicted coordinates (I, B, L, n_atom, 3)
        - true: true coordinates (B, L, n_atom, 3)
    Output: str loss
    '''
    I = pred.shape[0]
    true = true.unsqueeze(0)
    t_tilde_ij = get_t(true[:,:,:,0], true[:,:,:,1], true[:,:,:,2])
    t_ij = get_t(pred[:,:,:,0], pred[:,:,:,1], pred[:,:,:,2])

    difference = torch.sqrt(torch.square(t_tilde_ij-t_ij).sum(dim=-1) + eps)
    clamp = torch.zeros_like(difference)
    clamp[:,same_chain==1] = d_clamp_intra
    clamp[:,same_chain==0] = d_clamp_inter
    difference = torch.clamp(difference, max=clamp)
    loss = difference / A # (I, B, L, L)

    # Get a mask information (ignore missing residue + inter-chain residues)
    # for positive cases, mask = mask_2d
    # for negative cases (non-interacting pairs) mask = mask_2d*same_chain
    if negative:
        mask = mask_2d * same_chain
    else:
        mask = mask_2d
    # calculate masked loss (ignore missing regions when calculate loss)
    loss = (mask[None]*loss).sum(dim=(1,2,3)) / (mask.sum()+eps) # (I)

    # weighting loss
    w_loss = torch.pow(torch.full((I,), gamma, device=pred.device), torch.arange(I, device=pred.device))
    w_loss = torch.flip(w_loss, (0,))
    w_loss = w_loss / w_loss.sum()

    tot_loss = (w_loss * loss).sum()
    return tot_loss, loss.detach() 

#resolve rotationally equivalent sidechains
def resolve_symmetry(xs, Rsnat_all, xsnat, Rsnat_all_alt, xsnat_alt, atm_mask):
    dists = torch.linalg.norm( xs[:,:,None,:] - xs[atm_mask,:][None,None,:,:], dim=-1)
    dists_nat = torch.linalg.norm( xsnat[:,:,None,:] - xsnat[atm_mask,:][None,None,:,:], dim=-1)
    dists_natalt = torch.linalg.norm( xsnat_alt[:,:,None,:] - xsnat_alt[atm_mask,:][None,None,:,:], dim=-1)

    drms_nat = torch.sum(torch.abs(dists_nat-dists),dim=(-1,-2))
    drms_natalt = torch.sum(torch.abs(dists_nat-dists_natalt), dim=(-1,-2))

    Rsnat_symm = Rsnat_all
    xs_symm = xsnat

    toflip = drms_natalt<drms_nat

    Rsnat_symm[toflip,...] = Rsnat_all_alt[toflip,...]
    xs_symm[toflip,...] = xsnat_alt[toflip,...]

    return Rsnat_symm, xs_symm

# resolve "equivalent" natives
def resolve_equiv_natives(xs, natstack, maskstack):
    if (len(natstack.shape)==4):
        return natstack, maskstack
    if (natstack.shape[1]==1):
        return natstack[:,0,...], maskstack[:,0,...]
    dx = torch.norm( xs[:,None,:,None,1,:]-xs[:,None,None,:,1,:], dim=-1)
    dnat = torch.norm( natstack[:,:,:,None,1,:]-natstack[:,:,None,:,1,:], dim=-1)
    delta = torch.sum( torch.abs(dnat-dx), dim=(-2,-1))
    return natstack[:,torch.argmin(delta),...], maskstack[:,torch.argmin(delta),...]


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

    # NOTE: I am pretty sure these next two lines are a mistake. 
    # if xyz_true.shape[1] == 1:
    #     return xyz_true[:, 0, ...], mask[:, 0, ...]

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
    
    valid_protein_permutations = torch.any(torch.any(mask[:,:, :total_protein_length],dim=2), dim=2) # only take distances over valid protein permutations
    xyz_true_valid_prot = xyz_true[:, valid_protein_permutations[0]] ## rk assumes B==1
    mask_valid_prot = mask[:, valid_protein_permutations[0]] ## rk assumes B==1
    true_ca_ca_distances = torch.norm(
        xyz_true_valid_prot[:, :, :total_protein_length, None, 1, :]
        - xyz_true_valid_prot[:, :, None, :total_protein_length, 1, :],
        dim=-1,
    )
    pred_true_ca_ca_dist_diff = torch.sum(
        torch.abs(true_ca_ca_distances - pred_ca_ca_distances), dim=(-2, -1)
    )
    pred_true_ca_ca_dist_diff = torch.nan_to_num(
        pred_true_ca_ca_dist_diff, nan=torch.inf
    )
    
    best_protein_perm_index_per_batch = torch.argmin(pred_true_ca_ca_dist_diff, axis=-1) # indices over indices of xyz_true_valid_prot

    best_protein_coords = xyz_true_valid_prot[
        torch.arange(batch_size),
        best_protein_perm_index_per_batch,
        :total_protein_length,
    ]
    matching_protein_mask = mask_valid_prot[
        torch.arange(batch_size),
        best_protein_perm_index_per_batch,
        :total_protein_length,
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
    label_to_ligand_coordinates = {} # unrolled coordinates for each unique ligand label
    label_to_ligand_offsets = {} # indexing for to get coords for this chosen sm in length dimension
    label_to_symmetry_offsets = {} # indexing for to get coords for this chosen sm in symm dimension
    label_to_position_index = {} # holds information on alternative automorphs for same ligand

    running_offset = total_protein_length
    # This first for loop groups ligand coordinates by 
    # "chain" label, e.g. groups by identical ligands. It assumes
    # that identical ligands are of the same length.
    for position_index, ligand_length in enumerate(Ls_sm):
        # Note: this functionality assumes a batch size of 1.
        ligand_label = ch_label[0, running_offset].item()
        ligand_offsets = [running_offset, running_offset + ligand_length]
        ligand_coordinates = xyz_true[:, :, ligand_offsets[0] : ligand_offsets[1]]
        ligand_mask = mask[:, :, ligand_offsets[0] : ligand_offsets[1]]
        # any across coordinates (x, y, z) and length
        ligand_mask_any_atoms = ligand_mask[:, :, :, 1].any(dim=-1) # B, NSymm
        ligand_coordinates_valid = ligand_coordinates[ligand_mask_any_atoms][None] # assumes B=1
        
        #NOTE: assumes that the first N dimensions are real coordinates and the rest are buffer 
        num_valid_ligand_perms = torch.sum(ligand_mask_any_atoms) 
        symmetry_offsets = torch.arange(
            num_valid_ligand_perms, device=ligand_coordinates_valid.device
        ) 
        ligand_offsets_tensor = (
            torch.tensor(ligand_offsets, device=ligand_coordinates_valid.device).reshape(1, 2).repeat(num_valid_ligand_perms, 1)
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
        else:
            label_to_ligand_coordinates[ligand_label] = [ligand_coordinates_valid]
            label_to_ligand_offsets[ligand_label] = [ligand_offsets_tensor]
            label_to_symmetry_offsets[ligand_label] = [symmetry_offsets]
            label_to_position_index[ligand_label] = [position_index_tensor]

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

    # This final for loop does the bulk of the computation:
    # for each ligand, it computes the ca-prot-lig distance
    # error between that ligands predicted coordinates
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
        ) # (B, L_prot + L_query_lig, 3)
        pred_prot_lig_ca_distance = torch.cdist(
            xyz_pred_prot_and_select_lig_ca,
            xyz_pred_query_lig_ca,
            compute_mode="donot_use_mm_for_euclid_dist",
        ) # (B, L_prot+ L_query_lig, L_query_lig)

        xyz_true_lig_ca = label_to_ligand_coordinates[ligand_label][:, :, :, 1]
        xyz_true_prot_and_select_lig_ca = torch.cat(
            [
                xyz_true_prot_ca[None].repeat((1, xyz_true_lig_ca.shape[1], 1, 1)),
                xyz_true_lig_ca,
            ],
            dim=2,
        ) # (B, Nsymm*Nrepeats, L_prot+ L_query_lig, 3)
        all_true_prot_lig_ca_distance = torch.cdist(
            xyz_true_prot_and_select_lig_ca,
            xyz_true_lig_ca,
            compute_mode="donot_use_mm_for_euclid_dist",
        ) # (B, Nsymm*Nrepeats, L_prot+ L_query_lig, L_query_lig)

        pred_true_lig_prot_dist_diff = torch.sum(
            torch.abs(pred_prot_lig_ca_distance - all_true_prot_lig_ca_distance),
            dim=(-2, -1),
        )
        pred_true_lig_prot_dist_diff = torch.nan_to_num(
            pred_true_lig_prot_dist_diff, nan=torch.inf
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

        running_offset += ligand_length

    return xyz_out, mask_out


def calc_rmsd(pred, true, mask):
    # pred (N,B,Lasu,natom,3)
    # true (B,Lasu,natom,3)
    # mask (B,Lasu,natom)
    def rmsd(V, W, eps=1e-6):
        L = V.shape[1]
        return torch.sqrt(torch.sum((V-W)*(V-W), dim=(1,2)) / L + eps)
    def centroid(X):
        return X.mean(dim=-2, keepdim=True)

    N, B, L, Natm = pred.shape[:4]
    resmask = mask[0,:,1]
    pred = pred[:,:,resmask,1].squeeze(1)
    true = true[:,resmask,1]
    cP = centroid(pred)
    cT = centroid(true)
    pred = pred - cP
    true = true - cT
    C = torch.einsum('bji,njk->bik', pred, true)
    V, S, W = torch.svd(C)
    d = torch.ones([N,3,3], device=pred.device)
    d[:,:,-1] = torch.sign(torch.det(V)*torch.det(W)).unsqueeze(1)
    U = torch.matmul(d*V, W.permute(0,2,1)) # (IB, 3, 3)
    rpred = torch.matmul(pred, U) # (IB, L*3, 3)
    rms = rmsd(rpred, true).reshape(N)
    return rms, U, cP, cT


def resolve_symmetry_predictions(pred, true, mask, Lasu):
    # pred (N,B,Lpred,natom,3)
    # true (B,Ltrue,natom,3)
    # mask (B,Ltrue,natom)
    # symmR (S,3,3)
    
    N,B,Lpred = pred.shape[:3]
    Ltrue = true.shape[1]
    Opred = Lpred//Lasu
    Otrue = Ltrue//Lasu
    #if (Opred < Otrue):
    #    print (Opred,Otrue,Lpred,Ltrue,Lasu)

    # U[i] rotates layer i pred to native[0]
    r, U, cP, cT = calc_rmsd(pred[:,:,:Lasu].detach(), true[:,:Lasu], mask[:,:Lasu]) # no grads through RMSd

    # get com for each subunit in pred, align to true
    pcoms = (
        torch.sum(
            pred[:,:,:,1].view(N,Opred,Lasu,3).detach()
            *mask[0,:Lasu,1].view(1,1,Lasu,1)
            , dim=-2)
        / torch.sum(mask[0,:Lasu,1].view(1,1,Lasu,1), dim=-2)
    )
    pcoms = torch.einsum('lij,lsj->lsi',U,pcoms-cP)+cT

    # get com for each subunit in true
    ncoms = (
        torch.sum(
            true[0,:,1].view(-1,Lasu,3).detach()
            *mask[0,:,1,None].view(-1,Lasu,1)
            , dim=-2)
        / torch.sum(mask[0,:,1].view(-1,Lasu,1), dim=-2)
    )
    ds = torch.linalg.norm(pcoms[:,None] - ncoms[None,:,None], dim=-1) # (N,Otrue,Opred)

    # find correspondance P->T
    mapping_T_to_P = torch.full((N,Otrue),-1, device=ds.device)
    for i in range(Otrue):
        minj, dj = torch.min(ds, dim=2)
        di = torch.argmin(minj, dim=1)
        dj = dj.gather(1,di[:,None]).squeeze(1)
        mapping_T_to_P.scatter_(1,di[:,None],dj[:,None])
        ds.scatter_(1,di[:,None,None].repeat(1,1,Opred),torch.full((N,1,Opred),9999.0, device=ds.device))
        ds.scatter_(2,dj[:,None,None].repeat(1,Otrue,1),torch.full((N,Otrue,1),9999.0, device=ds.device))

    # convert subunit indices to residue indices
    mapping_T_to_P = torch.repeat_interleave(mapping_T_to_P,Lasu,dim=1)
    mapping_T_to_P = mapping_T_to_P*Lasu + torch.arange(Lasu, device=ds.device).repeat(1,Otrue)
    return mapping_T_to_P


#torsion angle predictor loss
def torsionAngleLoss( alpha, alphanat, alphanat_alt, tors_mask, tors_planar, eps=1e-8 ):
    I = alpha.shape[0]
    lnat = torch.sqrt( torch.sum( torch.square(alpha), dim=-1 ) + eps )
    anorm = alpha / (lnat[...,None])

    l_tors_ij = torch.min(
            torch.sum(torch.square( anorm - alphanat[None] ),dim=-1),
            torch.sum(torch.square( anorm - alphanat_alt[None] ),dim=-1)
        )
    l_tors = torch.sum( l_tors_ij*tors_mask[None] ) / (torch.sum( tors_mask )*I + eps)
    l_norm = torch.sum( torch.abs(lnat-1.0)*tors_mask[None] ) / (torch.sum( tors_mask )*I + eps)
    l_planar = torch.sum( torch.abs( alpha[...,0] )*tors_planar[None] ) / (torch.sum( tors_planar )*I + eps)

    return l_tors+0.02*l_norm+0.02*l_planar

def compute_FAPE(Rs, Ts, xs, Rsnat, Tsnat, xsnat, Z=10.0, dclamp=10.0, eps=1e-4):
    xij = torch.einsum('rji,rsj->rsi', Rs, xs[None,...] - Ts[:,None,...])
    xij_t = torch.einsum('rji,rsj->rsi', Rsnat, xsnat[None,...] - Tsnat[:,None,...])

    #torch.norm(xij-xij_t,dim=-1)
    diff = torch.sqrt( torch.sum( torch.square(xij-xij_t), dim=-1 ) + eps )

    loss = (1.0/Z) * (torch.clamp(diff, max=dclamp)).mean()

    return loss

def compute_pae_loss(X, X_y, uX, Y, Y_y, uY, logit_pae, frame_mask, atom_mask, pae_bin_step=0.5, eps=1e-4, frame_atom_mask_2d=None):
    """Predicted Aligned Error: C-alpha (or sm. mol atom) distances in backbone frames from final layer"""
    frame_mask_bb = frame_mask[0,:,0] # valid backbone frames (L,)
    atom_mask_ca = atom_mask[0,:,1] # valid CA atoms (L,)

    xij_ca = torch.einsum(
        'fji,faj->fai',
        uX[-1,frame_mask_bb,0],
        X[-1,None,atom_mask_ca,1] - X_y[-1,frame_mask_bb,None,0]
    ) # (N_valid_frames, N_valid_ca, 3)

    xij_ca_t = torch.einsum(
        'fji,faj->fai',
        uY[-1,frame_mask_bb,0],
        Y[-1,None,atom_mask_ca,1] - Y_y[-1,frame_mask_bb,None,0]
    ) # (N_valid_frames, N_valid_ca, 3)

    eij_label = torch.sqrt(torch.square(xij_ca - xij_ca_t).sum(dim=-1)+eps).clone().detach()

    nbin = logit_pae.shape[1]
    pae_bins = torch.linspace(pae_bin_step, pae_bin_step*(nbin-1), nbin-1, dtype=logit_pae.dtype, device=logit_pae.device)
    true_pae_label = torch.bucketize(eij_label, pae_bins, right=True).long()

    logit_pae_masked = logit_pae[:,:,frame_mask_bb][...,atom_mask_ca] # (1, nbins, N_valid_frames, N_valid_ca)

    cross_entropy_loss =  torch.nn.CrossEntropyLoss(reduction='none')(logit_pae_masked, true_pae_label[None]) # assumes B=1

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


def compute_pde_loss(X, Y, logit_pde, atom_mask, pde_bin_step=0.3, frame_atom_mask_2d=None):
    """Predicted Distance Error: C-alpha (or sm. mol atom) pairwise distances"""
    atom_mask_ca = atom_mask[0,:,1] # valid CA atoms (L,)

    dX = torch.cdist(X[-1,atom_mask_ca,1], X[-1,atom_mask_ca,1], compute_mode='donot_use_mm_for_euclid_dist')
    dY = torch.cdist(Y[0,atom_mask_ca,1], Y[0,atom_mask_ca,1], compute_mode='donot_use_mm_for_euclid_dist')
    dist_err = torch.abs(dX-dY).clone().detach()

    nbin = logit_pde.shape[1]
    pde_bins = torch.linspace(pde_bin_step, pde_bin_step*(nbin-1), nbin-1, dtype=logit_pde.dtype, device=logit_pde.device)
    true_pde_label = torch.bucketize(dist_err, pde_bins, right=True).long()
    logit_pde_masked = logit_pde[:,:,atom_mask_ca][...,atom_mask_ca] # (1, nbins, N_valid_ca, N_valid_ca)

    cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')(logit_pde_masked, true_pde_label[None]) # assumes B=1
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

# from Ivan: FAPE generalized over atom sets & frames
def compute_general_FAPE(X, Y, atom_mask, frames, frame_mask, frame_atom_mask=None, frame_atom_mask_2d=None, 
    logit_pae=None, logit_pde=None, Z=10.0, dclamp=10.0, dclamp_2d=None, gamma=0.99, eps=1e-4):

    # X (predicted) N x L x natoms x 3
    # Y (native)    1 x L x natoms x 3
    # atom_mask     1 x L x natoms masks the atoms over which fape is calculated
    # frames        1 x L x nframes x 3 x 2
    # frame_mask    1 x L x nframes 
    # frame_atom_mask     1 x L x natoms masks the frames over which fape is calculated
    # frame_atom_mask_2d 1 x L x nframes x L x natoms 2d mask, 2nd dimension frames, 3rd/4th dimension atoms so fape can be taken over some atoms for some frames (only works for BB fape)

    if frame_atom_mask is None:
        frame_atom_mask = atom_mask

    N, L, natoms, _ = X.shape

    # flatten middle dims so can gather across residues
    X_prime = X.reshape(N, L*natoms, -1, 3).repeat(1,1,NFRAMES,1)
    Y_prime = Y.reshape(1, L*natoms, -1, 3).repeat(1,1,NFRAMES,1)

    frames_reindex, frame_mask = mask_unresolved_frames(frames, frame_mask, frame_atom_mask)
    
    if torch.sum(frame_mask) == 0:
        return torch.tensor([0.0], device=X.device), torch.tensor([0.0], device=X.device), torch.tensor([0.0], device=X.device)
        
    X_x = torch.gather(X_prime, 1, frames_reindex[...,0:1].repeat(N,1,1,3))
    X_y = torch.gather(X_prime, 1, frames_reindex[...,1:2].repeat(N,1,1,3))
    X_z = torch.gather(X_prime, 1, frames_reindex[...,2:3].repeat(N,1,1,3))
    uX,tX = rigid_from_3_points(X_x, X_y, X_z)

    Y_x = torch.gather(Y_prime, 1, frames_reindex[...,0:1].repeat(1,1,1,3))
    Y_y = torch.gather(Y_prime, 1, frames_reindex[...,1:2].repeat(1,1,1,3))
    Y_z = torch.gather(Y_prime, 1, frames_reindex[...,2:3].repeat(1,1,1,3))
    uY,tY = rigid_from_3_points(Y_x, Y_y, Y_z)

    xij = torch.einsum(
        'brji,brsj->brsi',
        uX[:,frame_mask[0]], X[:,atom_mask[0]][:,None,...] - X_y[:,frame_mask[0]][:,:,None,...]
    )
    xij_t = torch.einsum('rji,rsj->rsi', uY[frame_mask], Y[atom_mask][None,...] - Y_y[frame_mask][:,None,...])
    diff = torch.sqrt( torch.sum( torch.square(xij-xij_t[None,...]), dim=-1 ) + eps )
    
    # multiply diff by frame_atom_mask_2d if frame_atom_mask_2d not None
    if frame_atom_mask_2d is not None:
        diff = diff*frame_atom_mask_2d[:,frame_mask[0]][:, :, atom_mask[0]]

    assert dclamp is not None or dclamp_2d is not None, "need to provide either dclamp or dclamp_2d to compute_general_FAPE"
    assert not (dclamp is not None and dclamp_2d is not None), "you provided both dclamp and dclamp_2d, please only provide one"

    if dclamp_2d is not None:
        dclamp = dclamp_2d[:,frame_mask[0]][:, :, atom_mask[0]]

    loss = (1.0/Z) * (torch.clamp(diff, max=dclamp)).mean(dim=(1,2))

    pae_loss = compute_pae_loss(X, X_y, uX, Y, Y_y, uY, logit_pae, frame_mask, atom_mask, frame_atom_mask_2d=frame_atom_mask_2d) \
                   if logit_pae is not None \
                   else torch.tensor(0).to(frames.device)
    pde_loss = compute_pde_loss(X, Y, logit_pde, atom_mask, frame_atom_mask_2d=frame_atom_mask_2d) if logit_pde is not None \
                   else torch.tensor(0).to(frames.device)

    return loss, pae_loss, pde_loss

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

    frame_mask_update = frame_mask.clone()
    # reindex frames for flat X
    frames_reindex = torch.zeros(frames.shape[:-1], device=frames.device)
    for i in range(L):
        frames_reindex[:, i, :, :] = (i+frames[..., i, :, :, 0])*natoms + frames[..., i, :, :, 1]
    frames_reindex = frames_reindex.long()

    masked_atom_frames = torch.any(frames_reindex>L*natoms, dim=-1) # find frames with atoms that aren't resolved
    masked_atom_frames *= torch.any(frames_reindex<0, dim=-1)
    frame_mask_update *= ~masked_atom_frames 
    # There are currently indices for frames that aren't in the coordinates bc they arent resolved, reset these indices to 0 to avoid 
    # indexing errors
    frames_reindex[masked_atom_frames, :] = 0 

    frame_mask_update *= torch.all(
        torch.gather(atom_mask.reshape(1, L*natoms),1,frames_reindex.reshape(1,L*NFRAMES*3)).reshape(1,L,-1,3),
        axis=-1)

    return frames_reindex, frame_mask_update

def calc_crd_rmsd(pred, true, atom_mask, rmsd_mask=None, alignment_radius=None):
    '''
    Calculate coordinate RMSD
    Input:
        - pred: predicted coordinates (B, L, natoms, 3)
        - true: true coordinates (B, L, natoms, 3)
        - atom_mask: mask for coordinates used for alignment (B, L, natoms)
        - rmsd_mask: mask for coordinates used for rmsd calculation
        - alignment_radius: radius around the rmsd mask that will be used for alignment (float)
    Output: RMSD after superposition
    '''
    def rmsd(V, W, eps=1e-6):
        L = V.shape[1]
        return torch.sqrt(torch.sum((V-W)*(V-W), dim=(1,2)) / L + eps)
    def centroid(X):
        return X.mean(dim=-2, keepdim=True)
    
    if rmsd_mask is None:
        rmsd_mask = atom_mask.clone()
    if alignment_radius is None:
        alignment_radius = torch.inf
    dist = torch.cdist(true[atom_mask][None], true[rmsd_mask][None])
    in_radius = (dist<alignment_radius).any(dim=-1)[0] # shape: (num seen atoms in atom_mask)

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
    C = torch.matmul(pred_allatom_origin.permute(0,2,1), true_allatom_origin)
    
    # Compute optimal rotation matrix using SVD
    V, S, W = torch.svd(C)
    
    # get sign to ensure right-handedness
    d = torch.ones([B,3,3], device=pred.device)
    d[:,:,-1] = torch.sign(torch.det(V)*torch.det(W)).unsqueeze(1)
    
    # Rotation matrix U
    U = torch.matmul(d*V, W.permute(0,2,1)) # (IB, 3, 3)
    
    pred_rms = pred[rmsd_mask][None] - centroid(pred_allatom)
    true_rms = true[rmsd_mask][None] - centroid(true_allatom)
    # Rotate pred
    rP = torch.matmul(pred_rms, U) # (IB, L*3, 3)
    
    # get RMS
    rms = rmsd(rP, true_rms).reshape(B)
    return rms
    
def angle(a, b, c, eps=1e-6):
    '''
    Calculate cos/sin angle between ab and cb
    a,b,c have shape of (B, L, 3)
    '''
    B,L = a.shape[:2]

    u1 = a-b
    u2 = c-b

    u1_norm = torch.norm(u1, dim=-1, keepdim=True) + eps
    u2_norm = torch.norm(u2, dim=-1, keepdim=True) + eps

    # normalize u1 & u2 --> make unit vector
    u1 = u1 / u1_norm
    u2 = u2 / u2_norm
    u1 = u1.reshape(B*L, 3)
    u2 = u2.reshape(B*L, 3)

    # sin_theta = norm(a cross b)/(norm(a)*norm(b))
    # cos_theta = norm(a dot b) / (norm(a)*norm(b))
    sin_theta = torch.norm(torch.cross(u1, u2, dim=1), dim=1, keepdim=True).reshape(B, L, 1) # (B,L,1)
    cos_theta = torch.matmul(u1[:,None,:], u2[:,:,None]).reshape(B, L, 1)
    
    return torch.cat([cos_theta, sin_theta], axis=-1) # (B, L, 2)

def length(a, b):
    return torch.norm(a-b, dim=-1)

def torsion(a,b,c,d, eps=1e-6):
    #A function that takes in 4 atom coordinates:
    # a - [B,L,3]
    # b - [B,L,3]
    # c - [B,L,3]
    # d - [B,L,3]
    # and returns cos and sin of the dihedral angle between those 4 points in order a, b, c, d
    # output - [B,L,2]
    u1 = b-a
    u1 = u1 / (torch.norm(u1, dim=-1, keepdim=True) + eps)
    u2 = c-b
    u2 = u2 / (torch.norm(u2, dim=-1, keepdim=True) + eps)
    u3 = d-c
    u3 = u3 / (torch.norm(u3, dim=-1, keepdim=True) + eps)
    #
    t1 = torch.cross(u1, u2, dim=-1) #[B, L, 3]
    t2 = torch.cross(u2, u3, dim=-1)
    t1_norm = torch.norm(t1, dim=-1, keepdim=True)
    t2_norm = torch.norm(t2, dim=-1, keepdim=True)
    
    cos_angle = torch.matmul(t1[:,:,None,:], t2[:,:,:,None])[:,:,0]
    sin_angle = torch.norm(u2, dim=-1,keepdim=True)*(torch.matmul(u1[:,:,None,:], t2[:,:,:,None])[:,:,0])
    
    cos_sin = torch.cat([cos_angle, sin_angle], axis=-1)/(t1_norm*t2_norm+eps) #[B,L,2]
    return cos_sin

# ideal N-C distance, ideal cos(CA-C-N angle), ideal cos(C-N-CA angle)
# for NA, we do not compute this as it is not computable from the stubs alone
def calc_BB_bond_geom(
    seq, pred, idx, eps=1e-6, 
    ideal_NC=1.329, ideal_CACN=-0.4415, ideal_CNCA=-0.5255, 
    ideal_OP=1.607, ideal_OPO=-0.3106, ideal_OPC=-0.4970, 
    sig_len=0.02, sig_ang=0.05):
    '''
    Calculate backbone bond geometry (bond length and angle) and put loss on them
    Input:
     - pred: predicted coords (B, L, :, 3), 0; N / 1; CA / 2; C
     - true: True coords (B, L, :, 3)
    Output:
     - bond length loss, bond angle loss
    '''
    def cosangle( A,B,C ):
        AB = A-B
        BC = C-B
        ABn = torch.sqrt( torch.sum(torch.square(AB),dim=-1) + eps)
        BCn = torch.sqrt( torch.sum(torch.square(BC),dim=-1) + eps)
        return torch.clamp(torch.sum(AB*BC,dim=-1)/(ABn*BCn), -0.999,0.999)

    B, L = pred.shape[:2]

    bonded = (idx[:,1:] - idx[:,:-1])==1
    is_prot = is_protein(seq)[:-1]
    is_NA = is_nucleic(seq)[:-1]

    # bond length: N-CA, CA-C, C-N
    blen_CN_pred  = length(pred[:,:-1,2], pred[:,1:,0]).reshape(B,L-1) # (B, L-1)
    CN_loss = bonded*is_prot*torch.clamp( torch.square(blen_CN_pred - ideal_NC) - sig_len**2, min=0.0 )
    n_viol = (CN_loss > 0.0).sum()
    blen_loss = CN_loss.sum() / (n_viol + eps)

    # bond length: P-O
    blen_OP_pred  = length(pred[:,:-1,8], pred[:,1:,1]).reshape(B,L-1) # (B, L-1)
    OP_loss = bonded*is_NA*torch.clamp( torch.square(blen_OP_pred - ideal_OP) - sig_len**2, min=0.0 )
    n_viol = (OP_loss > 0.0).sum()
    blen_loss += OP_loss.sum() / (n_viol + eps)

    # bond angle: CA-C-N, C-N-CA
    bang_CACN_pred = cosangle(pred[:,:-1,1], pred[:,:-1,2], pred[:,1:,0]).reshape(B,L-1)
    bang_CNCA_pred = cosangle(pred[:,:-1,2], pred[:,1:,0], pred[:,1:,1]).reshape(B,L-1)
    CACN_loss = bonded*is_prot*torch.clamp( torch.square(bang_CACN_pred - ideal_CACN) - sig_ang**2,  min=0.0 )
    CNCA_loss = bonded*is_prot*torch.clamp( torch.square(bang_CNCA_pred - ideal_CNCA) - sig_ang**2,  min=0.0 )
    bang_loss_prot = CACN_loss + CNCA_loss
    n_viol = (bang_loss_prot > 0.0).sum()
    bang_loss = bang_loss_prot.sum() / (n_viol+eps)

    # bond angle: O1PO, O2PO, OPC
    bang_O1PO_pred = cosangle(pred[:,:-1,8], pred[:,1:,1], pred[:,1:,0]).reshape(B,L-1)
    bang_O2PO_pred = cosangle(pred[:,:-1,8], pred[:,1:,1], pred[:,1:,2]).reshape(B,L-1)
    bang_OPC_pred  = cosangle(pred[:,:-1,7], pred[:,:-1,8], pred[:,1:,1]).reshape(B,L-1)
    O1PO_loss = bonded*is_NA*torch.clamp( torch.square(bang_O1PO_pred - ideal_OPO) - sig_ang**2,  min=0.0 )
    O2PO_loss = bonded*is_NA*torch.clamp( torch.square(bang_O2PO_pred - ideal_OPO) - sig_ang**2,  min=0.0 )
    OPC_loss = bonded*is_NA*torch.clamp( torch.square(bang_OPC_pred - ideal_OPC) - sig_ang**2,  min=0.0 )
    bang_loss_na = O1PO_loss + O2PO_loss + OPC_loss
    n_viol = (bang_loss_na > 0.0).sum()
    bang_loss += bang_loss_na.sum() / (n_viol+eps)

    return blen_loss + bang_loss

def calc_atom_bond_loss(pred, true, bond_feats, seq, beta=0.2, eps=1e-6):
    """
    loss on distances between bonded atoms
    """
    loss_func_sum = torch.nn.SmoothL1Loss(reduction='sum', beta=beta)
    loss_func_mean = torch.nn.SmoothL1Loss(reduction='mean', beta=beta)

    # intra-ligand bonds
    atom_bonds = (bond_feats>0)*(bond_feats < 5)
    b, i, j = torch.where(atom_bonds>0)
    nat_dist = torch.sum(torch.square(true[:,i,1]-true[:,j,1]),dim=-1)
    pred_dist = torch.sum(torch.square(pred[:,i,1]-pred[:,j,1]),dim=-1)
    #lig_dist_loss = torch.sum(torch.clamp(torch.square(nat_dist-pred_dist), max=clamp)) # from EquiBind
    lig_dist_loss = loss_func_sum(nat_dist, pred_dist)

    # bonds between protein residues and ligand atoms (i.e. atomized protein)
    inter_bonds = bond_feats==6 
    _, i, j = torch.where(inter_bonds)
    a = (seq[:,i]<22) & (seq[:,j]==39) # res N - atom C: binary indicator
    b = (seq[:,i]<22) & (seq[:,j]==55) # res C - atom N
    c = (seq[:,i]==39) & (seq[:,j]<22) # atom C - res N
    d = (seq[:,i]==55) & (seq[:,j]<22) # atom N - res C
    i_atom = 0*a + 2*b + 1*c + 1*d # (B, N_bonds) : indexes of atom that is bonded (N:0, C:2, 1:ligand atom)
    j_atom = 1*a + 1*b + 0*c + 2*d # (B, N_bonds)
    nat_dist = torch.sum(torch.square(true[0,i,i_atom[0],:]-true[0,j,j_atom[0],:]), dim=-1) # assumes B=1
    pred_dist = torch.sum(torch.square(pred[0,i,i_atom[0],:]-pred[0,j,j_atom[0],:]), dim=-1)
    #inter_dist_loss = torch.sum(torch.clamp(torch.square(nat_dist-pred_dist), max=clamp))
    inter_dist_loss = loss_func_sum(nat_dist, pred_dist)

    bond_dist_loss = (lig_dist_loss + inter_dist_loss)/(atom_bonds.sum() + inter_bonds.sum() + eps)

    # enforce LAS constraints between atoms 2 bonds away and aromatic groups
    atom_bonds_np = atom_bonds[0].cpu().numpy()
    G = nx.from_numpy_matrix(atom_bonds_np)
    paths = find_all_paths_of_length_n(G,2)
    if paths:
        paths = torch.tensor(paths, device=pred.device)
        nat_dist = torch.sum(torch.square(true[:,paths[:,0],1]-true[:,paths[:,2],1]),dim=-1)
        pred_dist = torch.sum(torch.square(pred[:,paths[:,0],1]-pred[:,paths[:,2],1]),dim=-1)
        #skip_bond_dist_loss = torch.sum(torch.clamp(torch.square(nat_dist-pred_dist),max=clamp))/(paths.shape[0]+eps)
        skip_bond_dist_loss = loss_func_mean(nat_dist, pred_dist)
    else:
        skip_bond_dist_loss = torch.tensor(0, device=pred.device)
    rigid_groups = find_all_rigid_groups(bond_feats)
    if rigid_groups != None:
        nat_dist = torch.sum(torch.square(true[:,rigid_groups[:,0],1]-true[:,rigid_groups[:,1],1]),dim=-1)
        pred_dist = torch.sum(torch.square(pred[:,rigid_groups[:,0],1]-pred[:,rigid_groups[:,1],1]),dim=-1)
        #rigid_group_dist_loss = torch.sum(torch.clamp(torch.square(nat_dist-pred_dist),max=clamp))/(rigid_groups.shape[0]+eps)
        rigid_group_dist_loss = loss_func_mean(nat_dist, pred_dist)
    else:
        rigid_group_dist_loss = torch.tensor(0, device=pred.device)

    return bond_dist_loss, skip_bond_dist_loss, rigid_group_dist_loss

def calc_cart_bonded(seq, pred, idx, len_param, ang_param, tor_param, eps=1e-6):
    # pred: N x L x 27 x 3
    # idx: 1 x L
    # seq: 1 x L
    def gen_ang( A,B,C ):
        AB = A-B
        BC = C-B
        ABn = torch.sqrt( torch.sum(torch.square(AB),dim=-1) + eps)
        BCn = torch.sqrt( torch.sum(torch.square(BC),dim=-1) + eps)
        return torch.acos( torch.clamp(torch.sum(AB*BC,dim=-1)/(ABn*BCn), -0.999,0.999) )

    # quadratic from [-1,1], linear elsewhere
    def boundfunc(X):
        Y = torch.abs(X)
        Y[Y<1.0] = torch.square(Y[Y<1.0])
        #Y = torch.square(X)
        return Y

    N,L = pred.shape[:2]
    cb_loss = torch.zeros(N, device=pred.device)

    ## intra-res
    cblens = len_param[seq]
    len_idx = cblens[...,:2].to(torch.long).reshape(1,L,-1,1).repeat(N,1,1,3)
    len_all = torch.gather(pred, 2, len_idx).reshape(N,L,-1,2,3)
    len_mask = cblens[...,0]!=cblens[...,1]
    E_cb_len = (
        len_mask[None,...] * 
        cblens[None,...,3] *
        boundfunc( length(len_all[...,0,:],len_all[...,1,:]) - cblens[...,2] )
    ).sum(dim=(0,3)) / len_mask.sum()

    # figure out which his are his_d
    cblens[seq==8] = len_param[-1]
    len_idx = cblens[...,:2].to(torch.long).reshape(1,L,-1,1).repeat(N,1,1,3)
    len_all_a = torch.gather(pred, 2, len_idx).reshape(N,L,-1,2,3)
    len_mask_a = cblens[...,0]!=cblens[...,1]
    E_cb_len_a = (
        len_mask_a[None,...] * 
        cblens[None,...,3] *
        boundfunc( length(len_all_a[...,0,:],len_all_a[...,1,:]) - cblens[...,2] )
    ).sum(dim=(0,3)) / len_mask.sum() # N,L
    is_his_d = (seq==8)*(E_cb_len_a<E_cb_len)

    cb_loss += torch.min(E_cb_len_a,E_cb_len).sum(dim=1)

    cbangs = ang_param[seq].repeat(N,1,1,1)
    cbangs[is_his_d] = ang_param[-1]
    ang_idx = cbangs[...,:3].to(torch.long).reshape(N,L,-1,1).repeat(1,1,1,3)
    ang_all = torch.gather(pred, 2, ang_idx).reshape(N,L,-1,3,3)
    ang_mask = cbangs[...,0]!=cbangs[...,1]
    E_cb_ang = (
        ang_mask[None,...] * 
        cbangs[None,...,4] *
        boundfunc( get_ang(ang_all[...,0,:],ang_all[...,1,:],ang_all[...,2,:]) - cbangs[None,...,3] )
    ).sum(dim=(0,2,3)) / ang_mask.sum()
    cb_loss += E_cb_ang

    cbtors = tor_param[seq].repeat(N,1,1,1)
    cbtors[is_his_d] = tor_param[-1]
    tor_idx = cbtors[...,:4].to(torch.long).reshape(N,L,-1,1).repeat(1,1,1,3)
    tor_all = torch.gather(pred, 2, tor_idx).reshape(N,L,-1,4,3)
    tor_mask = cbtors[...,0]!=cbtors[...,1]
    offset = 2*np.pi/cbtors[None,...,6]
    tor_deltas = (
        get_dih( 
            tor_all[...,0,:],tor_all[...,1,:],tor_all[...,2,:],tor_all[...,3,:]
        ) - cbtors[None,...,4] + 0.5*offset
    ) % offset - 0.5*offset

    dihs = get_dih( 
            tor_all[...,0,:],tor_all[...,1,:],tor_all[...,2,:],tor_all[...,3,:]
        )

    E_cb_tor = (
        tor_mask[None,...] * 
        cbtors[None,...,5] *
        boundfunc( tor_deltas )
    ).sum(dim=(0,2,3)) / tor_mask.sum()
    cb_loss += E_cb_tor

    # inter-res
    # bond length: C-N
    bonded = (idx[:,1:] - idx[:,:-1])==1
    blen_CN_pred  = length(pred[:,:-1,2], pred[:,1:,0]).reshape(N,L-1) # (B, L-1)
    CN_loss = cb_lengths_CN[1] * boundfunc(blen_CN_pred - cb_lengths_CN[0])
    cb_loss += (bonded*CN_loss).sum(dim=1) / (bonded.sum())

    # bond angle: CA-C-N, C-N-CA
    bang_CACN_pred = get_ang(pred[:,:-1,2], pred[:,1:,0], pred[:,1:,1]).reshape(N,L-1)
    CACN_loss = cb_angles_CACN[1] * boundfunc(bang_CACN_pred - cb_angles_CACN[0])
    cb_loss += (bonded*CACN_loss).sum(dim=1) / (bonded.sum())

    bang_CNCA_pred = get_ang(pred[:,:-1,2], pred[:,1:,0], pred[:,1:,1]).reshape(N,L-1)
    CNCA_loss = cb_angles_CNCA[1] * boundfunc(bang_CNCA_pred - cb_angles_CNCA[0])
    cb_loss += (bonded*CNCA_loss).sum(dim=1) / (bonded.sum())

    # improper torsions CA-C-N-H (CD-C-N-CA), CA-N-C-O
    # planarity around N (H for non-pro, CD for pro)
    atom4idx = torch.full_like(seq, 14)
    atom4idx[seq==14] = 6 # set to CD for proline
    atom4 = torch.gather( pred, 2, atom4idx[:,:,None,None].repeat(1,1,1,3) )
    btor_CACNH_delta = (
        get_dih(
            pred[:,:-1,1], pred[:,:-1,2], pred[:,1:,0], atom4[:,1:,0]
        ) - cb_torsions_CACNH[0] + np.pi/2
    ) % np.pi - np.pi/2
    CACNH_loss = cb_torsions_CACNH[1] * boundfunc( btor_CACNH_delta )
    cb_loss += (bonded*CACNH_loss).sum(dim=1) / (bonded.sum())

    # planarity around C
    btor_CANCO_delta = (
        get_dih(
            pred[:,:-1,1], pred[:,1:,0], pred[:,:-1,2], pred[:,:-1,3]
        ) - cb_torsions_CANCO[0] + np.pi/2
    ) % np.pi - np.pi/2
    CANCO_loss = cb_torsions_CANCO[1] * boundfunc( btor_CANCO_delta )
    cb_loss += (bonded*CANCO_loss).sum(dim=1) / (bonded.sum())

    return cb_loss

# AF2-like version of clash score
def calc_clash(xs, mask):
    DISTCUT=2.0 # (d_lit - tau) from AF2 MS
    L = xs.shape[0]
    dij = torch.sqrt(
        torch.sum( torch.square( xs[:,:,None,None,:]-xs[None,None,:,:,:] ), dim=-1 ) + 1e-8
    )

    allmask = mask[:,:,None,None]*mask[None,None,:,:]
    allmask[torch.arange(L),:,torch.arange(L),:] = False # ignore res-self
    allmask[torch.arange(1,L),0,torch.arange(L-1),2] = False # ignore N->C
    allmask[torch.arange(L-1),2,torch.arange(1,L),0] = False # ignore N->C

    clash = torch.sum( torch.clamp(DISTCUT-dij[allmask],0.0) ) / torch.sum(mask)
    return clash


#fd more efficient LJ loss
class LJLoss(torch.autograd.Function):
    @staticmethod
    def ljVdV(deltas, sigma, epsilon, lj_lin, eps):
        # deltas - (N,natompair,3)
        N = deltas.shape[0]

        dist = torch.sqrt( torch.sum ( torch.square( deltas ), dim=-1 ) + eps )
        linpart = dist<lj_lin*sigma[None]
        deff = dist.clone()
        deff[linpart] = lj_lin*sigma.repeat(N,1)[linpart]
        sd = sigma / deff
        sd2 = sd*sd
        sd6 = sd2 * sd2 * sd2
        sd12 = sd6 * sd6
        ljE = epsilon * (sd12 - 2 * sd6)
        ljE[linpart] += epsilon.repeat(N,1)[linpart] * (
            -12 * sd12[linpart]/deff[linpart] + 12 * sd6[linpart]/deff[linpart]
        ) * (dist[linpart]-deff[linpart])

        # works for linpart too
        dljEdd_over_r = epsilon * (-12 * sd12/deff + 12 * sd6/deff) / (dist)

        return ljE.sum(dim=-1), dljEdd_over_r

    @staticmethod
    def forward(
        ctx, xs, seq, aamask, bond_feats, dist_matrix, ljparams, ljcorr, num_bonds, 
        lj_lin=0.75, lj_hb_dis=3.0, lj_OHdon_dis=2.6, lj_hbond_hdis=1.75, 
        eps=1e-8, training=True
    ):
        N, L, A = xs.shape[:3]
        assert (N==1) # see comment below

        ds_res = torch.sqrt( torch.sum ( torch.square( 
            xs.detach()[:,:,None,1,:]-xs.detach()[:,None,:,1,:]), dim=-1 ))
        rs = torch.triu_indices(L,L,0, device=xs.device)
        ri,rj = rs[0],rs[1]

        # batch during inference for huge systems
        BATCHSIZE = len(ri)
        if (not training):
            BATCHSIZE = 65536/N

        ljval = 0
        dljEdx = torch.zeros_like(xs, dtype=torch.float)

        for i_batch in range((len(ri)-1)//BATCHSIZE + 1):
            idx = torch.arange(
                i_batch*BATCHSIZE, 
                min( (i_batch+1)*BATCHSIZE, len(ri)),
                device=xs.device
            )
            rii,rjj = ri[idx],rj[idx] # residue pairs we consider

            ridx,ai,aj = (
                aamask[seq[rii]][:,:,None]*aamask[seq[rjj]][:,None,:]
            ).nonzero(as_tuple=True)

            deltas = xs[:,rii,:,None,:]-xs[:,rjj,None,:,:] # N,BATCHSIZE,Natm,Natm,3
            seqi,seqj = seq[rii[ridx]], seq[rjj[ridx]]

            mask = torch.ones_like(ridx, dtype=torch.bool) # are atoms defined?

            # mask out atom pairs from too-distant residues (C-alpha dist > 24A)
            ca_dist = torch.linalg.norm(deltas[:,:,1,1],dim=-1)
            mask *= (ca_dist[:,ridx]<24).any(dim=0)  # will work for batch>1 but very inefficient

            intrares = (rii[ridx]==rjj[ridx])
            mask[intrares*(ai<aj)] = False  # upper tri (atoms)

            ## count-pair
            # a) intra-protein
            mask[intrares] *= num_bonds[seqi[intrares],ai[intrares],aj[intrares]]>=4
            pepbondres = ri[ridx]+1==rj[ridx]
            mask[pepbondres] *= (
                num_bonds[seqi[pepbondres],ai[pepbondres],2]
                + num_bonds[seqj[pepbondres],0,aj[pepbondres]]
                + 1) >=4

            # b) intra-ligand
            atommask = (ai==1)*(aj==1)
            dist_matrix = torch.nan_to_num(dist_matrix, posinf=4.0) #NOTE: need to run nan_to_num to remove infinities
            resmask = (dist_matrix[0,rii,rjj] >= 4) # * will only work for batch=1
            mask[atommask] *= resmask[ ridx[atommask] ]

            # c) protein/ligand
            ##fd NOTE1: changed 6->5 in masking (atom 5 is CG which should always be 4+ bonds away from connected atom)
            ##fd NOTE2: this does NOT work correctly for nucleic acids
            ##fd     for NAs atoms 0-4 are masked, but also 5,7,8 and 9 should be masked!
            bbatommask = (ai<5)*(aj<5)
            resmask = (bond_feats[0,rii,rjj] != 6) # * will only work for batch=1
            mask[bbatommask] *= resmask[ ridx[bbatommask] ]

            # apply mask.  only interactions to be scored remain
            ai,aj,seqi,seqj,ridx = ai[mask],aj[mask],seqi[mask],seqj[mask],ridx[mask]
            deltas = deltas[:,ridx,ai,aj]

            # hbond correction
            use_hb_dis = (
                ljcorr[seqi,ai,0]*ljcorr[seqj,aj,1] 
                + ljcorr[seqi,ai,1]*ljcorr[seqj,aj,0] ).nonzero()
            use_ohdon_dis = ( # OH are both donors & acceptors
                ljcorr[seqi,ai,0]*ljcorr[seqi,ai,1]*ljcorr[seqj,aj,0] 
                +ljcorr[seqi,ai,0]*ljcorr[seqj,aj,0]*ljcorr[seqj,aj,1] 
            ).nonzero()
            use_hb_hdis = (
                ljcorr[seqi,ai,2]*ljcorr[seqj,aj,1] 
                +ljcorr[seqi,ai,1]*ljcorr[seqj,aj,2] 
            ).nonzero()

            # disulfide correction
            potential_disulf = (ljcorr[seqi,ai,3]*ljcorr[seqj,aj,3] ).nonzero()

            ljrs = ljparams[seqi,ai,0] + ljparams[seqj,aj,0]
            ljrs[use_hb_dis] = lj_hb_dis
            ljrs[use_ohdon_dis] = lj_OHdon_dis
            ljrs[use_hb_hdis] = lj_hbond_hdis

            ljss = torch.sqrt( ljparams[seqi,ai,1] * ljparams[seqj,aj,1] + eps )
            ljss [potential_disulf] = 0.0

            natoms = torch.sum(aamask[seq])
            ljval_i,dljEdd_i = LJLoss.ljVdV(deltas,ljrs,ljss,lj_lin,eps)

            ljval += ljval_i / natoms

            # sum per-atom-pair grads into per-atom grads
            # note this is stochastic op on GPU
            idxI,idxJ = rii[ridx]*A + ai, rjj[ridx]*A + aj

            dljEdx.view(N,-1,3).index_add_(1, idxI, dljEdd_i[...,None]*deltas, alpha=1.0/natoms)
            dljEdx.view(N,-1,3).index_add_(1, idxJ, dljEdd_i[...,None]*deltas, alpha=-1.0/natoms)

        ctx.save_for_backward(dljEdx)

        return ljval



    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        dljEdx, = ctx.saved_tensors
        return (
            grad_output * dljEdx, 
            None, None, None, None, None, None, None, None, None, None, None, None, None
        )

# Rosetta-like version of LJ (fa_atr+fa_rep)
#   lj_lin is switch from linear to 12-6.  Smaller values more sharply penalize clashes
def calc_lj(
    seq, xs, aamask, bond_feats, dist_matrix, ljparams, ljcorr, num_bonds,  
    lj_lin=0.75, lj_hb_dis=3.0, lj_OHdon_dis=2.6, lj_hbond_hdis=1.75, 
    lj_maxrad=-1.0, eps=1e-8,
    training=True
):
    lj = LJLoss.apply
    ljval = lj(
        xs, seq, aamask, bond_feats, dist_matrix, ljparams, ljcorr, num_bonds, 
        lj_lin, lj_hb_dis, lj_OHdon_dis, lj_hbond_hdis, eps, training)

    return ljval


def calc_hb(
    seq, xs, aamask, hbtypes, hbbaseatoms, hbpolys,
    hb_sp2_range_span=1.6, hb_sp2_BAH180_rise=0.75, hb_sp2_outer_width=0.357, 
    hb_sp3_softmax_fade=2.5, threshold_distance=6.0, eps=1e-8, normalize=True
):
    def evalpoly( ds, xrange, yrange, coeffs ):
        v = coeffs[...,0]
        for i in range(1,10):
            v = v * ds + coeffs[...,i]
        minmask = ds<xrange[...,0]
        v[minmask] = yrange[minmask][...,0]
        maxmask = ds>xrange[...,1]
        v[maxmask] = yrange[maxmask][...,1]
        return v
    
    def cosangle( A,B,C ):
        AB = A-B
        BC = C-B
        ABn = torch.sqrt( torch.sum(torch.square(AB),dim=-1) + eps)
        BCn = torch.sqrt( torch.sum(torch.square(BC),dim=-1) + eps)
        return torch.clamp(torch.sum(AB*BC,dim=-1)/(ABn*BCn), -0.999,0.999)

    hbts = hbtypes[seq]
    hbba = hbbaseatoms[seq]

    rh,ah = (hbts[...,0]>=0).nonzero(as_tuple=True)
    ra,aa = (hbts[...,1]>=0).nonzero(as_tuple=True)
    D_xs = xs[rh,hbba[rh,ah,0]][:,None,:]
    H_xs = xs[rh,ah][:,None,:]
    A_xs = xs[ra,aa][None,:,:]
    B_xs = xs[ra,hbba[ra,aa,0]][None,:,:]
    B0_xs = xs[ra,hbba[ra,aa,1]][None,:,:]
    hyb = hbts[ra,aa,2]
    polys = hbpolys[hbts[rh,ah,0][:,None],hbts[ra,aa,1][None,:]]

    AH = torch.sqrt( torch.sum( torch.square( H_xs-A_xs), axis=-1) + eps )
    AHD = torch.acos( cosangle( B_xs, A_xs, H_xs) )
    
    Es = polys[...,0,0]*evalpoly(
        AH,polys[...,0,1:3],polys[...,0,3:5],polys[...,0,5:])
    Es += polys[...,1,0] * evalpoly(
        AHD,polys[...,1,1:3],polys[...,1,3:5],polys[...,1,5:])

    Bm = 0.5*(B0_xs[:,hyb==HbHybType.RING]+B_xs[:,hyb==HbHybType.RING])
    cosBAH = cosangle( Bm, A_xs[:,hyb==HbHybType.RING], H_xs )
    Es[:,hyb==HbHybType.RING] += polys[:,hyb==HbHybType.RING,2,0] * evalpoly(
        cosBAH, 
        polys[:,hyb==HbHybType.RING,2,1:3], 
        polys[:,hyb==HbHybType.RING,2,3:5], 
        polys[:,hyb==HbHybType.RING,2,5:])

    cosBAH1 = cosangle( B_xs[:,hyb==HbHybType.SP3], A_xs[:,hyb==HbHybType.SP3], H_xs )
    cosBAH2 = cosangle( B0_xs[:,hyb==HbHybType.SP3], A_xs[:,hyb==HbHybType.SP3], H_xs )
    Esp3_1 = polys[:,hyb==HbHybType.SP3,2,0] * evalpoly(
        cosBAH1, 
        polys[:,hyb==HbHybType.SP3,2,1:3], 
        polys[:,hyb==HbHybType.SP3,2,3:5], 
        polys[:,hyb==HbHybType.SP3,2,5:])
    Esp3_2 = polys[:,hyb==HbHybType.SP3,2,0] * evalpoly(
        cosBAH2, 
        polys[:,hyb==HbHybType.SP3,2,1:3], 
        polys[:,hyb==HbHybType.SP3,2,3:5], 
        polys[:,hyb==HbHybType.SP3,2,5:])
    Es[:,hyb==HbHybType.SP3] += torch.log(
        torch.exp(Esp3_1 * hb_sp3_softmax_fade)
        + torch.exp(Esp3_2 * hb_sp3_softmax_fade)
    ) / hb_sp3_softmax_fade

    cosBAH = cosangle( B_xs[:,hyb==HbHybType.SP2], A_xs[:,hyb==HbHybType.SP2], H_xs )
    Es[:,hyb==HbHybType.SP2] += polys[:,hyb==HbHybType.SP2,2,0] * evalpoly(
        cosBAH, 
        polys[:,hyb==HbHybType.SP2,2,1:3], 
        polys[:,hyb==HbHybType.SP2,2,3:5], 
        polys[:,hyb==HbHybType.SP2,2,5:])

    BAH = torch.acos( cosBAH )
    B0BAH = get_dih(B0_xs[:,hyb==HbHybType.SP2], B_xs[:,hyb==HbHybType.SP2], A_xs[:,hyb==HbHybType.SP2], H_xs)

    d,m,l = hb_sp2_BAH180_rise, hb_sp2_range_span, hb_sp2_outer_width
    Echi = torch.full_like( B0BAH, m-0.5 )

    mask1 = BAH>np.pi * 2.0 / 3.0
    H = 0.5 * (torch.cos(2 * B0BAH) + 1)
    F = d / 2 * torch.cos(3 * (np.pi - BAH[mask1])) + d / 2 - 0.5
    Echi[mask1] = H[mask1] * F + (1 - H[mask1]) * d - 0.5

    mask2 = BAH>np.pi * (2.0 / 3.0 - l)
    mask2 *= ~mask1
    outer_rise = torch.cos(np.pi - (np.pi * 2 / 3 - BAH[mask2]) / l)
    F = m / 2 * outer_rise + m / 2 - 0.5
    G = (m - d) / 2 * outer_rise + (m - d) / 2 + d - 0.5
    Echi[mask2] = H[mask2] * F + (1 - H[mask2]) * d - 0.5

    Es[:,hyb==HbHybType.SP2] += polys[:,hyb==HbHybType.SP2,2,0] * Echi

    tosquish = torch.logical_and(Es > -0.1,Es < 0.1)
    Es[tosquish] = -0.025 + 0.5 * Es[tosquish] - 2.5 * torch.square(Es[tosquish])
    Es[Es > 0.1] = 0.
    if (normalize):
        return (torch.sum( Es ) / torch.sum(aamask[seq]))
    else:
        return torch.sum( Es )

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
    pred_dih = get_dih(chiral_dih[...,0, :], chiral_dih[...,1, :], chiral_dih[...,2, :], chiral_dih[...,3, :]) # n_symm, b, n, 36, 3
    l = torch.square(pred_dih-chirals[...,-1]).mean()
    return l

@torch.enable_grad()
def calc_BB_bond_geom_grads(
    seq, idx, xyz, alpha, toaa, eps=1e-6,
    ideal_NC=1.329, ideal_CACN=-0.4415, ideal_CNCA=-0.5255, 
    ideal_OP=1.607, ideal_POP=-0.3106, ideal_OPC=-0.4970, 
    sig_len=0.02, sig_ang=0.05
):
    pred.requires_grad_(True)
    Ebond = calc_BB_bond_geom(seq, pred, idx, eps, ideal_NC, ideal_CACN, ideal_CNCA, sig_len, sig_ang)
    return torch.autograd.grad(Ebond, pred)

@torch.enable_grad()
def calc_cart_bonded_grads(seq, pred, idx, len_param, ang_param, tor_param, eps=1e-6):
    pred.requires_grad_(True)
    Ecb = calc_cart_bonded(seq, pred, idx, len_param, ang_param, tor_param, eps)
    return torch.autograd.grad(Ecb, pred)

@torch.enable_grad()
def calc_ljallatom_grads(
    seq, xyzaa, 
    aamask, bond_feats, dist_matrix, 
    ljparams, ljcorr, num_bonds, 
    lj_lin=0.85, lj_hb_dis=3.0, lj_OHdon_dis=2.6, lj_hbond_hdis=1.75, 
    lj_maxrad=-1.0, eps=1e-8
):
    xyzaa.requires_grad_(True)
    Elj = calc_lj(
        seq[0], 
        xyzaa[...,:3], 
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
        lj_maxrad,
        eps
    )
    return torch.autograd.grad(Elj, (xyzaa,))

@torch.enable_grad()
def calc_lj_grads(
    seq, xyz, alpha, toaa, bond_feats, dist_matrix, 
    aamask, ljparams, ljcorr, num_bonds, 
    lj_lin=0.85, lj_hb_dis=3.0, lj_OHdon_dis=2.6, lj_hbond_hdis=1.75, 
    lj_maxrad=-1.0, eps=1e-8
):
    xyz.requires_grad_(True)
    alpha.requires_grad_(True)
    _, xyzaa = toaa(seq, xyz, alpha)
    Elj = calc_lj(
        seq[0], 
        xyzaa[...,:3], 
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
        lj_maxrad,
        eps
    )
    return torch.autograd.grad(Elj, (xyz,alpha))

@torch.enable_grad()
def calc_hb_grads(
    seq, xyz, alpha, toaa, 
    aamask, hbtypes, hbbaseatoms, hbpolys,
    hb_sp2_range_span=1.6, hb_sp2_BAH180_rise=0.75, hb_sp2_outer_width=0.357, 
    hb_sp3_softmax_fade=2.5, threshold_distance=6.0, eps=1e-8, normalize=True
):
    xyz.requires_grad_(True)
    alpha.requires_grad_(True)
    _, xyzaa = toaa(seq, xyz, alpha)
    Ehb = calc_hb(
        seq, 
        xyzaa[0,...,:3], 
        aamask, 
        hbtypes, 
        hbbaseatoms, 
        hbpolys,
        hb_sp2_range_span,
        hb_sp2_BAH180_rise,
        hb_sp2_outer_width, 
        hb_sp3_softmax_fade,    
        threshold_distance,
        eps,
        normalize)
    return torch.autograd.grad(Ehb, (xyz, alpha))

@torch.enable_grad()
def calc_chiral_grads(xyz, chirals):
    xyz.requires_grad_(True)
    l = calc_chiral_loss(xyz, chirals)
    if l.item() == 0.0:
        return (torch.zeros(xyz.shape, device=xyz.device),) # autograd returns a tuple..
    return torch.autograd.grad(l, xyz)

def calc_pseudo_dih(pred, true, eps=1e-6):
    '''
    calculate pseudo CA dihedral angle and put loss on them
    Input:
    - predicted & true CA coordinates (I,B,L,3) / (B, L, 3)
    Output:
    - dihedral angle loss
    '''
    I, B, L = pred.shape[:3]
    pred = pred.reshape(I*B, L, -1)
    true_dih = torsion(true[:,:-3,:],true[:,1:-2,:],true[:,2:-1,:],true[:,3:,:]) # (B, L', 2)
    pred_dih = torsion(pred[:,:-3,:],pred[:,1:-2,:],pred[:,2:-1,:],pred[:,3:,:]) # (I*B, L', 2)
    pred_dih = pred_dih.reshape(I, B, -1, 2)
    dih_loss = torch.square(pred_dih - true_dih).sum(dim=-1).mean()
    dih_loss = torch.sqrt(dih_loss + eps)
    return dih_loss

def calc_lddt(pred_ca, true_ca, mask_crds, mask_2d, same_chain, negative=False, interface=False, eps=1e-6):
    # Input
    # pred_ca: predicted CA coordinates (I, B, L, 3)
    # true_ca: true CA coordinates (B, L, 3)
    # pred_lddt: predicted lddt values (I-1, B, L)

    I, B, L = pred_ca.shape[:3]
    
    pred_dist = torch.cdist(pred_ca, pred_ca) # (I, B, L, L)
    true_dist = torch.cdist(true_ca, true_ca).unsqueeze(0) # (1, B, L, L)

    mask = torch.logical_and(true_dist > 0.0, true_dist < 15.0) # (1, B, L, L)
    # update mask information
    mask *= mask_2d[None]
    if negative:
        mask *= same_chain.bool()[None]
    elif interface:
        # ignore atoms between the same chain
        mask *= ~same_chain.bool()[None]

    mask_crds = mask_crds * (mask[0].sum(dim=-1) != 0)

    delta = torch.abs(pred_dist-true_dist) # (I, B, L, L)

    true_lddt = torch.zeros((I,B,L), device=pred_ca.device)
    for distbin in [0.5, 1.0, 2.0, 4.0]:
        true_lddt += 0.25*torch.sum((delta<=distbin)*mask, dim=-1) / (torch.sum(mask, dim=-1) + eps)
    
    true_lddt = mask_crds*true_lddt
    true_lddt = true_lddt.sum(dim=(1,2)) / (mask_crds.sum() + eps)
    return true_lddt


#fd allatom lddt
def calc_allatom_lddt(P, Q, idx, atm_mask, eps=1e-6):
    # P - N x L x 27 x 3
    # Q - L x 27 x 3
    N, L = P.shape[:2]

    # distance matrix
    Pij = torch.square(P[:,:,None,:,None,:]-P[:,None,:,None,:,:]) # (N, L, L, 27, 27)
    Pij = torch.sqrt( Pij.sum(dim=-1) + eps)
    Qij = torch.square(Q[None,:,None,:,None,:]-Q[None,None,:,None,:,:]) # (1, L, L, 27, 27)
    Qij = torch.sqrt( Qij.sum(dim=-1) + eps)

    # get valid pairs
    pair_mask = torch.logical_and(Qij>0,Qij<15).float() # only consider atom pairs within 15A
    # ignore missing atoms
    pair_mask *= (atm_mask[:,:,None,:,None] * atm_mask[:,None,:,None,:]).float()

    # ignore atoms within same residue
    pair_mask *= (idx[:,:,None,None,None] != idx[:,None,:,None,None]).float() # (1, L, L, 27, 27)

    delta_PQ = torch.abs(Pij-Qij+eps) # (N, L, L, 14, 14)

    lddt = torch.zeros( (N,L,27), device=P.device ) # (N, L, 27)
    for distbin in (0.5,1.0,2.0,4.0):
        lddt += 0.25 * torch.sum( (delta_PQ<=distbin)*pair_mask, dim=(2,4)
            ) / ( torch.sum( pair_mask, dim=(2,4) ) + 1e-8)

    lddt = (lddt * atm_mask).sum(dim=(1,2)) / (atm_mask.sum() + eps)
    return lddt


def calc_allatom_lddt_loss(P, Q, pred_lddt, idx, atm_mask, mask_2d, same_chain, negative=False, 
    interface=False, bin_scaling=1, N_stripe=1, eps=1e-6):
    # P - N x L x natoms x 3
    # Q - L x natoms x 3
    # pred_lddt - 1 x nbucket x L
    # idx - 1 x L
    # 
    N, L, Natm = P.shape[:3]

    # striped evaluation of L x L x N_atoms x N_atoms distances to save GPU mem
    L_stripe = int(np.ceil(L/N_stripe)) # how many residues in each stripe
    lddt_s = []
    pair_mask_accum = torch.zeros((N,L,Natm), device=P.device)
    for i1 in np.arange(0, L, L_stripe):
        i2 = min(i1+L_stripe, L)

        # distance matrix
        Pij = torch.square(P[:,i1:i2,None,:,None,:]-P[:,None,:,None,:,:]) # (N, L_stripe, L, 27, 27)
        Pij = torch.sqrt( Pij.sum(dim=-1) + eps)
        Qij = torch.square(Q[None,i1:i2,None,:,None,:]-Q[None,None,:,None,:,:]) # (1, L_stripe, L, 27, 27)
        Qij = torch.sqrt( Qij.sum(dim=-1) + eps)

        # get valid pairs
        pair_mask = torch.logical_and(Qij>0,Qij<15).float() # only consider atom pairs within 15A
        # ignore missing atoms
        pair_mask *= (atm_mask[:,i1:i2,None,:,None] * atm_mask[:,None,:,None,:]).float()

        # ignore atoms within same residue
        pair_mask *= (idx[:,i1:i2,None,None,None] != idx[:,None,:,None,None]).float() # (1, L_stripe, L, 27, 27)
        if negative:
            # ignore atoms between different chains
            pair_mask *= same_chain.bool()[:,i1:i2,:,None,None]
        elif interface:
            # ignore atoms between the same chain
            pair_mask *= ~same_chain.bool()[:,i1:i2,:,None,None]

        pair_mask *= mask_2d.bool()[:,i1:i2,:,None,None]

        delta_PQ = torch.abs(Pij-Qij+eps) # (N, L_stripe, L, 14, 14)

        lddt_ = torch.zeros( (N,i2-i1,Natm), device=P.device )
        for distbin in (0.5,1.0,2.0,4.0):
            lddt_ += 0.25 * torch.sum( (delta_PQ<=distbin*bin_scaling)*pair_mask, dim=(2,4)) \
                     / ( torch.sum( pair_mask, dim=(2,4) ) + eps)
        lddt_s.append(lddt_)
        pair_mask_accum += pair_mask.sum(dim=(1,3))

    lddt = torch.cat(lddt_s, dim=1) # (N, L, Natm)

    final_lddt_by_res = torch.clamp(
        (lddt[-1]*atm_mask[0]).sum(-1)
        / (atm_mask.sum(-1) + eps), min=0.0, max=1.0)

    # calculate lddt prediction loss
    nbin = pred_lddt.shape[1]
    bin_step = 1.0 / nbin
    lddt_bins = torch.linspace(bin_step, 1.0, nbin, dtype=pred_lddt.dtype, device=pred_lddt.device)
    true_lddt_label = torch.bucketize(final_lddt_by_res[None,...], lddt_bins).long()
    lddt_loss = torch.nn.CrossEntropyLoss(reduction='none')(
        pred_lddt, true_lddt_label[-1])

    res_mask = atm_mask.any(dim=-1)
    lddt_loss = (lddt_loss * res_mask).sum() / (res_mask.sum() + eps)
   
    # method 1: average per-residue
    #lddt = lddt.sum(dim=-1) / (atm_mask.sum(dim=-1)+1e-8) # L
    #lddt = (res_mask*lddt).sum() / (res_mask.sum() + 1e-8)

    # method 2: average per-atom
    atm_mask = atm_mask * (pair_mask_accum != 0)
    lddt = (lddt * atm_mask).sum(dim=(1,2)) / (atm_mask.sum() + eps)

    return lddt_loss, lddt


def compute_binder_nonbinder_metrics(
    loss_df: pd.DataFrame,
    binder_nonbinder_tasks: List[str] = ["sm_compl", "sm_compl_permuted_neg", "sm_compl_furthest_neg", "sm_compl_docked_neg", "dude_actives", "dude_inactives"]
) -> Dict[str, float]:
    """compute_binder_nonbinder_metrics 
    Computes binary classification metrics for the binder/non-binder prediction task.

    Args:
        loss_df (pd.DataFrame): The dataframe of records from valid_pdb_cycle.
        binder_nonbinder_tasks (List[str], optional): Which tasks to evaluate the prediction on. 
        Defaults to ["sm_compl", "sm_compl_permuted_neg", "sm_compl_furthest_neg", "sm_compl_docked_neg"].

    Returns:
        Dict[str, float]: A dictionary of metrics
    """

    #fd  Not called from training workflow.  Not clear add'l package needed to compute these.
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

    binder_nonbinder_df = loss_df[loss_df["task"].isin(binder_nonbinder_tasks)]
    true_binder_labels = binder_nonbinder_df["is_binder_label"].to_numpy()
    binding_probabilities = binder_nonbinder_df["binding_probability"].to_numpy()
    binding_predictions = (binding_probabilities > 0.5).astype(int)

    binding_auc = roc_auc_score(true_binder_labels, binding_probabilities)
    binding_f1 = f1_score(true_binder_labels, binding_predictions)
    binding_accuracy = accuracy_score(true_binder_labels, binding_predictions)
    binding_metrics_dictionary = {
        "binding_auc": binding_auc,
        "binding_f1": binding_f1,
        "binding_accuracy": binding_accuracy
    }
    return binding_metrics_dictionary
