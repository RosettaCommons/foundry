import torch
import numpy as np
import scipy
import networkx as nx

from util import (
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
from chemical import NFRAMES, NTOTAL

from kinematics import get_dih, get_ang
from scoring import HbHybType

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

def compute_pae_loss(X, X_y, uX, Y, Y_y, uY, logit_pae, pae_bin_step=0.5, eps=1e-4):
    # predicted aligned error: C-alpha (or sm. mol atom) distances in backbone frames
    xij_ca = torch.einsum('rji,rsj->rsi', uX[-1,:,0], X[-1,:,None,1] - X_y[-1,None,:,0,:]) # last bb prediction
    xij_ca_t = torch.einsum('rji,rsj->rsi', uY[0,:,0], Y[0,:,None,1] - Y_y[0,None,:,0,:]) # assumes B=1
    eij_label = torch.sqrt(torch.square(xij_ca - xij_ca_t).sum(dim=-1)+eps).clone().detach()

    nbin = logit_pae.shape[1]
    pae_bins = torch.linspace(pae_bin_step, pae_bin_step*(nbin-1), nbin-1, dtype=logit_pae.dtype, device=logit_pae.device)
    true_pae_label = torch.bucketize(eij_label, pae_bins, right=True).long()
    return torch.nn.CrossEntropyLoss(reduction='mean')(logit_pae, true_pae_label[None]) # assumes B=1

def compute_pde_loss(X, Y, logit_pde, pde_bin_step=0.3):
    # predicted distance error: C-alpha (or sm. mol atom) pairwise distances
    dX = torch.cdist(X[-1,:,1], X[-1,:,1], compute_mode='donot_use_mm_for_euclid_dist')
    dY = torch.cdist(Y[0,:,1], Y[0,:,1], compute_mode='donot_use_mm_for_euclid_dist')
    dist_err = torch.abs(dX-dY).clone().detach()

    nbin = logit_pde.shape[1]
    pde_bins = torch.linspace(pde_bin_step, pde_bin_step*(nbin-1), nbin-1, dtype=logit_pde.dtype, device=logit_pde.device)
    true_pde_label = torch.bucketize(dist_err, pde_bins, right=True).long()
    return torch.nn.CrossEntropyLoss(reduction='mean')(logit_pde, true_pde_label[None]) # assumes B=1


# from Ivan: FAPE generalized over atom sets & frames
def compute_general_FAPE(X, Y, atom_mask, frames, frame_mask, frame_atom_mask=None, 
    logit_pae=None, logit_pde=None, Z=10.0, dclamp=10.0, gamma=0.99, eps=1e-4):

    # X (predicted) N x L x natoms x 3
    # Y (native)    1 x L x natoms x 3
    # atom_mask     1 x L x natoms
    # frames        1 x L x nframes x 3 x 2
    # frame_mask    1 x L x nframes
    # frame_atom_mask     1 x L x natoms

    if frame_atom_mask is None:
        frame_atom_mask = atom_mask

    N, L, natoms, _ = X.shape

    # flatten middle dims so can gather across residues
    X_prime = X.reshape(N, L*natoms, -1, 3).repeat(1,1,NFRAMES,1)
    Y_prime = Y.reshape(1, L*natoms, -1, 3).repeat(1,1,NFRAMES,1)
    
    # reindex frames for flat X
    frames_reindex = torch.zeros(frames.shape[:-1], device=frames.device)
    for i in range(L):
        frames_reindex[:, i, :, :] = (i+frames[..., i, :, :, 0])*natoms + frames[..., i, :, :, 1]
    frames_reindex = frames_reindex.long()

    frame_mask *= torch.all(
        torch.gather(frame_atom_mask.reshape(1, L*natoms),1,frames_reindex.reshape(1,L*NFRAMES*3)).reshape(1,L,-1,3),
        axis=-1)

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

    loss = (1.0/Z) * (torch.clamp(diff, max=dclamp)).mean(dim=(1,2))
    pae_loss = compute_pae_loss(X, X_y, uX, Y, Y_y, uY, logit_pae) if logit_pae is not None \
                   else torch.tensor(0).to(frames.device)
    pde_loss = compute_pde_loss(X, Y, logit_pde) if logit_pde is not None \
                   else torch.tensor(0).to(frames.device)

    return loss, pae_loss, pde_loss

def calc_crd_rmsd(pred, true, atom_mask, rmsd_mask=None):
    '''
    Calculate coordinate RMSD
    Input:
        - pred: predicted coordinates (B, L, natoms, 3)
        - true: true coordinates (B, L, natoms, 3)
        - atom_mask: mask for seen coordinates (B, L, natoms)
    Output: RMSD after superposition
    '''
    def rmsd(V, W, eps=1e-6):
        L = V.shape[1]
        return torch.sqrt(torch.sum((V-W)*(V-W), dim=(1,2)) / L + eps)
    def centroid(X):
        return X.mean(dim=-2, keepdim=True)
    if rmsd_mask == None:
        rmsd_mask = atom_mask.clone()

    B, L, natoms = pred.shape[:3]

    # center to centroid
    pred_allatom = pred[atom_mask][None]
    true_allatom = true[atom_mask][None]

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
    is_prot = ~is_nucleic(seq)[:-1]

    # bond length: C-N
    blen_CN_pred  = length(pred[:,:-1,2], pred[:,1:,0]).reshape(B,L-1) # (B, L-1)
    CN_loss = torch.clamp( torch.abs(blen_CN_pred - ideal_NC) - sig_len, min=0.0 )
    CN_loss = (bonded*is_prot*CN_loss).sum() / ((bonded*is_prot).sum() + eps)
    blen_loss = CN_loss   #fd squared loss

    # bond angle: CA-C-N, C-N-CA
    bang_CACN_pred = cosangle(pred[:,:-1,2], pred[:,1:,0], pred[:,1:,1]).reshape(B,L-1)
    bang_CNCA_pred = cosangle(pred[:,:-1,2], pred[:,1:,0], pred[:,1:,1]).reshape(B,L-1)
    CACN_loss = torch.clamp( torch.abs(bang_CACN_pred - ideal_CACN) - sig_ang,  min=0.0 )
    CACN_loss = (bonded*is_prot*CACN_loss).sum() / ((bonded*is_prot).sum() + eps)
    CNCA_loss = torch.clamp( torch.abs(bang_CNCA_pred - ideal_CNCA) - sig_ang,  min=0.0 )
    CNCA_loss = (bonded*is_prot*CNCA_loss).sum() / ((bonded*is_prot).sum() + eps)
    bang_loss = CACN_loss + CNCA_loss

    return blen_loss+bang_loss

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

# Rosetta-like version of LJ (fa_atr+fa_rep)
#   lj_lin is switch from linear to 12-6.  Smaller values more sharply penalize clashes
def calc_lj(
    seq, xs, aamask, bond_feats, ljparams, ljcorr, num_bonds,  
    lj_lin=0.85, lj_hb_dis=3.0, lj_OHdon_dis=2.6, lj_hbond_hdis=1.75, 
    lj_maxrad=-1.0, eps=1e-8
):
    def ljV(dist, sigma, epsilon, lj_lin, lj_maxrad):
        N = dist.shape[0]
        linpart = dist<lj_lin*sigma[None]
        deff = dist.clone()
        deff[linpart] = lj_lin*sigma.repeat(N,1)[linpart]
        sd = sigma[None] / deff
        sd2 = sd*sd
        sd6 = sd2 * sd2 * sd2
        sd12 = sd6 * sd6
        ljE = epsilon * (sd12 - 2 * sd6)
        ljE[linpart] += epsilon.repeat(N,1)[linpart] * (
            -12 * sd12[linpart]/deff[linpart] + 12 * sd6[linpart]/deff[linpart]
        ) * (dist[linpart]-deff[linpart])
        if (lj_maxrad>0):
            sdmax = sigma / lj_maxrad
            sd2 = sd*sd
            sd6 = sd2 * sd2 * sd2
            sd12 = sd6 * sd6
            ljE = ljE - epsilon * (sd12 - 2 * sd6)
        return ljE

    N, L = xs.shape[:2]

    # mask keeps running total of what to compute
    mask = aamask[seq][...,None,None]*aamask[seq][None,None,...]
    idxes1r = torch.tril_indices(L,L,-1)
    mask[idxes1r[0],:,idxes1r[1],:] = False
    idxes2r = torch.arange(L)
    idxes2a = torch.tril_indices(NTOTAL,NTOTAL,0)
    mask[idxes2r[:,None],idxes2a[0:1],idxes2r[:,None],idxes2a[1:2]] = False

    # "countpair" can be enforced by making this a weight
    mask[idxes2r,:,idxes2r,:] *= num_bonds[seq,:,:] >= 4 #intra-res
    mask[idxes2r[:-1],:,idxes2r[1:],:] *= (
        num_bonds[seq[:-1],:,2:3] + num_bonds[seq[1:],0:1,:] + 1 >= 4 #inter-res
    )
    atom_bonds = (bond_feats > 0)*(bond_feats<5)
    dist_matrix = scipy.sparse.csgraph.shortest_path(atom_bonds[0].long().detach().cpu().numpy(), directed=False)
    dist_matrix = torch.tensor(np.nan_to_num(dist_matrix, posinf=4.0), device=mask.device) # protein portion is inf and you don't want to mask it out
    mask[:,1,:,1] *= dist_matrix >=4
    si,ai,sj,aj = mask.nonzero(as_tuple=True)

    ds = torch.sqrt( torch.sum ( torch.square( xs[:,si,ai]-xs[:,sj,aj] ), dim=-1 ) + eps )
    
    # hbond correction
    use_hb_dis = (
        ljcorr[seq[si],ai,0]*ljcorr[seq[sj],aj,1] 
        + ljcorr[seq[si],ai,1]*ljcorr[seq[sj],aj,0] )
    use_ohdon_dis = ( # OH are both donors & acceptors
        ljcorr[seq[si],ai,0]*ljcorr[seq[si],ai,1]*ljcorr[seq[sj],aj,0] 
        +ljcorr[seq[si],ai,0]*ljcorr[seq[sj],aj,0]*ljcorr[seq[sj],aj,1] 
    )
    use_hb_hdis = (
        ljcorr[seq[si],ai,2]*ljcorr[seq[sj],aj,1] 
        +ljcorr[seq[si],ai,1]*ljcorr[seq[sj],aj,2] 
    )

    # disulfide correction
    potential_disulf = ljcorr[seq[si],ai,3]*ljcorr[seq[sj],aj,3] 

    ljrs = ljparams[seq[si],ai,0] + ljparams[seq[sj],aj,0]
    ljrs[use_hb_dis] = lj_hb_dis
    ljrs[use_ohdon_dis] = lj_OHdon_dis
    ljrs[use_hb_hdis] = lj_hbond_hdis

    ljss = torch.sqrt( ljparams[seq[si],ai,1] * ljparams[seq[sj],aj,1] + eps )
    ljss [potential_disulf] = 0.0

    ljval = ljV(ds,ljrs,ljss,lj_lin,lj_maxrad)
    return (torch.sum( ljval, dim=-1 )/torch.sum(aamask[seq]))


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
def calc_BB_bond_geom_grads(seq, pred, idx, eps=1e-6, ideal_NC=1.329, ideal_CACN=-0.4415, ideal_CNCA=-0.5255, sig_len=0.02, sig_ang=0.05):
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
    aamask, bond_feats, ljparams, ljcorr, num_bonds, 
    lj_lin=0.85, lj_hb_dis=3.0, lj_OHdon_dis=2.6, lj_hbond_hdis=1.75, 
    lj_maxrad=-1.0, eps=1e-8
):
    xyzaa.requires_grad_(True)
    Elj = calc_lj(
        seq[0], 
        xyzaa[...,:3], 
        aamask,
        bond_feats,
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
    seq, xyz, alpha, toaa, bond_feats, 
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
    return torch.autograd.grad(Ehb, xs)

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


def calc_allatom_lddt_loss(P, Q, pred_lddt, idx, atm_mask, mask_2d, same_chain, negative=False, interface=False, bin_scaling=1, eps=1e-6):
    # P - N x L x 27 x 3
    # Q - L x 27 x 3
    # pred_lddt - 1 x nbucket x L
    N, L, Natm = P.shape[:3]

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
    if negative:
        # ignore atoms between different chains
        pair_mask *= same_chain.bool()[:,:,:,None,None]
    elif interface:
            # ignore atoms between the same chain
            pair_mask *= ~same_chain.bool()[:,:,:,None,None]
    delta_PQ = torch.abs(Pij-Qij+eps) # (N, L, L, 14, 14)

    lddt = torch.zeros( (N,L,Natm), device=P.device ) # (N, L, 27)
    for distbin in (0.5,1.0,2.0,4.0):
        lddt += 0.25 * torch.sum( (delta_PQ<=distbin*bin_scaling)*pair_mask, dim=(2,4)
            ) / ( torch.sum( pair_mask, dim=(2,4) ) + eps)

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
    atm_mask = atm_mask * (pair_mask.sum(dim=(1,3)) != 0)
    lddt = (lddt * atm_mask).sum(dim=(1,2)) / (atm_mask.sum() + eps)

    return lddt_loss, lddt
