import sys
import warnings

import numpy as np
import torch
import warnings

import scipy.sparse
import networkx as nx
from itertools import combinations
from openbabel import openbabel
from scipy.spatial.transform import Rotation

from chemical import *
from kinematics import get_atomize_protein_chirals
from scoring import *


def random_rot_trans(xyz, random_noise=20.0):
    # xyz: (N, L, 27, 3)
    N, L = xyz.shape[:2]

    # pick random rotation axis
    R_mat = torch.tensor(Rotation.random(N).as_matrix(), dtype=xyz.dtype).to(xyz.device)
    xyz = torch.einsum('nij,nlaj->nlai', R_mat, xyz) + torch.rand(N,1,1,3, device=xyz.device)*random_noise
    return xyz

def get_prot_sm_mask(atom_mask, seq):
    """
    Parameters
    ----------
    atom_mask : (..., L, Natoms) 
    seq : (L) 

    Returns
    -------
    mask : (..., L) 
    """
    sm_mask = is_atom(seq).to(atom_mask.device) # (L)
    mask_prot = atom_mask[...,:3].all(dim=-1) & ~sm_mask # valid protein/NA residues (L)
    mask_ca_sm = atom_mask[...,1] & sm_mask # valid sm mol positions (L)
    mask = mask_prot | mask_ca_sm # valid positions
    return mask

def center_and_realign_missing(xyz, mask_t, seq=None, same_chain=None):
    """
    Moves center of mass of xyz to origin, then moves positions with missing
    coordinates to nearest existing residue on same chain.

    Parameters
    ----------
    seq : (L)
    xyz : (L, Natms, 3)
    mask_t : (L, Natms)
    same_chain : (L, L)

    Returns
    -------
    xyz : (L, Natms, 3)
    
    """
    L = xyz.shape[0]

    if same_chain is None:
        same_chain = torch.full((L,L), True)

    # valid protein/NA/small mol. positions
    if seq is None:
        mask = torch.full((L,), True)
    else:
        mask = get_prot_sm_mask(mask_t, seq)

    # center c.o.m of existing residues at the origin
    center_CA = (mask[...,None]*xyz[:,1]).sum(dim=0) / (mask[...,None].sum(dim=0) + 1e-5) # (3)
    xyz = torch.where(mask.view(L,1,1), xyz - center_CA.view(1, 1, 3), xyz)

    # move missing residues to the closest valid residues on same chain
    exist_in_xyz = torch.where(mask)[0] # (L_sub)
    same_chain_in_xyz = same_chain[:,mask].bool() # (L, L_sub)
    seqmap = (torch.arange(L, device=xyz.device)[:,None] - exist_in_xyz[None,:]).abs() # (L, L_sub)
    seqmap[~same_chain_in_xyz] += 99999
    seqmap = torch.argmin(seqmap, dim=-1) # (L)
    idx = torch.gather(exist_in_xyz, 0, seqmap) # (L)
    offset_CA = torch.gather(xyz[:,1], 0, idx.reshape(L,1).expand(-1,3))
    has_neighbor = same_chain_in_xyz.all(-1) 
    offset_CA[~has_neighbor] = 0 # stay at origin if nothing on same chain has coords
    xyz = torch.where(mask.view(L, 1, 1), xyz, xyz + offset_CA.reshape(L,1,3))

    return xyz

def th_ang_v(ab,bc,eps:float=1e-8):
    def th_norm(x,eps:float=1e-8):
        return x.square().sum(-1,keepdim=True).add(eps).sqrt()
    def th_N(x,alpha:float=0):
        return x/th_norm(x).add(alpha)
    ab, bc = th_N(ab),th_N(bc)
    cos_angle = torch.clamp( (ab*bc).sum(-1), -1, 1)
    sin_angle = torch.sqrt(1-cos_angle.square() + eps)
    dih = torch.stack((cos_angle,sin_angle),-1)
    return dih

def th_dih_v(ab,bc,cd):
    def th_cross(a,b):
        a,b = torch.broadcast_tensors(a,b)
        return torch.cross(a,b, dim=-1)
    def th_norm(x,eps:float=1e-8):
        return x.square().sum(-1,keepdim=True).add(eps).sqrt()
    def th_N(x,alpha:float=0):
        return x/th_norm(x).add(alpha)

    ab, bc, cd = th_N(ab),th_N(bc),th_N(cd)
    n1 = th_N( th_cross(ab,bc) )
    n2 = th_N( th_cross(bc,cd) )
    sin_angle = (th_cross(n1,bc)*n2).sum(-1)
    cos_angle = (n1*n2).sum(-1)
    dih = torch.stack((cos_angle,sin_angle),-1)
    return dih

def th_dih(a,b,c,d):
    return th_dih_v(a-b,b-c,c-d)

# build a frame from 3 points
#fd  -  more complicated version splits angle deviations between CA-N and CA-C (giving more accurate CB position)
#fd  -  makes no assumptions about input dims (other than last 1 is xyz)
def rigid_from_3_points(N, Ca, C, is_na=None, eps=1e-4):
    dims = N.shape[:-1]

    v1 = C-Ca
    v2 = N-Ca
    e1 = v1/(torch.norm(v1, dim=-1, keepdim=True)+eps)
    u2 = v2-(torch.einsum('...li, ...li -> ...l', e1, v2)[...,None]*e1)
    e2 = u2/(torch.norm(u2, dim=-1, keepdim=True)+eps)
    e3 = torch.cross(e1, e2, dim=-1)
    R = torch.cat([e1[...,None], e2[...,None], e3[...,None]], axis=-1) #[B,L,3,3] - rotation matrix

    v2 = v2/(torch.norm(v2, dim=-1, keepdim=True)+eps)
    cosref = torch.sum(e1*v2, dim=-1)

    costgt = torch.full(dims, -0.3616, device=N.device)
    if is_na is not None:
       costgt[is_na] = -0.4929
    
    cos2del = torch.clamp( cosref*costgt + torch.sqrt((1-cosref*cosref)*(1-costgt*costgt)+eps), min=-1.0, max=1.0 )

    cosdel = torch.sqrt(0.5*(1+cos2del)+eps)

    sindel = torch.sign(costgt-cosref) * torch.sqrt(1-0.5*(1+cos2del)+eps)

    Rp = torch.eye(3, device=N.device).repeat(*dims,1,1)
    Rp[...,0,0] = cosdel
    Rp[...,0,1] = -sindel
    Rp[...,1,0] = sindel
    Rp[...,1,1] = cosdel
    R = torch.einsum('...ij,...jk->...ik', R,Rp)

    return R, Ca

# note: needs consistency with chemical.py
def is_nucleic(seq):
    return (seq>=NPROTAAS) * (seq <= NNAPROTAAS)

def is_atom(seq):
    return seq > NNAPROTAAS
    
def idealize_reference_frame(seq, xyz_in):
    xyz = xyz_in.clone()

    namask = is_nucleic(seq)
    Rs, Ts = rigid_from_3_points(xyz[...,0,:],xyz[...,1,:],xyz[...,2,:], namask)

    protmask = ~namask

    Nideal = torch.tensor([-0.5272, 1.3593, 0.000], device=xyz_in.device)
    Cideal = torch.tensor([1.5233, 0.000, 0.000], device=xyz_in.device)

    OP1ideal = torch.tensor([-0.7319, 1.2920, 0.000], device=xyz_in.device)
    OP2ideal = torch.tensor([1.4855, 0.000, 0.000], device=xyz_in.device)

    pmask_bs,pmask_rs = protmask.nonzero(as_tuple=True)
    nmask_bs,nmask_rs = namask.nonzero(as_tuple=True)
    xyz[pmask_bs,pmask_rs,0,:] = torch.einsum('...ij,j->...i', Rs[pmask_bs,pmask_rs], Nideal) + Ts[pmask_bs,pmask_rs]
    xyz[pmask_bs,pmask_rs,2,:] = torch.einsum('...ij,j->...i', Rs[pmask_bs,pmask_rs], Cideal) + Ts[pmask_bs,pmask_rs]
    xyz[nmask_bs,nmask_rs,0,:] = torch.einsum('...ij,j->...i', Rs[nmask_bs,nmask_rs], OP1ideal) + Ts[nmask_bs,nmask_rs]
    xyz[nmask_bs,nmask_rs,2,:] = torch.einsum('...ij,j->...i', Rs[nmask_bs,nmask_rs], OP2ideal) + Ts[nmask_bs,nmask_rs]

    return xyz

# works for both dna and protein
# alphas in order:
#    omega/phi/psi: 0-2
#    chi_1-4(prot): 3-6
#    cb/cg bend: 7-9
#    eps(p)/zeta(p): 10-11
#    alpha/beta/gamma/delta: 12-15
#    nu2/nu1/nu0: 16-18
#    chi_1(na): 19
def get_tor_mask(seq, torsion_indices, mask_in=None):
    B,L = seq.shape[:2]
    dna_mask = is_nucleic(seq)

    tors_mask = torsion_indices[seq,:,-1] > 0

    if mask_in != None:
        N = mask_in.shape[2]
        ts = torsion_indices[seq]
        bs = torch.arange(B, device=seq.device)[:,None,None,None]
        rs = torch.arange(L, device=seq.device)[None,:,None,None] - (ts<0)*1 # ts<-1 ==> prev res
        ts = torch.abs(ts)
        tors_mask *= mask_in[bs,rs,ts].all(dim=-1)

    return tors_mask


def get_torsions(xyz_in, seq, torsion_indices, torsion_can_flip, ref_angles, mask_in=None):
    B,L = xyz_in.shape[:2]

    tors_mask = get_tor_mask(seq, torsion_indices, mask_in)
    # idealize given xyz coordinates before computing torsion angles
    xyz = idealize_reference_frame(seq, xyz_in)

    ts = torsion_indices[seq]
    bs = torch.arange(B, device=xyz_in.device)[:,None,None,None]
    xs = torch.arange(L, device=xyz_in.device)[None,:,None,None] - (ts<0)*1 # ts<-1 ==> prev res
    ys = torch.abs(ts)
    xyzs_bytor = xyz[bs,xs,ys,:]

    torsions = torch.zeros( (B,L,NTOTALDOFS,2), device=xyz_in.device )
    torsions[...,:7,:] = th_dih(
        xyzs_bytor[...,:7,0,:],xyzs_bytor[...,:7,1,:],xyzs_bytor[...,:7,2,:],xyzs_bytor[...,:7,3,:]
    )
    torsions[:,:,2,:] = -1 * torsions[:,:,2,:] # shift psi by pi
    torsions[...,10:,:] = th_dih(
        xyzs_bytor[...,10:,0,:],xyzs_bytor[...,10:,1,:],xyzs_bytor[...,10:,2,:],xyzs_bytor[...,10:,3,:]
    )

    # angles (hardcoded)
    # CB bend
    NC = 0.5*( xyz[:,:,0,:3] + xyz[:,:,2,:3] )
    CA = xyz[:,:,1,:3]
    CB = xyz[:,:,4,:3]
    t = th_ang_v(CB-CA,NC-CA)
    t0 = ref_angles[seq][...,0,:]
    torsions[:,:,7,:] = torch.stack( 
        (torch.sum(t*t0,dim=-1),t[...,0]*t0[...,1]-t[...,1]*t0[...,0]),
        dim=-1 )
    
    # CB twist
    NCCA = NC-CA
    NCp = xyz[:,:,2,:3] - xyz[:,:,0,:3]
    NCpp = NCp - torch.sum(NCp*NCCA, dim=-1, keepdim=True)/ torch.sum(NCCA*NCCA, dim=-1, keepdim=True) * NCCA
    t = th_ang_v(CB-CA,NCpp)
    t0 = ref_angles[seq][...,1,:]
    torsions[:,:,8,:] = torch.stack( 
        (torch.sum(t*t0,dim=-1),t[...,0]*t0[...,1]-t[...,1]*t0[...,0]),
        dim=-1 )

    # CG bend
    CG = xyz[:,:,5,:3]
    t = th_ang_v(CG-CB,CA-CB)
    t0 = ref_angles[seq][...,2,:]
    torsions[:,:,9,:] = torch.stack( 
        (torch.sum(t*t0,dim=-1),t[...,0]*t0[...,1]-t[...,1]*t0[...,0]),
        dim=-1 )
    
    mask0 = (torch.isnan(torsions[...,0])).nonzero()
    mask1 = (torch.isnan(torsions[...,1])).nonzero()
    torsions[mask0[:,0],mask0[:,1],mask0[:,2],0] = 1.0
    torsions[mask1[:,0],mask1[:,1],mask1[:,2],1] = 0.0

    # alt chis
    torsions_alt = torsions.clone()
    torsions_alt[torsion_can_flip[seq,:]] *= -1

    # torsions to restrain to 0 or 180 degree
    # (this should be specified in chemical?)
    tors_planar = torch.zeros((B, L, NTOTALDOFS), dtype=torch.bool, device=xyz_in.device)
    tors_planar[:,:,5] = seq == aa2num['TYR'] # TYR chi 3 should be planar

    return torsions, torsions_alt, tors_mask, tors_planar

def xyz_to_frame_xyz(xyz, seq_unmasked, atom_frames):
    """
    xyz (1, L, natoms, 3)
    seq_unmasked (1, L)
    atom_frames (1, L, 3, 2)
    """ 
    xyz_frame = xyz.clone()
    atoms = is_atom(seq_unmasked)
    if torch.all(~atoms):
        return xyz_frame

    atom_crds = xyz_frame[atoms]
    atom_L, natoms, _ = atom_crds.shape
    frames_reindex = torch.zeros(atom_frames.shape[:-1])
    
    for i in range(atom_L):
        frames_reindex[:, i, :] = (i+atom_frames[..., i, :, 0])*natoms + atom_frames[..., i, :, 1]
    frames_reindex = frames_reindex.long()
    xyz_frame[atoms, :, :3] = atom_crds.reshape(atom_L*natoms, 3)[frames_reindex]
    return xyz_frame

def xyz_frame_from_rotation_mask(xyz,rotation_mask, atom_frames):
    """
    function to get xyz_frame for l1 feature in Structure module
    xyz (1, L, natoms, 3)
    rotation_mask (1, L)
    atom_frames (1, L, 3, 2)
    """
    xyz_frame = xyz.clone()
    if torch.all(~rotation_mask):
        return xyz_frame

    atom_crds = xyz_frame[rotation_mask]
    atom_L, natoms, _ = atom_crds.shape
    frames_reindex = torch.zeros(atom_frames.shape[:-1])
    
    for i in range(atom_L):
        frames_reindex[:, i, :] = (i+atom_frames[..., i, :, 0])*natoms + atom_frames[..., i, :, 1]
    frames_reindex = frames_reindex.long()
    xyz_frame[rotation_mask, :, :3] = atom_crds.reshape(atom_L*natoms, 3)[frames_reindex]
    return xyz_frame

def xyz_t_to_frame_xyz(xyz_t, seq_unmasked, atom_frames):
    """
    xyz_t (1, T, L, natoms, 3)
    seq_unmasked (B, L)
    atom_frames (1, L, 3, 2)
    """
    xyz_t_frame = xyz_t.clone()
    atoms = is_atom(seq_unmasked[0])
    if torch.all(~atoms):
        return xyz_t_frame
    atom_crds_t = xyz_t_frame[:, :, atoms]

    B, T, atom_L, natoms, _ = atom_crds_t.shape
    frames_reindex = torch.zeros(atom_frames.shape[:-1])
    for i in range(atom_L):
        frames_reindex[:, i, :] = (i+atom_frames[..., i, :, 0])*natoms + atom_frames[..., i, :, 1]
    frames_reindex = frames_reindex.long()
    xyz_t_frame[:, :, atoms, :3] = atom_crds_t.reshape(T, atom_L*natoms, 3)[:, frames_reindex.squeeze(0)]
    return xyz_t_frame

def get_frames(xyz_in, xyz_mask, seq, frame_indices, atom_frames=None):
    B,L,natoms = xyz_in.shape[:3]
    frames = frame_indices[seq]
    atoms = seq > NNAPROTAAS
    if torch.any(atoms):
        frames[:,atoms[0].nonzero().flatten(), 0] = atom_frames

    frame_mask = ~torch.all(frames[...,0, :] == frames[...,1, :], axis=-1)

    # frame_mask *= torch.all(
    #     torch.gather(xyz_mask,2,frames.reshape(B,L,-1)).reshape(B,L,-1,3),
    #     axis=-1)

    return frames, frame_mask

def get_tips(xyz, seq):
    B,L = xyz.shape[:2]

    xyz_tips = torch.gather(xyz, 2, tip_indices.to(xyz.device)[seq][:,:,None,None].expand(-1,-1,-1,3)).reshape(B, L, 3)
    if torch.isnan(xyz_tips).any(): # replace NaN tip atom with virtual Cb atom
        # three anchor atoms
        N  = xyz[:,:,0]
        Ca = xyz[:,:,1]
        C  = xyz[:,:,2]

        # recreate Cb given N,Ca,C
        b = Ca - N
        c = C - Ca
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca    

        xyz_tips = torch.where(torch.isnan(xyz_tips), Cb, xyz_tips)
    return xyz_tips

def superimpose(pred, true, atom_mask):
    
    def centroid(X):
        return X.mean(dim=-2, keepdim=True)
    
    B, L, natoms = pred.shape[:3]

    # center to centroid
    pred_allatom = pred[atom_mask][None]
    true_allatom = true[atom_mask][None]

    cp = centroid(pred_allatom)
    ct = centroid(true_allatom)
    
    pred_allatom_origin = pred_allatom - cp
    true_allatom_origin = true_allatom - ct

    # Computation of the covariance matrix
    C = torch.matmul(pred_allatom_origin.permute(0,2,1), true_allatom_origin)

    # Compute optimal rotation matrix using SVD
    V, S, W = torch.svd(C)

    # get sign to ensure right-handedness
    d = torch.ones([B,3,3], device=pred.device)
    d[:,:,-1] = torch.sign(torch.det(V)*torch.det(W)).unsqueeze(1)

    # Rotation matrix U
    U = torch.matmul(d*V, W.permute(0,2,1)) # (IB, 3, 3)
    pred_rms = pred - cp
    true_rms = true - ct
    
    # Rotate pred
    rP = torch.matmul(pred_rms, U) # (IB, L*3, 3)
    
    return rP+ct

def writepdb(filename, atoms, seq, modelnum=None, chain="A", idx_pdb=None, bfacts=None, 
             bond_feats=None, file_mode="w"):

    f = open(filename, file_mode)
    ctr = 1
    scpu = seq.cpu().squeeze(0)
    atomscpu = atoms.cpu().squeeze(0)

    if bfacts is None:
        bfacts = torch.zeros(atomscpu.shape[0])
    if idx_pdb is None:
        idx_pdb = 1 + torch.arange(atomscpu.shape[0])

    Bfacts = torch.clamp( bfacts.cpu(), 0, 1)
    atom_idxs = {}
    if modelnum is not None:
        f.write(f"MODEL        {modelnum}\n")
    for i,s in enumerate(scpu):
        natoms = atomscpu.shape[-2]
        if (natoms!=NHEAVY and natoms!=NTOTAL and natoms!=3):
            print ('bad size!', natoms, NHEAVY, NTOTAL, atoms.shape)
            assert(False)

        if s >= len(aa2long):
            lig_name = "LG1"
            atom_idxs[i] = ctr
            f.write ("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
                    "HETATM", ctr, num2aa[s], lig_name,
                    chain, torch.max(idx_pdb)+10, atomscpu[i,1,0], atomscpu[i,1,1], atomscpu[i,1,2],
                    1.0, Bfacts[i] ) )
            ctr += 1
            continue

        atms = aa2long[s]
        # his prot hack
        #if (s==8 and torch.linalg.norm( atomscpu[i,9,:]-atomscpu[i,5,:] ) < 1.7):
        #    atms = (
        #        " N  "," CA "," C  "," O  "," CB "," CG "," NE2"," CD2"," CE1"," ND1",
        #          None,  None,  None,  None," H  "," HA ","1HB ","2HB "," HD2"," HE1",
        #        " HD1",  None,  None,  None,  None,  None,  None) # his_d

        for j,atm_j in enumerate(atms):
            if (j<natoms and atm_j is not None and not torch.isnan(atomscpu[i,j,:]).any()):
                f.write ("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
                    "ATOM", ctr, atm_j, num2aa[s],
                    chain, idx_pdb[i], atomscpu[i,j,0], atomscpu[i,j,1], atomscpu[i,j,2],
                    1.0, Bfacts[i] ) )
                ctr += 1
    if bond_feats != None:
        atom_bonds = (bond_feats > 0) * (bond_feats <5)
        atom_bonds = atom_bonds.cpu()
        b, i, j = atom_bonds.nonzero(as_tuple=True)
        for start, end in zip(i,j):
            f.write(f"CONECT{atom_idxs[int(start.cpu().numpy())]:5d}{atom_idxs[int(end.cpu().numpy())]:5d}\n")
    if modelnum is not None:
        f.write("ENDMDL\n")
    
# process ideal frames
def make_frame(X, Y):
    Xn = X / torch.linalg.norm(X)
    Y = Y - torch.dot(Y, Xn) * Xn
    Yn = Y / torch.linalg.norm(Y)
    Z = torch.cross(Xn,Yn)
    Zn =  Z / torch.linalg.norm(Z)
    return torch.stack((Xn,Yn,Zn), dim=-1)


# resolve tip atom indices
tip_indices = torch.full((NAATOKENS,), 0)
for i in range(NAATOKENS):
    if i > NNAPROTAAS-1:
        # all atoms are at index 1 in the atom array 
        tip_indices[i] = 1
    else:
        tip_atm = aa2tip[i]
        atm_long = aa2long[i]
        tip_indices[i] = atm_long.index(tip_atm)


# resolve torsion indices
#  a negative index indicates the previous residue
# order:
#    omega/phi/psi: 0-2
#    chi_1-4(prot): 3-6
#    cb/cg bend: 7-9
#    eps(p)/zeta(p): 10-11
#    alpha/beta/gamma/delta: 12-15
#    nu2/nu1/nu0: 16-18
#    chi_1(na): 19
torsion_indices = torch.full((NAATOKENS,NTOTALDOFS,4),0)
torsion_can_flip = torch.full((NAATOKENS,NTOTALDOFS),False,dtype=torch.bool)
for i in range(NPROTAAS):
    i_l, i_a = aa2long[i], aa2longalt[i]

    # protein omega/phi/psi
    torsion_indices[i,0,:] = torch.tensor([-1,-2,0,1]) # omega
    torsion_indices[i,1,:] = torch.tensor([-2,0,1,2]) # phi
    torsion_indices[i,2,:] = torch.tensor([0,1,2,3]) # psi (+pi)

    # protein chis
    for j in range(4):
        if torsions[i][j] is None:
            continue
        for k in range(4):
            a = torsions[i][j][k]
            torsion_indices[i,3+j,k] = i_l.index(a)
            if (i_l.index(a) != i_a.index(a)):
                torsion_can_flip[i,3+j] = True ##bb tors never flip

    # CB/CG angles (only masking uses these indices)
    torsion_indices[i,7,:] = torch.tensor([0,2,1,4]) # CB ang1
    torsion_indices[i,8,:] = torch.tensor([0,2,1,4]) # CB ang2
    torsion_indices[i,9,:] = torch.tensor([0,2,4,5]) # CG ang (arg 1 ignored)

# HIS is a special case for flip
torsion_can_flip[8,4]=False

for i in range(NPROTAAS,NNAPROTAAS):
    # NA BB tors
    torsion_indices[i,10,:] = torch.tensor([-5,-7,-8,1])  # epsilon_prev
    torsion_indices[i,11,:] = torch.tensor([-7,-8,1,3])   # zeta_prev
    torsion_indices[i,12,:] = torch.tensor([0,1,3,4])     # alpha (+2pi/3)
    torsion_indices[i,13,:] = torch.tensor([1,3,4,5])     # beta
    torsion_indices[i,14,:] = torch.tensor([3,4,5,7])     # gamma
    torsion_indices[i,15,:] = torch.tensor([4,5,7,8])     # delta

    if (i<NPROTAAS+5):
        # is DNA
        torsion_indices[i,16,:] = torch.tensor([4,5,6,10])     # nu2
        torsion_indices[i,17,:] = torch.tensor([5,6,10,9])     # nu1
        torsion_indices[i,18,:] = torch.tensor([6,10,9,7])     # nu0
    else:   
        # is RNA (fd: my fault since I flipped C1'/C2' order for DNA and RNA)
        torsion_indices[i,16,:] = torch.tensor([4,5,6,9])     # nu2
        torsion_indices[i,17,:] = torch.tensor([5,6,9,10])     # nu1
        torsion_indices[i,18,:] = torch.tensor([6,9,10,7])     # nu0

    # NA chi
    if torsions[i][0] is not None:
        i_l = aa2long[i]
        for k in range(4):
            a = torsions[i][0][k]
            torsion_indices[i,19,k] = i_l.index(a) # chi
        # no NA torsion flips

# build the mapping from atoms in the full rep (Nx27) to the "alternate" rep
allatom_mask = torch.zeros((NAATOKENS,NTOTAL), dtype=torch.bool)
long2alt = torch.zeros((NAATOKENS,NTOTAL), dtype=torch.long)
for i in range(NNAPROTAAS):
    i_l, i_lalt = aa2long[i],  aa2longalt[i]
    for j,a in enumerate(i_l):
        if (a is None):
            long2alt[i,j] = j
        else:
            long2alt[i,j] = i_lalt.index(a)
            allatom_mask[i,j] = True
for i in range(NNAPROTAAS, NAATOKENS):
    for j in range(NTOTAL):
        long2alt[i, j] = j
allatom_mask[NNAPROTAAS:,1] = True

# bond graph traversal
num_bonds = torch.zeros((NAATOKENS,NTOTAL,NTOTAL), dtype=torch.long)
for i in range(NNAPROTAAS):
    num_bonds_i = np.zeros((NTOTAL,NTOTAL))
    for (bnamei,bnamej) in aabonds[i]:
        bi,bj = aa2long[i].index(bnamei),aa2long[i].index(bnamej)
        num_bonds_i[bi,bj] = 1
    num_bonds_i = scipy.sparse.csgraph.shortest_path (num_bonds_i,directed=False)
    num_bonds_i[num_bonds_i>=4] = 4
    num_bonds[i,...] = torch.tensor(num_bonds_i)


# atom type indices
idx2aatype = []
for x in aa2type:
    for y in x:
        if y and y not in idx2aatype:
            idx2aatype.append(y)
aatype2idx = {x:i for i,x in enumerate(idx2aatype)}

# element indices
idx2elt = []
for x in aa2elt:
    for y in x:
        if y and y not in idx2elt:
            idx2elt.append(y)
elt2idx = {x:i for i,x in enumerate(idx2elt)}

# LJ/LK scoring parameters
atom_type_index = torch.zeros((NAATOKENS,NTOTAL), dtype=torch.long)
element_index = torch.zeros((NAATOKENS,NTOTAL), dtype=torch.long)

ljlk_parameters = torch.zeros((NAATOKENS,NTOTAL,5), dtype=torch.float)
lj_correction_parameters = torch.zeros((NAATOKENS,NTOTAL,4), dtype=bool) # donor/acceptor/hpol/disulf
for i in range(NNAPROTAAS):
    for j,a in enumerate(aa2type[i]):
        if (a is not None):
            atom_type_index[i,j] = aatype2idx[a]
            ljlk_parameters[i,j,:] = torch.tensor( type2ljlk[a] )
            lj_correction_parameters[i,j,0] = (type2hb[a]==HbAtom.DO)+(type2hb[a]==HbAtom.DA)
            lj_correction_parameters[i,j,1] = (type2hb[a]==HbAtom.AC)+(type2hb[a]==HbAtom.DA)
            lj_correction_parameters[i,j,2] = (type2hb[a]==HbAtom.HP)
            lj_correction_parameters[i,j,3] = (a=="SH1" or a=="HS")
    for j,a in enumerate(aa2elt[i]):
        if (a is not None):
            element_index[i,j] = elt2idx[a]

# hbond scoring parameters
def donorHs(D,bonds,atoms):
    dHs = []
    for (i,j) in bonds:
        if (i==D):
            idx_j = atoms.index(j)
            if (idx_j>=NHEAVY):  # if atom j is a hydrogen
                dHs.append(idx_j)
        if (j==D):
            idx_i = atoms.index(i)
            if (idx_i>=NHEAVY):  # if atom j is a hydrogen
                dHs.append(idx_i)
    assert (len(dHs)>0)
    return dHs

def acceptorBB0(A,hyb,bonds,atoms):
    if (hyb == HbHybType.SP2):
        for (i,j) in bonds:
            if (i==A):
                B = atoms.index(j)
                if (B<NHEAVY):
                    break
            if (j==A):
                B = atoms.index(i)
                if (B<NHEAVY):
                    break
        for (i,j) in bonds:
            if (i==atoms[B]):
                B0 = atoms.index(j)
                if (B0<NHEAVY):
                    break
            if (j==atoms[B]):
                B0 = atoms.index(i)
                if (B0<NHEAVY):
                    break
    elif (hyb == HbHybType.SP3 or hyb == HbHybType.RING):
        for (i,j) in bonds:
            if (i==A):
                B = atoms.index(j)
                if (B<NHEAVY):
                    break
            if (j==A):
                B = atoms.index(i)
                if (B<NHEAVY):
                    break
        for (i,j) in bonds:
            if (i==A and j!=atoms[B]):
                B0 = atoms.index(j)
                break
            if (j==A and i!=atoms[B]):
                B0 = atoms.index(i)
                break

    return B,B0


hbtypes = torch.full((NAATOKENS,NTOTAL,3),-1, dtype=torch.long) # (donortype, acceptortype, acchybtype)
hbbaseatoms = torch.full((NAATOKENS,NTOTAL,2),-1, dtype=torch.long) # (B,B0) for acc; (D,-1) for don
hbpolys = torch.zeros((HbDonType.NTYPES,HbAccType.NTYPES,3,15)) # weight,xmin,xmax,ymin,ymax,c9,...,c0

for i in range(NNAPROTAAS):
    for j,a in enumerate(aa2type[i]):
        if (a in type2dontype):
            j_hs = donorHs(aa2long[i][j],aabonds[i],aa2long[i])
            for j_h in j_hs:
                hbtypes[i,j_h,0] = type2dontype[a]
                hbbaseatoms[i,j_h,0] = j
        if (a in type2acctype):
            j_b, j_b0 = acceptorBB0(aa2long[i][j],type2hybtype[a],aabonds[i],aa2long[i])
            hbtypes[i,j,1] = type2acctype[a]
            hbtypes[i,j,2] = type2hybtype[a]
            hbbaseatoms[i,j,0] = j_b
            hbbaseatoms[i,j,1] = j_b0

for i in range(HbDonType.NTYPES):
    for j in range(HbAccType.NTYPES):
        weight = dontype2wt[i]*acctype2wt[j]

        pdist,pbah,pahd = hbtypepair2poly[(i,j)]
        xrange,yrange,coeffs = hbpolytype2coeffs[pdist]
        hbpolys[i,j,0,0] = weight
        hbpolys[i,j,0,1:3] = torch.tensor(xrange)
        hbpolys[i,j,0,3:5] = torch.tensor(yrange)
        hbpolys[i,j,0,5:] = torch.tensor(coeffs)
        xrange,yrange,coeffs = hbpolytype2coeffs[pahd]
        hbpolys[i,j,1,0] = weight
        hbpolys[i,j,1,1:3] = torch.tensor(xrange)
        hbpolys[i,j,1,3:5] = torch.tensor(yrange)
        hbpolys[i,j,1,5:] = torch.tensor(coeffs)
        xrange,yrange,coeffs = hbpolytype2coeffs[pbah]
        hbpolys[i,j,2,0] = weight
        hbpolys[i,j,2,1:3] = torch.tensor(xrange)
        hbpolys[i,j,2,3:5] = torch.tensor(yrange)
        hbpolys[i,j,2,5:] = torch.tensor(coeffs)

# cartbonded scoring parameters
# (0) inter-res
cb_lengths_CN = (1.32868, 369.445)
cb_angles_CACN = (2.02807,160)
cb_angles_CNCA = (2.12407,96.53)
cb_torsions_CACNH = (0.0,41.830) # also used for proline CACNCD
cb_torsions_CANCO = (0.0,38.668)

# note for the below, the extra amino acid corrsponds to cb params for HIS_D
# (1) intra-res lengths
cb_lengths = [[] for i in range(NAATOKENS+1)]
for cst in cartbonded_data_raw['lengths']:
    res_idx = aa2num[ cst['res'] ]
    cb_lengths[res_idx].append( (
        aa2long[res_idx].index(cst['atm1']),
        aa2long[res_idx].index(cst['atm2']),
        cst['x0'],cst['K']
    ) )
ncst_per_res=max([len(i) for i in cb_lengths])
cb_length_t = torch.zeros(NAATOKENS+1,ncst_per_res,4)
for i in range(NNAPROTAAS+1):
    src = i
    if (num2aa[i]=='UNK' or num2aa[i]=='MAS'):
        src=aa2num['ALA']
    if (len(cb_lengths[src])>0):
        cb_length_t[i,:len(cb_lengths[src]),:] = torch.tensor(cb_lengths[src])

# (2) intra-res angles
cb_angles = [[] for i in range(NAATOKENS+1)]
for cst in cartbonded_data_raw['angles']:
    res_idx = aa2num[ cst['res'] ]
    cb_angles[res_idx].append( (
        aa2long[res_idx].index(cst['atm1']),
        aa2long[res_idx].index(cst['atm2']),
        aa2long[res_idx].index(cst['atm3']),
        cst['x0'],cst['K']
    ) )
ncst_per_res=max([len(i) for i in cb_angles])
cb_angle_t = torch.zeros(NAATOKENS+1,ncst_per_res,5)
for i in range(NNAPROTAAS+1):
    src = i
    if (num2aa[i]=='UNK' or num2aa[i]=='MAS'):
        src=aa2num['ALA']

    if (len(cb_angles[src])>0):
        cb_angle_t[i,:len(cb_angles[src]),:] = torch.tensor(cb_angles[src])

# (3) intra-res torsions
cb_torsions = [[] for i in range(NAATOKENS+1)]
for cst in cartbonded_data_raw['torsions']:
    res_idx = aa2num[ cst['res'] ]
    cb_torsions[res_idx].append( (
        aa2long[res_idx].index(cst['atm1']),
        aa2long[res_idx].index(cst['atm2']),
        aa2long[res_idx].index(cst['atm3']),
        aa2long[res_idx].index(cst['atm4']),
        cst['x0'],cst['K'],cst['period']
    ) )
ncst_per_res=max([len(i) for i in cb_torsions])
cb_torsion_t = torch.zeros(NAATOKENS+1,ncst_per_res,7)
cb_torsion_t[...,6]=1.0 # periodicity
for i in range(NNAPROTAAS):
    src = i
    if (num2aa[i]=='UNK' or num2aa[i]=='MAS'):
        src=aa2num['ALA']

    if (len(cb_torsions[src])>0):
        cb_torsion_t[i,:len(cb_torsions[src]),:] = torch.tensor(cb_torsions[src])

# kinematic parameters
base_indices = torch.full((NAATOKENS,NTOTAL),0, dtype=torch.long) # base frame that builds each atom
xyzs_in_base_frame = torch.ones((NAATOKENS,NTOTAL,4)) # coords of each atom in the base frame
RTs_by_torsion = torch.eye(4).repeat(NAATOKENS,NTOTALTORS,1,1) # torsion frames
reference_angles = torch.ones((NAATOKENS,NPROTANGS,2)) # reference values for bendable angles

## PROTEIN
for i in range(NPROTAAS):
    i_l = aa2long[i]
    for name, base, coords in ideal_coords[i]:
        idx = i_l.index(name)
        base_indices[i,idx] = base
        xyzs_in_base_frame[i,idx,:3] = torch.tensor(coords)

    # omega frame
    RTs_by_torsion[i,0,:3,:3] = torch.eye(3)
    RTs_by_torsion[i,0,:3,3] = torch.zeros(3)

    # phi frame
    RTs_by_torsion[i,1,:3,:3] = make_frame(
        xyzs_in_base_frame[i,0,:3] - xyzs_in_base_frame[i,1,:3],
        torch.tensor([1.,0.,0.])
    )
    RTs_by_torsion[i,1,:3,3] = xyzs_in_base_frame[i,0,:3]

    # psi frame
    RTs_by_torsion[i,2,:3,:3] = make_frame(
        xyzs_in_base_frame[i,2,:3] - xyzs_in_base_frame[i,1,:3],
        xyzs_in_base_frame[i,1,:3] - xyzs_in_base_frame[i,0,:3]
    )
    RTs_by_torsion[i,2,:3,3] = xyzs_in_base_frame[i,2,:3]

    # chi1 frame
    if torsions[i][0] is not None:
        a0,a1,a2 = torsion_indices[i,3,0:3]
        RTs_by_torsion[i,3,:3,:3] = make_frame(
            xyzs_in_base_frame[i,a2,:3]-xyzs_in_base_frame[i,a1,:3],
            xyzs_in_base_frame[i,a0,:3]-xyzs_in_base_frame[i,a1,:3],
        )
        RTs_by_torsion[i,3,:3,3] = xyzs_in_base_frame[i,a2,:3]

    # chi2/3/4 frame
    for j in range(1,4):
        if torsions[i][j] is not None:
            a2 = torsion_indices[i,3+j,2]
            if ((i==18 and j==2) or (i==8 and j==2)):  # TYR CZ-OH & HIS CE1-HE1 a special case
                a0,a1 = torsion_indices[i,3+j,0:2]
                RTs_by_torsion[i,3+j,:3,:3] = make_frame(
                    xyzs_in_base_frame[i,a2,:3]-xyzs_in_base_frame[i,a1,:3],
                    xyzs_in_base_frame[i,a0,:3]-xyzs_in_base_frame[i,a1,:3] )
            else:
                RTs_by_torsion[i,3+j,:3,:3] = make_frame(
                    xyzs_in_base_frame[i,a2,:3],
                    torch.tensor([-1.,0.,0.]), )
            RTs_by_torsion[i,3+j,:3,3] = xyzs_in_base_frame[i,a2,:3]

    # CB/CG angles
    NCr = 0.5*(xyzs_in_base_frame[i,0,:3]+xyzs_in_base_frame[i,2,:3])
    CAr = xyzs_in_base_frame[i,1,:3]
    CBr = xyzs_in_base_frame[i,4,:3]
    CGr = xyzs_in_base_frame[i,5,:3]
    reference_angles[i,0,:]=th_ang_v(CBr-CAr,NCr-CAr)
    NCp = xyzs_in_base_frame[i,2,:3]-xyzs_in_base_frame[i,0,:3]
    NCpp = NCp - torch.dot(NCp,NCr)/ torch.dot(NCr,NCr) * NCr
    reference_angles[i,1,:]=th_ang_v(CBr-CAr,NCpp)
    reference_angles[i,2,:]=th_ang_v(CGr,torch.tensor([-1.,0.,0.]))

## NUCLEIC ACIDS
for i in range(NPROTAAS, NNAPROTAAS):
    i_l = aa2long[i]

    for name, base, coords in ideal_coords[i]:
        idx = i_l.index(name)
        base_indices[i,idx] = base
        xyzs_in_base_frame[i,idx,:3] = torch.tensor(coords)

    # epsilon(p)/zeta(p) - like omega in protein, not used to build atoms
    #                    - keep as identity
    RTs_by_torsion[i,NPROTTORS+0,:3,:3] = torch.eye(3)
    RTs_by_torsion[i,NPROTTORS+0,:3,3] = torch.zeros(3)
    RTs_by_torsion[i,NPROTTORS+1,:3,:3] = torch.eye(3)
    RTs_by_torsion[i,NPROTTORS+1,:3,3] = torch.zeros(3)

    # alpha
    RTs_by_torsion[i,NPROTTORS+2,:3,:3] = make_frame(
        xyzs_in_base_frame[i,3,:3] - xyzs_in_base_frame[i,1,:3], # P->O5'
        xyzs_in_base_frame[i,0,:3] - xyzs_in_base_frame[i,1,:3]  # P<-OP1
    )
    RTs_by_torsion[i,NPROTTORS+2,:3,3] = xyzs_in_base_frame[i,3,:3] # O5'

    # beta
    RTs_by_torsion[i,NPROTTORS+3,:3,:3] = make_frame(
        xyzs_in_base_frame[i,4,:3] , torch.tensor([-1.,0.,0.])
    )
    RTs_by_torsion[i,NPROTTORS+3,:3,3] = xyzs_in_base_frame[i,4,:3] # C5'

    # gamma
    RTs_by_torsion[i,NPROTTORS+4,:3,:3] = make_frame(
        xyzs_in_base_frame[i,5,:3] , torch.tensor([-1.,0.,0.])
    )
    RTs_by_torsion[i,NPROTTORS+4,:3,3] = xyzs_in_base_frame[i,5,:3] # C4'

    # delta
    RTs_by_torsion[i,NPROTTORS+5,:3,:3] = make_frame(
        xyzs_in_base_frame[i,7,:3] , torch.tensor([-1.,0.,0.])
    )
    RTs_by_torsion[i,NPROTTORS+5,:3,3] = xyzs_in_base_frame[i,7,:3] # C3'

    # nu2
    RTs_by_torsion[i,NPROTTORS+6,:3,:3] = make_frame(
        xyzs_in_base_frame[i,6,:3] , torch.tensor([-1.,0.,0.])
    )
    RTs_by_torsion[i,NPROTTORS+6,:3,3] = xyzs_in_base_frame[i,6,:3] # O4'

    # nu1
    if i<NPROTAAS+5:
        # is DNA
        C1idx,C2idx = 10,9
    else:
        # is RNA
        C1idx,C2idx = 9,10

    RTs_by_torsion[i,NPROTTORS+7,:3,:3] = make_frame(
        xyzs_in_base_frame[i,C1idx,:3] , torch.tensor([-1.,0.,0.])
    )
    RTs_by_torsion[i,NPROTTORS+7,:3,3] = xyzs_in_base_frame[i,C1idx,:3] # C1'

    # nu0
    RTs_by_torsion[i,NPROTTORS+8,:3,:3] = make_frame(
        xyzs_in_base_frame[i,C2idx,:3] , torch.tensor([-1.,0.,0.])
    )
    RTs_by_torsion[i,NPROTTORS+8,:3,3] = xyzs_in_base_frame[i,C2idx,:3] # C2'

    # NA chi
    if torsions[i][0] is not None:
        a2 = torsion_indices[i,19,2]
        RTs_by_torsion[i,NPROTTORS+9,:3,:3] = make_frame(
            xyzs_in_base_frame[i,a2,:3] , torch.tensor([-1.,0.,0.])
        )
        RTs_by_torsion[i,NPROTTORS+9,:3,3] = xyzs_in_base_frame[i,a2,:3]
#Small molecules
xyzs_in_base_frame[NNAPROTAAS:,1, :3] = 0
# general FAPE parameters
frame_indices = torch.full((NAATOKENS,NFRAMES,3,2),0, dtype=torch.long)
for i in range(NNAPROTAAS):
    i_l = aa2long[i]
    for j,x in enumerate(frames[i]):
        if x is not None:
            # frames are stored as (residue offset, atom position)
            frame_indices[i,j,0] = torch.tensor((0, i_l.index(x[0])))
            frame_indices[i,j,1] = torch.tensor((0, i_l.index(x[1])))
            frame_indices[i,j,2] = torch.tensor((0, i_l.index(x[2])))
    
### Create atom frames for FAPE loss calculation ###
def get_nxgraph(mol):
    '''build NetworkX graph from openbabel's OBMol'''

    N = mol.NumAtoms()

    # pairs of bonded atoms, openbabel indexes from 1 so readjust to indexing from 0
    bonds = [(bond.GetBeginAtomIdx()-1, bond.GetEndAtomIdx()-1) for bond in openbabel.OBMolBondIter(mol)]

    # connectivity graph
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(bonds)

    return G

def find_all_rigid_groups(bond_feats):
    """
    remove all single bonds from the graph and find connected components
    """
    rigid_atom_bonds = (bond_feats>1)*(bond_feats<5)
    rigid_atom_bonds_np = rigid_atom_bonds[0].cpu().numpy()
    G = nx.from_numpy_matrix(rigid_atom_bonds_np)
    connected_components = nx.connected_components(G)
    connected_components = [cc for cc in connected_components if len(cc)>2]
    connected_components = [torch.tensor(list(combinations(cc,2))) for cc in connected_components]
    if connected_components:
        connected_components = torch.cat(connected_components, dim=0)
    else:
        connected_components = None
    return connected_components

def find_all_paths_of_length_n(G : nx.Graph,
                               n : int) -> torch.Tensor:
    '''find all paths of length N in a networkx graph
    https://stackoverflow.com/questions/28095646/finding-all-paths-walks-of-given-length-in-a-networkx-graph'''

    def findPaths(G,u,n):
        if n==0:
            return [[u]]
        paths = [[u]+path for neighbor in G.neighbors(u) for path in findPaths(G,neighbor,n-1) if u not in path]
        return paths

    # all paths of length n
    allpaths = [tuple(p) if p[0]<p[-1] else tuple(reversed(p))
                for node in G for p in findPaths(G,node,n)]

    # unique paths
    allpaths = list(set(allpaths))

    #return torch.tensor(allpaths)
    return allpaths

def get_atom_frames(msa, G):
    """choose a frame of 3 bonded atoms for each atom in the molecule, rule based system that chooses frame based on atom priorities"""

    query_seq = msa
    frames = find_all_paths_of_length_n(G, 2)
    selected_frames = []
    for n in range(msa.shape[0]):
        frames_with_n = [frame for frame in frames if n == frame[1]]

        # some chemical groups don't have two bonded heavy atoms; so choose a frame with an atom 2 bonds away
        if not frames_with_n:
            frames_with_n = [frame for frame in frames if n in frame]
        # if the atom isn't in a 3 atom frame, it should be ignored in loss calc, set all the atoms to n
        if not frames_with_n:
            selected_frames.append([(0,0),(0,0),(0, 0)])
            continue
        frame_priorities = []
        for frame in frames_with_n:
            # hacky but uses the "query_seq" to convert index of the atom into an "atom type" and converts that into a priority
            indices = [index for index in frame if index!=n]
            aas = [num2aa[int(query_seq[index].numpy())] for index in indices]
            frame_priorities.append(sorted([atom2frame_priority[aa] for aa in aas]))
            
        # np.argsort doesn't sort tuples correctly so just sort a list of indices using a key
        sorted_indices = sorted(range(len(frame_priorities)), key=lambda i: frame_priorities[i])
        # calculate residue offset for frame
        frame = [(frame-n, 1) for frame in frames_with_n[sorted_indices[0]]]
        selected_frames.append(frame)
    assert msa.shape[0] == len(selected_frames)
    return torch.tensor(selected_frames).long()

def get_atomized_protein_frames(msa, ra):
    """ returns a unique frame for each atom in a n_res_atomize of residues """
    r,a = ra.T
    residue_frames = atomized_protein_frames[msa]
    # handle out of bounds at the termini, terminal N and C inherit Calpha frame, offset dimension needs to be updated
    offset = torch.zeros(3,2)
    offset[:,0] = 1
    residue_frames[r[0], 0] = residue_frames[r[0],1] + offset
    residue_frames[r[-1], 2] = residue_frames[r[-1],1] - offset
    # terminal O needs to have O-C-Calpha frame
    residue_frames[r[-1],3,2] = torch.tensor([-2,1])
    frames = residue_frames[r,a]

    # make sure no masked atoms are in frames
    atom_idx = torch.arange(frames.shape[0]).unsqueeze(1).repeat(1,3)
    masked_atoms = ~torch.all((frames[...,0] + atom_idx) < frames.shape[0], dim=-1)
    reframe_indices = masked_atoms.nonzero()
    for i in reframe_indices:
        frames[i] = frames[i-1]-offset
    return frames.long() 
    
### Generate bond features for small molecules ###
def get_bond_feats(mol):                                                                                 
    """creates 2d bond graph for small molecules"""
    N = mol.NumAtoms()
    bond_feats = torch.zeros((N, N)).long()

    for bond in openbabel.OBMolBondIter(mol):
        i,j = (bond.GetBeginAtomIdx()-1, bond.GetEndAtomIdx()-1)
        bond_feats[i,j] = bond.GetBondOrder() if not bond.IsAromatic() else 4
        bond_feats[j,i] = bond_feats[i,j]

    return bond_feats.long()

def get_protein_bond_feats(protein_L):
    """ creates protein residue connectivity graphs """
    bond_feats = torch.zeros((protein_L, protein_L))
    residues = torch.arange(protein_L-1)
    bond_feats[residues, residues+1] = 5
    bond_feats[residues+1, residues] = 5
    return bond_feats

def get_atomize_protein_bond_feats(i_start, msa, ra, n_res_atomize=5):
    """ 
    generate atom bond features for atomized residues 
    currently ignores long-range bonds like disulfides
    """
    ra2ind = {}
    for i, two_d in enumerate(ra):
        ra2ind[tuple(two_d.numpy())] = i
    N = len(ra2ind.keys())
    bond_feats = torch.zeros((N, N))
    for i, res in enumerate(msa[0, i_start:i_start+n_res_atomize]):
        for j, bond in enumerate(aabonds[res]):
            start_idx = aa2long[res].index(bond[0])
            end_idx = aa2long[res].index(bond[1])
            if (i, start_idx) not in ra2ind or (i, end_idx) not in ra2ind:
                #skip bonds with atoms that aren't observed in the structure
                continue
            start_idx = ra2ind[(i, start_idx)]
            end_idx = ra2ind[(i, end_idx)]

            # maps the 2d index of the start and end indices to btype
            bond_feats[start_idx, end_idx] = aabtypes[res][j]
            bond_feats[end_idx, start_idx] = aabtypes[res][j]
        #accounting for peptide bonds
        if i > 0:
            if (i-1, 2) not in ra2ind or (i, 0) not in ra2ind:
                #skip bonds with atoms that aren't observed in the structure
                continue
            start_idx = ra2ind[(i-1, 2)]
            end_idx = ra2ind[(i, 0)]
            bond_feats[start_idx, end_idx] = aabtypes[res][j]
            bond_feats[end_idx, start_idx] = aabtypes[res][j]
    return bond_feats

### Generate atom features for proteins ###
def atomize_protein(i_start, msa, xyz, mask, n_res_atomize=5):
    """ given an index i_start, make the following flank residues into "atom" nodes """
    residues_atomize = msa[0, i_start:i_start+n_res_atomize]
    residues_atom_types = [aa2elt[num][:14] for num in residues_atomize]
    residue_atomize_mask = mask[i_start:i_start+n_res_atomize].float()
    xyz_atomize = xyz[i_start:i_start+n_res_atomize]

    # handle symmetries
    xyz_alt = torch.zeros_like(xyz.unsqueeze(0))
    xyz_alt.scatter_(2, long2alt[msa[0],:,None].repeat(1,1,1,3), xyz.unsqueeze(0))
    xyz_alt_atomize = xyz_alt[0, i_start:i_start+n_res_atomize]

    coords_stack = torch.stack((xyz_atomize, xyz_alt_atomize), dim=0)
    swaps = (coords_stack[0] == coords_stack[1]).all(dim=1).all(dim=1).squeeze() #checks whether theres a swap at each position
    swaps = torch.nonzero(~swaps).squeeze() # indices with a swap eg. [2,3]
    if swaps.numel() != 0:
        # if there are residues with alternate numbering scheme, create a stack of coordinate with each combo of swaps
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",category=UserWarning)
            combs = torch.combinations(torch.tensor([0,1]), r=swaps.numel(), with_replacement=True) #[[0,0], [0,1], [1,1]]
        stack = torch.stack((combs, swaps.repeat(swaps.numel()+1,1)), dim=-1).squeeze()
        coords_stack = coords_stack.repeat(swaps.numel()+1,1,1,1)
        nat_symm = coords_stack[0].repeat(swaps.numel()+1,1,1,1) # (N_symm, num_atomize_residues, natoms, 3)
        swapped_coords = coords_stack[stack[...,0], stack[...,1]].squeeze(1) #
        nat_symm[:,swaps] = swapped_coords
    else:
        nat_symm = xyz_atomize.unsqueeze(0)
    ra = residue_atomize_mask.nonzero()
    lig_seq = torch.tensor([aa2num[residues_atom_types[r][a]] if residues_atom_types[r][a] in aa2num else aa2num["ATM"] for r,a in ra])
    ins = torch.zeros_like(lig_seq)

    r,a = ra.T
    last_C = torch.all(ra==torch.tensor([r[-1],2]),dim=1).nonzero()
    lig_xyz = torch.zeros((len(ra), 3))
    lig_xyz = nat_symm[:, r, a]
    lig_mask = residue_atomize_mask[r, a].repeat(nat_symm.shape[0], 1)
    frames = get_atomized_protein_frames(residues_atomize, ra)
    bond_feats = get_atomize_protein_bond_feats(i_start, msa, ra, n_res_atomize=n_res_atomize)

    chirals = get_atomize_protein_chirals(residues_atomize, lig_xyz[0], residue_atomize_mask, bond_feats)
    return lig_seq, ins, lig_xyz, lig_mask, frames, bond_feats, last_C, chirals

def same_chain_2d_from_Ls(Ls):
    """Given list of chain lengths, returns binary matrix with 1 if two residues are on the same chain."""
    same_chain = torch.zeros((sum(Ls),sum(Ls))).long()
    i_curr = 0
    for L in Ls:
        same_chain[i_curr:i_curr+L, i_curr:i_curr+L] = 1
        i_curr += L
    return same_chain

def Ls_from_same_chain_2d(same_chain):
    """Given binary matrix indicating whether two residues are on same chain, returns list of chain lengths"""
    if len(same_chain.shape)==3: # remove batch dimension
        same_chain = same_chain.squeeze(0)
    Ls = []
    i_curr = 0
    while i_curr < len(same_chain):
        idx = torch.where(same_chain[i_curr])[0]
        Ls.append(idx[-1]-idx[0]+1)
        i_curr = idx[-1]+1
    return Ls       
