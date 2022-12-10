import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract as einsum
import copy
import dgl
import networkx as nx
from util import base_indices, RTs_by_torsion, xyzs_in_base_frame, \
    rigid_from_3_points, is_nucleic, is_atom

def init_lecun_normal(module, scale=1.0):
    def truncated_normal(uniform, mu=0.0, sigma=1.0, a=-2, b=2):
        normal = torch.distributions.normal.Normal(0, 1)

        alpha = (a - mu) / sigma
        beta = (b - mu) / sigma

        alpha_normal_cdf = normal.cdf(torch.tensor(alpha))
        p = alpha_normal_cdf + (normal.cdf(torch.tensor(beta)) - alpha_normal_cdf) * uniform

        v = torch.clamp(2 * p - 1, -1 + 1e-8, 1 - 1e-8)
        x = mu + sigma * np.sqrt(2) * torch.erfinv(v)
        x = torch.clamp(x, a, b)

        return x

    def sample_truncated_normal(shape, scale=1.0):
        stddev = np.sqrt(scale/shape[-1])/.87962566103423978  # shape[-1] = fan_in
        return stddev * truncated_normal(torch.rand(shape))

    module.weight = torch.nn.Parameter( (sample_truncated_normal(module.weight.shape)) )
    return module

def init_lecun_normal_param(weight, scale=1.0):
    def truncated_normal(uniform, mu=0.0, sigma=1.0, a=-2, b=2):
        normal = torch.distributions.normal.Normal(0, 1)

        alpha = (a - mu) / sigma
        beta = (b - mu) / sigma

        alpha_normal_cdf = normal.cdf(torch.tensor(alpha))
        p = alpha_normal_cdf + (normal.cdf(torch.tensor(beta)) - alpha_normal_cdf) * uniform

        v = torch.clamp(2 * p - 1, -1 + 1e-8, 1 - 1e-8)
        x = mu + sigma * np.sqrt(2) * torch.erfinv(v)
        x = torch.clamp(x, a, b)

        return x

    def sample_truncated_normal(shape, scale=1.0):
        stddev = np.sqrt(scale/shape[-1])/.87962566103423978  # shape[-1] = fan_in
        return stddev * truncated_normal(torch.rand(shape))

    weight = torch.nn.Parameter( (sample_truncated_normal(weight.shape)) )
    return weight

# for gradient checkpointing
def create_custom_forward(module, **kwargs):
    def custom_forward(*inputs):
        return module(*inputs, **kwargs)
    return custom_forward

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Dropout(nn.Module):
    # Dropout entire row or column
    def __init__(self, broadcast_dim=None, p_drop=0.15):
        super(Dropout, self).__init__()
        # give ones with probability of 1-p_drop / zeros with p_drop
        self.sampler = torch.distributions.bernoulli.Bernoulli(torch.tensor([1-p_drop]))
        self.broadcast_dim=broadcast_dim
        self.p_drop=p_drop
    def forward(self, x):
        if not self.training: # no drophead during evaluation mode
            return x
        shape = list(x.shape)
        if not self.broadcast_dim == None:
            shape[self.broadcast_dim] = 1
        mask = self.sampler.sample(shape).to(x.device).view(shape)

        x = mask * x / (1.0 - self.p_drop)
        return x

def rbf(D, D_min=0.0, D_count=64, D_sigma=0.5):
    # Distance radial basis function
    D_max = D_min + (D_count-1) * D_sigma
    D_mu = torch.linspace(D_min, D_max, D_count).to(D.device)
    D_mu = D_mu[None,:]
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
    return RBF

def get_seqsep(idx):
    '''
    Sequence separation feature for structure module. Protein-only.

    Input:
        - idx: residue indices of given sequence (B,L)
    Output:
        - seqsep: sequence separation feature with sign (B, L, L, 1)
                  Sergey found that having sign in seqsep features helps a little
    '''
    seqsep = idx[:,None,:] - idx[:,:,None]
    sign = torch.sign(seqsep)
    neigh = torch.abs(seqsep)
    neigh[neigh > 1] = 0.0 # if bonded -- 1.0 / else 0.0
    neigh = sign * neigh
    return neigh.unsqueeze(-1)

def get_seqsep_protein_sm(idx, bond_feats, sm_mask):
    '''
    Sequence separation features for protein-SM complex

    Input:
        - idx: residue indices of given sequence (B,L)
        - bond_feats: bond features (B, L, L)
        - sm_mask: boolean feature True if a position represents atom, False if residue (B, L)

    Output:
        - seqsep: sequence separation feature with sign (B, L, L, 1)
            -1 or 1 for bonded protein residues
            1 for bonded SM atoms or residue-atom bonds
            0 elsewhere
    '''
    sm_mask = sm_mask[0] # assume batch = 1
    res_dist, atom_dist = get_res_atom_dist(idx, bond_feats, sm_mask)

    sm_mask_2d = sm_mask[None,:]*sm_mask[:,None]
    prot_mask_2d = (~sm_mask[None,:]) * (~sm_mask[:,None])
    inter_mask_2d = (~sm_mask[None,:]) * (sm_mask[:,None]) + (sm_mask[None,:]) * (~sm_mask[:,None])

    res_dist[(res_dist > 1) | (res_dist < -1)] = 0.0
    atom_dist[(atom_dist > 1)] = 0.0
    
    seqsep = sm_mask_2d*atom_dist + prot_mask_2d*res_dist + inter_mask_2d*(bond_feats==6)

    return seqsep.unsqueeze(-1)

def get_res_atom_dist(idx, bond_feats, sm_mask, minpos_res=-32, maxpos_res=32, maxpos_atom=8):
    '''
    Calculates residue and atom bond distances of protein/SM complex. Used for positional
    embedding and structure module. 2nd version (2022-9-19); handles atomized proteins.

    Input:
        - idx: residue index (B, L)
        - bond_feats: bond features (B, L, L)
        - sm_mask: boolean feature (L). True if a position represents atom, False otherwise
        - minpos_res: minimum value of residue distances
        - maxpos_res: maximum value of residue distances
        - maxpos_atom: maximum value of atom bond distances

    Output:
        - res_dist: residue distance (B, L, L)
        - atom_dist: atom bond distance (B, L, L)
    '''
    bond_feats = bond_feats[0] # assume batch = 1
    L = bond_feats.shape[0]
    gpu = bond_feats.device

    sm_mask_2d = sm_mask[None,:]*sm_mask[:,None]
    prot_mask_2d = (~sm_mask[None,:]) * (~sm_mask[:,None])
    inter_mask_2d = (~sm_mask[None,:]) * (sm_mask[:,None]) + (sm_mask[None,:]) * (~sm_mask[:,None])

    # protein residue distances
    res_dist_prot = torch.clamp(idx[0,None,:] - idx[0,:,None],
                               min=minpos_res, max=maxpos_res).to(gpu) # (L, L) intra-protein
    res_dist_sm = torch.full((L,L), maxpos_res+1).to(gpu) # (L, L) with "unknown" res. dist. token

    # small molecule atom bond graph
    sm_bond_feats = torch.zeros_like(bond_feats) + sm_mask_2d*bond_feats
    G = nx.from_numpy_matrix(sm_bond_feats.detach().cpu().numpy())
    paths = dict(nx.all_pairs_shortest_path_length(G,cutoff=maxpos_atom))
    paths = [(i,j,vij) for i,vi in paths.items() for j,vij in vi.items()]
    i,j,v = torch.tensor(paths).T

    # small molecule atom bond distances
    atom_dist_sm = torch.full((L,L), maxpos_atom).to(gpu) - maxpos_atom*torch.eye(L).to(gpu).long()
    atom_dist_sm[i,j] = v.to(gpu)
    atom_dist_prot = torch.full((L,L), maxpos_atom+1).to(gpu)

    # s.m.-protein bonds
    sm_idx = torch.where(sm_mask)[0]
    prot_idx = torch.where(~sm_mask)[0]
    i_s, j_s = torch.where(bond_feats==6)
    i_prot = [j for i,j in zip(i_s,j_s) if i in sm_idx] # protein residues bonded to s.m. atoms
    i_sm = [i for i in i_s if i in sm_idx] # s.m. atoms bonded to protein residues

    # inter-protein-s.m. residue & atom distances
    # atoms inherit residue distances from their nearest bonded residue
    res_dist_inter = torch.full((L,L), maxpos_res).to(gpu)
    if len(i_prot) > 0: # prot & s.m. are connected
        for i in sm_idx:
            i_closest_res = i_prot[torch.argmin(atom_dist_sm[i,i_sm])]
            res_dist_inter[i,:] = res_dist_prot[i_closest_res,:]
            res_dist_inter[:,i] = res_dist_prot[:,i_closest_res]

    # residues inherit atom distances from their nearest bonded atom (+ 1 to count "boundary" res-atom bond)
    atom_dist_inter = torch.full((L, L), maxpos_atom).to(gpu)
    if len(i_prot) > 0: # prot & s.m. are connected
        for i in prot_idx:
            i_closest_atom = i_sm[torch.argmin(torch.abs(res_dist_prot[i,i_prot]))]
            atom_dist_inter[i,:] = atom_dist_sm[i_closest_atom,:] + 1
            atom_dist_inter[:,i] = atom_dist_sm[:,i_closest_atom] + 1
        atom_dist_inter = torch.minimum(atom_dist_inter, torch.tensor(maxpos_atom))

    res_dist = res_dist_prot * prot_mask_2d + res_dist_inter * inter_mask_2d + res_dist_sm * sm_mask_2d
    atom_dist = atom_dist_prot * prot_mask_2d + atom_dist_inter * inter_mask_2d + atom_dist_sm * sm_mask_2d

    return res_dist[None].to(gpu), atom_dist[None].to(gpu) # add batch dim.

def get_relpos(idx, bond_feats, sm_mask, inter_pos=32, maxpath=32):
    '''
    Relative position matrix of protein/SM complex. Used for positional
    embedding and structure module. Simple version from 9/2/2022 that doesn't
    handle atomized proteins.

    Input:
        - idx: residue index (B, L)
        - bond_feats: bond features (B, L, L)
        - sm_mask: boolean feature True if a position represents atom, False if residue (B, L)
        - inter_pos: value to assign as the protein-SM residue index differences
        - maxpath: bond distances greater than this are clipped to this value

    Output:
        - relpos: relative position feature (B, L, L)
            for intra-protein this is the residue index difference
            for intra-SM this is the bond distance
            for protein-SM this is user-defined value inter_pos
    '''
    bond_feats = bond_feats[0]

    sm_mask_2d = sm_mask[None,:]*sm_mask[:,None]
    prot_mask_2d = (~sm_mask[None,:]) * (~sm_mask[:,None])
    inter_mask_2d = (~sm_mask[None,:]) * (sm_mask[:,None]) + (sm_mask[None,:]) * (~sm_mask[:,None])

    # intra-protein: residue # differences
    seqsep = idx[:,None,:] - idx[:,:,None] # (B, L, L)

    # intra-small molecule: bond distances
    sm_bond_feats = torch.zeros_like(bond_feats) + sm_mask*bond_feats                                      
    G = nx.from_numpy_matrix(sm_bond_feats.detach().cpu().numpy())
    paths = dict(nx.all_pairs_shortest_path_length(G,cutoff=maxpath))
    paths = [(i,j,vij) for i,vi in paths.items() for j,vij in vi.items()]
    i,j,v = torch.tensor(paths).T

    bond_separation = torch.full_like(bond_feats, maxpath) \
        - maxpath*torch.eye(bond_feats.shape[0]).to(bond_feats.device).long()
    bond_separation[i,j] = v.to(bond_feats.device)

    # combine: protein-s.m. are always positive maximum distance apart
    # assumes one small molecule per example
    relpos = prot_mask_2d * seqsep + sm_mask_2d * bond_separation + inter_mask_2d * inter_pos # (B, L, L)
    relpos = relpos.to(bond_feats.device)

    return relpos

def make_full_graph(xyz, pair, idx):
    '''
    Input:
        - xyz: current backbone cooordinates (B, L, 3, 3)
        - pair: pair features from Trunk (B, L, L, E)
        - idx: residue index from ground truth pdb
    Output:
        - G: defined graph
    '''

    B, L = xyz.shape[:2]
    device = xyz.device
    
    # seq sep
    sep = idx[:,None,:] - idx[:,:,None]
    b,i,j = torch.where(sep.abs() > 0)
   
    src = b*L+i
    tgt = b*L+j
    G = dgl.graph((src, tgt), num_nodes=B*L).to(device)
    G.edata['rel_pos'] = (xyz[b,j,:] - xyz[b,i,:]).detach() # no gradient through basis function

    return G, pair[b,i,j][...,None]

def make_topk_graph(xyz, pair, idx, top_k=128, nlocal=33, topk_incl_local=True, eps=1e-6):
    '''
    Input:
        - xyz: current backbone cooordinates (B, L, 3, 3)
        - pair: pair features from Trunk (B, L, L, E)
        - idx: residue index from ground truth pdb
    Output:
        - G: defined graph
    '''

    B, L = xyz.shape[:2]
    device = xyz.device

    # distance map from current CA coordinates
    D = torch.cdist(xyz, xyz) + torch.eye(L, device=device).unsqueeze(0)*9999.9  # (B, L, L)

    # seq sep
    sep = idx[:,None,:] - idx[:,:,None]
    sep = sep.abs() + torch.eye(L, device=device).unsqueeze(0)*9999.9

    if (topk_incl_local):
        D = D + sep*eps
        D[sep<nlocal] = 0.0

        # get top_k neighbors
        D_neigh, E_idx = torch.topk(D, min(top_k, L-1), largest=False) # shape of E_idx: (B, L, top_k)
        topk_matrix = torch.zeros((B, L, L), device=device)
        topk_matrix.scatter_(2, E_idx, 1.0)
        cond = topk_matrix > 0.0

    else:

        D = D + sep*eps

        # get top_k neighbors
        D_neigh, E_idx = torch.topk(D, min(top_k, L-1), largest=False) # shape of E_idx: (B, L, top_k)
        topk_matrix = torch.zeros((B, L, L), device=device)
        topk_matrix.scatter_(2, E_idx, 1.0)

    # put an edge if any of the 3 conditions are met:
    #   1) |i-j| <= kmin (connect sequentially adjacent residues)
    #   2) top_k neighbors
    cond = torch.logical_or(topk_matrix > 0.0, sep < nlocal)
    b,i,j = torch.where(cond)

    src = b*L+i
    tgt = b*L+j
    G = dgl.graph((src, tgt), num_nodes=B*L).to(device)
    G.edata['rel_pos'] = (xyz[b,j,:] - xyz[b,i,:]).detach() # no gradient through basis function

    return G, pair[b,i,j][...,None]

def make_atom_graph( xyz, mask, num_bonds, top_k=16, maxbonds=4 ):
    B,L,A = xyz.shape[:3]
    device = xyz.device

    D = torch.norm(
        xyz[:,None,None,:,:] - xyz[:,:,:,None,None], dim=-1
    )
    mask2d = mask[:,:,:,None,None]*mask[:,None,None,:,:]
    D[~mask2d] = 9999.
    D[D==0] = 9999.

    # select top K neighbors for each atom
    # keep indices as batch/res/atm indices
    D_neigh, E_idx = torch.topk(D.reshape(B,L,A,-1), top_k, largest=False) # shape of E_idx: (B, L, top_k)
    Eres, Eatm = torch.div(E_idx,A,rounding_mode='trunc'), E_idx%A
    bi,ri,ai = mask.nonzero(as_tuple=True)
    bi = bi[:,None].repeat(1,top_k).reshape(-1)
    ri = ri[:,None].repeat(1,top_k).reshape(-1)
    ai = ai[:,None].repeat(1,top_k).reshape(-1)
    rj,aj = Eres[mask].reshape(-1), Eatm[mask].reshape(-1)

    # on each edge, 1-hot encode the number of bonds (up to maxbonds) seperating each atom
    edge = torch.full(ri.shape, maxbonds, device=device)
    resmask = ri==rj
    edge[resmask] = num_bonds[bi[resmask],ri[resmask],ai[resmask],aj[resmask]]-1
    resmask = ri+1==rj
    edge[resmask] = num_bonds[bi[resmask],ri[resmask],ai[resmask],2]+num_bonds[bi[resmask],rj[resmask],0,aj[resmask]]
    resmask = ri-1==rj
    edge[resmask] = num_bonds[bi[resmask],ri[resmask],ai[resmask],0]+num_bonds[bi[resmask],rj[resmask],2,aj[resmask]]
    edge = edge.clamp(0,maxbonds-1)
    edge = F.one_hot(edge)[...,None]

    natm = torch.sum(mask)
    index = torch.zeros_like(mask, dtype=torch.long, device=device)
    index[mask] = torch.arange(natm, device=device)
    src=index[bi,ri,ai]
    tgt=index[bi,rj,aj]
    
    G = dgl.graph((src, tgt), num_nodes=natm).to(device)
    G.edata['rel_pos'] = (xyz[bi,ri,ai] - xyz[bi,rj,aj]).detach() # no gradient through basis function

    return G, edge


# rotate about the x axis
def make_rotX(angs, eps=1e-6):
    B,L = angs.shape[:2]
    NORM = torch.linalg.norm(angs, dim=-1) + eps

    RTs = torch.eye(4,  device=angs.device).repeat(B,L,1,1)

    RTs[:,:,1,1] = angs[:,:,0]/NORM
    RTs[:,:,1,2] = -angs[:,:,1]/NORM
    RTs[:,:,2,1] = angs[:,:,1]/NORM
    RTs[:,:,2,2] = angs[:,:,0]/NORM
    return RTs

# rotate about the x axis
def make_rotZ(angs, eps=1e-6):
    B,L = angs.shape[:2]
    NORM = torch.linalg.norm(angs, dim=-1) + eps

    RTs = torch.eye(4,  device=angs.device).repeat(B,L,1,1)

    RTs[:,:,0,0] = angs[:,:,0]/NORM
    RTs[:,:,0,1] = -angs[:,:,1]/NORM
    RTs[:,:,1,0] = angs[:,:,1]/NORM
    RTs[:,:,1,1] = angs[:,:,0]/NORM
    return RTs

# rotate about an arbitrary axis
def make_rot_axis(angs, u, eps=1e-6):
    B,L = angs.shape[:2]
    NORM = torch.linalg.norm(angs, dim=-1) + eps

    RTs = torch.eye(4,  device=angs.device).repeat(B,L,1,1)

    ct = angs[:,:,0]/NORM
    st = angs[:,:,1]/NORM
    u0 = u[:,:,0]
    u1 = u[:,:,1]
    u2 = u[:,:,2]

    RTs[:,:,0,0] = ct+u0*u0*(1-ct)
    RTs[:,:,0,1] = u0*u1*(1-ct)-u2*st
    RTs[:,:,0,2] = u0*u2*(1-ct)+u1*st
    RTs[:,:,1,0] = u0*u1*(1-ct)+u2*st
    RTs[:,:,1,1] = ct+u1*u1*(1-ct)
    RTs[:,:,1,2] = u1*u2*(1-ct)-u0*st
    RTs[:,:,2,0] = u0*u2*(1-ct)-u1*st
    RTs[:,:,2,1] = u1*u2*(1-ct)+u0*st
    RTs[:,:,2,2] = ct+u2*u2*(1-ct)
    return RTs


# compute allatom structure from backbone frames and torsions
#
# alphas:
#    omega/phi/psi: 0-2
#    chi_1-4(prot): 3-6
#    cb/cg bend: 7-9
#    eps(p)/zeta(p): 10-11
#    alpha/beta/gamma/delta: 12-15
#    nu2/nu1/nu0: 16-18
#    chi_1(na): 19
# 
# RTs_in_base_frame:
#    omega/phi/psi: 0-2
#    chi_1-4(prot): 3-6
#    eps(p)/zeta(p): 7-8
#    alpha/beta/gamma/delta: 9-12
#    nu2/nu1/nu0: 13-15
#    chi_1(na): 16
#
# RT frames (output):
#    origin: 0
#    omega/phi/psi: 1-3
#    chi_1-4(prot): 4-7
#    cb bend: 8
#    alpha/beta/gamma/delta: 9-12
#    nu2/nu1/nu0: 13-15
#    chi_1(na): 16
#
class ComputeAllAtomCoords(nn.Module):
    def __init__(self):
        super(ComputeAllAtomCoords, self).__init__()

        self.base_indices = nn.Parameter(base_indices, requires_grad=False)
        self.RTs_in_base_frame = nn.Parameter(RTs_by_torsion, requires_grad=False)
        self.xyzs_in_base_frame = nn.Parameter(xyzs_in_base_frame, requires_grad=False)

    def forward(self, seq, xyz, alphas):
        B,L = xyz.shape[:2]

        is_NA = is_nucleic(seq)
        Rs, Ts = rigid_from_3_points(xyz[...,0,:],xyz[...,1,:],xyz[...,2,:], is_NA)

        RTF0 = torch.eye(4).repeat(B,L,1,1).to(device=Rs.device)

        # bb
        RTF0[:,:,:3,:3] = Rs
        RTF0[:,:,:3,3] = Ts

        # omega
        RTF1 = torch.einsum(
            'brij,brjk,brkl->bril',
            RTF0, self.RTs_in_base_frame[seq,0,:], make_rotX(alphas[:,:,0,:]))

        # phi
        RTF2 = torch.einsum(
            'brij,brjk,brkl->bril', 
            RTF0, self.RTs_in_base_frame[seq,1,:], make_rotX(alphas[:,:,1,:]))

        # psi
        RTF3 = torch.einsum(
            'brij,brjk,brkl->bril', 
            RTF0, self.RTs_in_base_frame[seq,2,:], make_rotX(alphas[:,:,2,:]))

        # CB bend
        basexyzs = self.xyzs_in_base_frame[seq]
        NCr = 0.5*(basexyzs[:,:,2,:3]+basexyzs[:,:,0,:3])
        CAr = (basexyzs[:,:,1,:3])
        CBr = (basexyzs[:,:,4,:3])
        CBrotaxis1 = (CBr-CAr).cross(NCr-CAr)
        CBrotaxis1 /= torch.linalg.norm(CBrotaxis1, dim=-1, keepdim=True)+1e-8

        # CB twist
        NCp = basexyzs[:,:,2,:3] - basexyzs[:,:,0,:3]
        NCpp = NCp - torch.sum(NCp*NCr, dim=-1, keepdim=True)/ torch.sum(NCr*NCr, dim=-1, keepdim=True) * NCr
        CBrotaxis2 = (CBr-CAr).cross(NCpp)
        CBrotaxis2 /= torch.linalg.norm(CBrotaxis2, dim=-1, keepdim=True)+1e-8
        
        CBrot1 = make_rot_axis(alphas[:,:,7,:], CBrotaxis1 )
        CBrot2 = make_rot_axis(alphas[:,:,8,:], CBrotaxis2 )
        
        RTF8 = torch.einsum(
            'brij,brjk,brkl->bril', 
            RTF0, CBrot1,CBrot2)

        # chi1 + CG bend
        RTF4 = torch.einsum(
            'brij,brjk,brkl,brlm->brim', 
            RTF8, 
            self.RTs_in_base_frame[seq,3,:], 
            make_rotX(alphas[:,:,3,:]), 
            make_rotZ(alphas[:,:,9,:]))

        # chi2
        RTF5 = torch.einsum(
            'brij,brjk,brkl->bril', 
            RTF4, self.RTs_in_base_frame[seq,4,:],make_rotX(alphas[:,:,4,:]))

        # chi3
        RTF6 = torch.einsum(
            'brij,brjk,brkl->bril', 
            RTF5,self.RTs_in_base_frame[seq,5,:],make_rotX(alphas[:,:,5,:]))

        # chi4
        RTF7 = torch.einsum(
            'brij,brjk,brkl->bril', 
            RTF6,self.RTs_in_base_frame[seq,6,:],make_rotX(alphas[:,:,6,:]))

        # ignore RTs_in_base_frame[seq,7:9,:] and alphas[:,:,10:12,:]

        # NA alpha
        RTF9 = torch.einsum(
            'brij,brjk,brkl->bril', 
            RTF0, self.RTs_in_base_frame[seq,9,:], make_rotX(alphas[:,:,12,:]))

        # NA beta
        RTF10 = torch.einsum(
            'brij,brjk,brkl->bril', 
            RTF9, self.RTs_in_base_frame[seq,10,:], make_rotX(alphas[:,:,13,:]))

        # NA gamma
        RTF11 = torch.einsum(
            'brij,brjk,brkl->bril', 
            RTF10, self.RTs_in_base_frame[seq,11,:], make_rotX(alphas[:,:,14,:]))

        # NA delta
        RTF12 = torch.einsum(
            'brij,brjk,brkl->bril', 
            RTF11, self.RTs_in_base_frame[seq,12,:], make_rotX(alphas[:,:,15,:]))

        # NA nu2 - from gamma frame
        RTF13 = torch.einsum(
            'brij,brjk,brkl->bril', 
            RTF11, self.RTs_in_base_frame[seq,13,:], make_rotX(alphas[:,:,16,:]))

        # NA nu1
        RTF14 = torch.einsum(
            'brij,brjk,brkl->bril', 
            RTF13, self.RTs_in_base_frame[seq,14,:], make_rotX(alphas[:,:,17,:]))

        # NA nu0
        RTF15 = torch.einsum(
            'brij,brjk,brkl->bril', 
            RTF14, self.RTs_in_base_frame[seq,15,:], make_rotX(alphas[:,:,18,:]))

        # NA chi - from nu1 frame
        RTF16= torch.einsum(
            'brij,brjk,brkl->bril', 
            RTF14, self.RTs_in_base_frame[seq,16,:], make_rotX(alphas[:,:,19,:]))

        RTframes = torch.stack((
            RTF0,RTF1,RTF2,RTF3,RTF4,RTF5,RTF6,RTF7,RTF8,
            RTF9,RTF10,RTF11,RTF12,RTF13,RTF14,RTF15,RTF16
        ),dim=2)

        xyzs = torch.einsum(
            'brtij,brtj->brti', 
            RTframes.gather(2,self.base_indices[seq][...,None,None].repeat(1,1,1,4,4)), basexyzs
        )

        return RTframes, xyzs[...,:3]
