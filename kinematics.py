import numpy as np
import torch
from util import INIT_CRDS, INIT_NA_CRDS, generate_Cbeta, is_nucleic, is_atom
from chemical import NTOTAL

PARAMS = {
    "DMIN"    : 2.0,
    "DMAX"    : 20.0,
    "DBINS"   : 36,
    "ABINS"   : 36,
}

# ============================================================
def get_pair_dist(a, b):
    """calculate pair distances between two sets of points
    
    Parameters
    ----------
    a,b : pytorch tensors of shape [batch,nres,3]
          store Cartesian coordinates of two sets of atoms
    Returns
    -------
    dist : pytorch tensor of shape [batch,nres,nres]
           stores paitwise distances between atoms in a and b
    """

    dist = torch.cdist(a, b, p=2)
    return dist

# ============================================================
def get_ang(a, b, c, eps=1e-6):
    """calculate planar angles for all consecutive triples (a[i],b[i],c[i])
    from Cartesian coordinates of three sets of atoms a,b,c 

    Parameters
    ----------
    a,b,c : pytorch tensors of shape [batch,nres,3]
            store Cartesian coordinates of three sets of atoms
    Returns
    -------
    ang : pytorch tensor of shape [batch,nres]
          stores resulting planar angles
    """
    v = a - b
    w = c - b
    vn = v / (torch.norm(v, dim=-1, keepdim=True)+eps)
    wn = w / (torch.norm(w, dim=-1, keepdim=True)+eps)
    vw = torch.sum(vn*wn, dim=-1)

    return torch.acos(torch.clamp(vw,-0.999,0.999))

# ============================================================
def get_dih(a, b, c, d, eps=1e-6):
    """calculate dihedral angles for all consecutive quadruples (a[i],b[i],c[i],d[i])
    given Cartesian coordinates of four sets of atoms a,b,c,d

    Parameters
    ----------
    a,b,c,d : pytorch tensors of shape [batch,nres,3]
              store Cartesian coordinates of four sets of atoms
    Returns
    -------
    dih : pytorch tensor of shape [batch,nres]
          stores resulting dihedrals
    """
    b0 = a - b
    b1 = c - b
    b2 = d - c

    b1n = b1 / (torch.norm(b1, dim=-1, keepdim=True) + eps)

    v = b0 - torch.sum(b0*b1n, dim=-1, keepdim=True)*b1n
    w = b2 - torch.sum(b2*b1n, dim=-1, keepdim=True)*b1n

    x = torch.sum(v*w, dim=-1)
    y = torch.sum(torch.cross(b1n,v,dim=-1)*w, dim=-1)

    return torch.atan2(y+eps, x+eps)


# ============================================================
def xyz_to_c6d(xyz, params=PARAMS):
    """convert cartesian coordinates into 2d distance 
    and orientation maps
    
    Parameters
    ----------
    xyz : pytorch tensor of shape [batch,nres,3,3]
          stores Cartesian coordinates of backbone N,Ca,C atoms
    Returns
    -------
    c6d : pytorch tensor of shape [batch,nres,nres,4]
          stores stacked dist,omega,theta,phi 2D maps 
    """
    
    batch = xyz.shape[0]
    nres = xyz.shape[1]

    # three anchor atoms
    N  = xyz[:,:,0]
    Ca = xyz[:,:,1]
    C  = xyz[:,:,2]

    # recreate Cb given N,Ca,C
    Cb = generate_Cbeta(N,Ca,C)

    # 6d coordinates order: (dist,omega,theta,phi)
    c6d = torch.zeros([batch,nres,nres,4],dtype=xyz.dtype,device=xyz.device)

    dist = get_pair_dist(Cb,Cb)
    dist[torch.isnan(dist)] = 999.9
    c6d[...,0] = dist + 999.9*torch.eye(nres,device=xyz.device)[None,...]
    b,i,j = torch.where(c6d[...,0]<params['DMAX'])

    c6d[b,i,j,torch.full_like(b,1)] = get_dih(Ca[b,i], Cb[b,i], Cb[b,j], Ca[b,j])
    c6d[b,i,j,torch.full_like(b,2)] = get_dih(N[b,i], Ca[b,i], Cb[b,i], Cb[b,j])
    c6d[b,i,j,torch.full_like(b,3)] = get_ang(Ca[b,i], Cb[b,i], Cb[b,j])

    # fix long-range distances
    c6d[...,0][c6d[...,0]>=params['DMAX']] = 999.9
    
    mask = torch.zeros((batch, nres,nres), dtype=xyz.dtype, device=xyz.device)
    mask[b,i,j] = 1.0
    return c6d, mask
    
def xyz_to_t2d(xyz_t, params=PARAMS):
    """convert template cartesian coordinates into 2d distance 
    and orientation maps
    
    Parameters
    ----------
    xyz_t : pytorch tensor of shape [batch,templ,nres,3,3]
            stores Cartesian coordinates of template backbone N,Ca,C atoms

    Returns
    -------
    t2d : pytorch tensor of shape [batch,nres,nres,37+6+3]
          stores stacked dist,omega,theta,phi 2D maps 
    """
    B, T, L = xyz_t.shape[:3]
    c6d, mask = xyz_to_c6d(xyz_t[:,:,:,:3].view(B*T,L,3,3), params=params)
    c6d = c6d.view(B, T, L, L, 4)
    mask = mask.view(B, T, L, L, 1)
    #
    # dist to one-hot encoded
    dist = dist_to_onehot(c6d[...,0], params)
    orien = torch.cat((torch.sin(c6d[...,1:]), torch.cos(c6d[...,1:])), dim=-1)*mask # (B, T, L, L, 6)
    #
    mask = torch.isnan(c6d[:,:,:,:,0]) # (B, T, L, L)
    t2d = torch.cat((dist, orien, mask.unsqueeze(-1)), dim=-1)
    t2d[torch.isnan(t2d)] = 0.0
    return t2d

def xyz_to_bbtor(xyz, params=PARAMS):
    batch = xyz.shape[0]
    nres = xyz.shape[1]

    # three anchor atoms
    N  = xyz[:,:,0]
    Ca = xyz[:,:,1]
    C  = xyz[:,:,2]

    # recreate Cb given N,Ca,C
    next_N = torch.roll(N, -1, dims=1)
    prev_C = torch.roll(C, 1, dims=1)
    phi = get_dih(prev_C, N, Ca, C)
    psi = get_dih(N, Ca, C, next_N)
    #
    phi[:,0] = 0.0
    psi[:,-1] = 0.0
    #
    astep = 2.0*np.pi / params['ABINS']
    phi_bin = torch.round((phi+np.pi-astep/2)/astep)
    psi_bin = torch.round((psi+np.pi-astep/2)/astep)
    return torch.stack([phi_bin, psi_bin], axis=-1).long()

# ============================================================
def dist_to_onehot(dist, params=PARAMS):
    dist[torch.isnan(dist)] = 999.9
    dstep = (params['DMAX'] - params['DMIN']) / params['DBINS']
    dbins = torch.linspace(params['DMIN']+dstep, params['DMAX'], params['DBINS'],dtype=dist.dtype,device=dist.device)
    db = torch.bucketize(dist.contiguous(),dbins).long()
    dist = torch.nn.functional.one_hot(db, num_classes=params['DBINS']+1).float()
    return dist

# ============================================================
def dist_to_bins(dist,params=PARAMS):
    """bin 2d distance maps
    """

    dstep = (params['DMAX'] - params['DMIN']) / params['DBINS']
    db = torch.round((dist-params['DMIN']-dstep/2)/dstep)

    db[db<0] = 0
    db[db>params['DBINS']] = params['DBINS']
    
    return db.long()


# ============================================================
def c6d_to_bins(c6d, same_chain, negative=False, params=PARAMS):
    """bin 2d distance and orientation maps
    """

    dstep = (params['DMAX'] - params['DMIN']) / params['DBINS']
    astep = 2.0*np.pi / params['ABINS']

    db = torch.round((c6d[...,0]-params['DMIN']-dstep/2)/dstep)
    ob = torch.round((c6d[...,1]+np.pi-astep/2)/astep)
    tb = torch.round((c6d[...,2]+np.pi-astep/2)/astep)
    pb = torch.round((c6d[...,3]-astep/2)/astep)

    # put all d<dmin into one bin
    db[db<0] = 0
    
    # synchronize no-contact bins
    db[db>params['DBINS']] = params['DBINS']
    ob[db==params['DBINS']] = params['ABINS']
    tb[db==params['DBINS']] = params['ABINS']
    pb[db==params['DBINS']] = params['ABINS']//2

    if negative:
        db = torch.where(same_chain.bool(), db.long(), params['DBINS'])
        ob = torch.where(same_chain.bool(), ob.long(), params['ABINS'])
        tb = torch.where(same_chain.bool(), tb.long(), params['ABINS'])
        pb = torch.where(same_chain.bool(), pb.long(), params['ABINS']//2)
    
    return torch.stack([db,ob,tb,pb],axis=-1).long()

def get_init_xyz(seq, xyz_t, same_chain):
    """
    Initializes coordinates of missing residues in xyz_t. Each template is
    centroid-subtracted. Then, missing positions (protein/NA residues with any
    NaN-coordinates or sm. mol. positions with NaN CA coord) are initialized as
    triangles at origin or to position of nearest neighbor on same chain.

    Parameters
    ----------
    seq : (B, L)
    xyz_t : (B, T, L, Natms, 3)
    same_chain : (B, L, L)

    Returns
    -------
    xyz : (B, T, L, Natms, 3)
    
    """
    B, T, L = xyz_t.shape[:3]

    init = torch.full((B,T,L,NTOTAL,3), np.nan, device=xyz_t.device)
    na_mask = is_nucleic(seq).to(xyz_t.device) # (1, L)
    sm_mask = is_atom(seq).to(xyz_t.device) # (1, L)
    b_nmask,l_nmask = na_mask.nonzero(as_tuple=True) # nucleic acid bases
    b_pmask,l_pmask = (~na_mask).nonzero(as_tuple=True) # protein residues & small mols

    init[b_pmask,:,l_pmask] = INIT_CRDS[None,...].to(xyz_t.device)
    init[b_nmask,:,l_nmask] = INIT_NA_CRDS[None,...].to(xyz_t.device)

    if torch.isnan(xyz_t).all():
        return init

    mask_prot = torch.isnan(xyz_t[:,:,:,:3]).any(dim=-1).any(dim=-1) & ~sm_mask # missing protein/NA residues (B, T, L)
    mask_ca_sm = torch.isnan(xyz_t[:,:,:,1]).any(dim=-1) & sm_mask # missing sm mol positions (B, T, L)
    mask = mask_prot | mask_ca_sm # missing any type of position

    # move CA centroid to origin
    center_CA = ((~mask[:,:,:,None]) * torch.nan_to_num(xyz_t[:,:,:,1,:])).sum(dim=2) / ((~mask[:,:,:,None]).sum(dim=2)+1e-4) # (B, T, 3)
    xyz_t = xyz_t - center_CA.view(B,T,1,1,3)

    # move to nearest neighbor on same chain
    idx_s = list()
    for i_b in range(B):
        for i_T in range(T):
            if mask[i_b, i_T].all():
                continue
            exist_in_templ = torch.where(~mask[i_b, i_T])[0] # (L_sub)
            is_same_chain_in_templ = same_chain[i_b][:,~mask[i_b,i_T]].bool() # (L, L_sub)
            seqmap = (torch.arange(L, device=xyz_t.device)[:,None] - exist_in_templ[None,:]).abs() # (L, L_sub)
            seqmap[~is_same_chain_in_templ] += 99999
            seqmap = torch.argmin(seqmap, dim=-1) # (L)
            idx = torch.gather(exist_in_templ, -1, seqmap) # (L)
            offset_CA = torch.gather(xyz_t[i_b, i_T, :, 1, :], 0, idx.reshape(L,1).expand(-1,3))
            has_neighbor = is_same_chain_in_templ.all(-1) 
            offset_CA[~has_neighbor] = 0 # stay at origin if nothing on same chain has coords

            init[i_b,i_T] += offset_CA.reshape(L,1,3)
    #
    xyz = torch.where(mask.view(B, T, L, 1, 1), init, xyz_t)
    return xyz
