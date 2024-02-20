from itertools import permutations
import numpy as np
import torch
from icecream import ic
from openbabel import openbabel
from rf2aa.chemical import ChemicalData as ChemData

PARAMS = {
    'DMIN':1, 
    'DMID':4, 
    'DMAX':20.0, 
    'DBINS1':30, 
    'DBINS2':30,
    'ABINS':36,
    'USE_CB':False
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

def generate_Cbeta(N,Ca,C):
    # recreate Cb given N,Ca,C
    b = Ca - N
    c = C - Ca
    a = torch.cross(b, c, dim=-1)
    #Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca
    # fd: below matches sidechain generator (=Rosetta params)
    Cb = -0.57910144*a + 0.5689693*b - 0.5441217*c + Ca

    return Cb
    
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

    if params['USE_CB']:
        dist = get_pair_dist(Cb,Cb)
    else:
        dist = get_pair_dist(Ca,Ca)

    dist[torch.isnan(dist)] = 999.9
    c6d[...,0] = dist + 999.9*torch.eye(nres,device=xyz.device)[None,...]
    b,i,j = torch.where(c6d[...,0]<params['DMAX'])

    c6d[b,i,j,torch.full_like(b,1)] = get_dih(Ca[b,i], Cb[b,i], Cb[b,j], Ca[b,j])
    c6d[b,i,j,torch.full_like(b,2)] = get_dih(N[b,i], Ca[b,i], Cb[b,i], Cb[b,j])
    c6d[b,i,j,torch.full_like(b,3)] = get_ang(Ca[b,i], Cb[b,i], Cb[b,j])

    # fix long-range distances
    c6d[...,0][c6d[...,0]>=params['DMAX']] = 999.9
    c6d = torch.nan_to_num(c6d)
    
    return c6d
    
def xyz_to_t2d(xyz_t, mask, params=PARAMS):
    """convert template cartesian coordinates into 2d distance 
    and orientation maps
    
    Parameters
    ----------
    xyz_t : pytorch tensor of shape [batch,templ,nres,3,3]
            stores Cartesian coordinates of template backbone N,Ca,C atoms
    mask :  pytorch tensor [batch,templ,nres,nres]
            indicates whether valid residue pairs or not
    Returns
    -------
    t2d : pytorch tensor of shape [batch,nres,nres,37+6+3]
          stores stacked dist,omega,theta,phi 2D maps 
    """
    B, T, L = xyz_t.shape[:3]
    c6d = xyz_to_c6d(xyz_t[:,:,:,:3].view(B*T,L,3,3), params=params)
    c6d = c6d.view(B, T, L, L, 4)

    # dist to one-hot encoded
    mask = mask[...,None]
    dist = dist_to_onehot(c6d[...,0], params)*mask
    orien = torch.cat((torch.sin(c6d[...,1:]), torch.cos(c6d[...,1:])), dim=-1)*mask # (B, T, L, L, 6)
    #
    t2d = torch.cat((dist, orien, mask), dim=-1)
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
    db = dist_to_bins(dist, params)
    dist = torch.nn.functional.one_hot(db, num_classes=params['DBINS1'] + params['DBINS2']+1).float()
    return dist

# ============================================================
def dist_to_bins(dist,params=PARAMS):
    """bin 2d distance maps
    """
    dist[torch.isnan(dist)] = 999.9
    dstep1 = (params['DMID'] - params['DMIN']) / params['DBINS1']
    dstep2 = (params['DMAX'] - params['DMID']) / params['DBINS2']
    dbins = torch.cat([
        torch.linspace(params['DMIN']+dstep1, params['DMID'], params['DBINS1'], 
                       dtype=dist.dtype,device=dist.device),
        torch.linspace(params['DMID']+dstep2, params['DMAX'], params['DBINS2'], 
                       dtype=dist.dtype,device=dist.device),
    ])
    db = torch.bucketize(dist.contiguous(),dbins).long()

    return db

# ============================================================
def c6d_to_bins(c6d, same_chain, negative=False, params=PARAMS):
    """bin 2d distance and orientation maps
    """

    db = dist_to_bins(c6d[...,0], params) # all dist < DMIN are in bin 0

    astep = 2.0*np.pi / params['ABINS']
    ob = torch.round((c6d[...,1]+np.pi-astep/2)/astep)
    tb = torch.round((c6d[...,2]+np.pi-astep/2)/astep)
    pb = torch.round((c6d[...,3]-astep/2)/astep)

    # synchronize no-contact bins
    params['DBINS'] = params['DBINS1'] + params['DBINS2']
    ob[db==params['DBINS']] = params['ABINS']
    tb[db==params['DBINS']] = params['ABINS']
    pb[db==params['DBINS']] = params['ABINS']//2

    if negative:
        db = torch.where(same_chain.bool(), db.long(), params['DBINS'])
        ob = torch.where(same_chain.bool(), ob.long(), params['ABINS'])
        tb = torch.where(same_chain.bool(), tb.long(), params['ABINS'])
        pb = torch.where(same_chain.bool(), pb.long(), params['ABINS']//2)
    
    return torch.stack([db,ob,tb,pb],axis=-1).long()

def standardize_dihedral_retain_first(a,b,c,d):
    isomorphisms = [(a,b,c,d), (a,c,b,d)]
    return sorted(isomorphisms)[0]

def get_chirals(obmol, xyz):
    '''
    get all quadruples of atoms forming chiral centers and the expected ideal pseudodihedral between them
    '''
    stereo = openbabel.OBStereoFacade(obmol)
    angle = np.arcsin(1/3**0.5)
    chiral_idx_set = set()
    for i in range(obmol.NumAtoms()):
        if not stereo.HasTetrahedralStereo(i):
            continue
        si = stereo.GetTetrahedralStereo(i)
        config = si.GetConfig()

        o = config.center
        c = config.from_or_towards
        i,j,k = list(config.refs)
        for a, b, c in permutations((c,i,j,k), 3):
            chiral_idx_set.add(standardize_dihedral_retain_first(o,a,b,c))

    chiral_idx = list(chiral_idx_set)
    chiral_idx.sort()
    chiral_idx = torch.tensor(chiral_idx, dtype=torch.float32)
    chiral_idx = chiral_idx[(chiral_idx<obmol.NumAtoms()).all(dim=-1)]

    if chiral_idx.numel() == 0:
        return torch.zeros((0,5))

    dih = get_dih(*xyz[chiral_idx.long()].split(split_size=1,dim=1))[:,0]
    chirals = torch.nn.functional.pad(chiral_idx, (0, 1), mode='constant', value=angle)
    chirals[dih<0.0,-1] *= -1
    return chirals

def get_atomize_protein_chirals(residues_atomize, lig_xyz, residue_atomize_mask, bond_feats):
    """
    Enumerate chiral centers in residues and provide features for chiral centers
    """
    angle = np.arcsin(1/3**0.5) # perfect tetrahedral geometry
    chiral_atoms = ChemData().aachirals[residues_atomize]
    ra = residue_atomize_mask.nonzero()
    r,a = ra.T

    chiral_atoms = chiral_atoms[r,a].nonzero().squeeze(1) #num_chiral_centers
    num_chiral_centers = chiral_atoms.shape[0]
    chiral_bonds = bond_feats[chiral_atoms] # find bonds to each chiral atom
    chiral_bonds_idx = chiral_bonds.nonzero() # find indices of each bonded neighbor to chiral atom
    # in practice all chiral atoms in proteins have 3 heavy atom neighbors, so reshape to 3 
    chiral_bonds_idx = chiral_bonds_idx.reshape(num_chiral_centers, 3, 2)
    
    chirals = torch.zeros((num_chiral_centers, 5))
    chirals[:,0] = chiral_atoms.long()
    chirals[:, 1:-1] = chiral_bonds_idx[...,-1].long()
    chirals[:, -1] = angle
    n = chirals.shape[0]
    if n>0:
        chirals = chirals.repeat(3,1).float()
        chirals[n:2*n,1:-1] = torch.roll(chirals[n:2*n,1:-1],1,1)
        chirals[2*n: ,1:-1] = torch.roll(chirals[2*n: ,1:-1],2,1)
        dih = get_dih(*lig_xyz[chirals[:,:4].long()].split(split_size=1,dim=1))[:,0]
        chirals[dih<0.0,-1] = -angle
    else:
        chirals = torch.zeros((0,5))
    return chirals
