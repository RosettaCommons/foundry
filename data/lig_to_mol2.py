import os, sys, pickle, gzip, time, argparse
from openbabel import openbabel
import pandas as pd
import numpy as np
import torch
sys.path.insert(0,'/home/jue/git/rf2a-fd3/')
import chemical
from chemical import atom_num, frame_priority2atom

parser = argparse.ArgumentParser()
parser.add_argument("-istart", type=int)
parser.add_argument("-num", type=int)
args = parser.parse_args()

def cif_ligand_to_xyz(atoms, asmb_xfs=None, ch2xf=None):

    elnum_to_atom = dict(zip(atom_num, frame_priority2atom))
    atoms_no_H = {k:v for k,v in atoms.items() if v.element != 1} # exclude hydrogens
    L = len(atoms_no_H)

    xyz = torch.zeros(L, 3)
    mask = torch.zeros(L,).bool()
    seq = torch.full((L,), np.nan)
    chid = ['-']*L
    akeys = [None]*L

    # create coords, atom mask, and seq tokens
    for i,(k,v) in enumerate(atoms_no_H.items()):
        xyz[i, :] = torch.tensor(v.xyz)
        mask[i] = v.occ
        if v.element not in elnum_to_atom:
            print('Element not in alphabet:',v.element)
            seq[i] = chemical.aa2num['ATM']
        else:
            seq[i] = chemical.aa2num[elnum_to_atom[v.element]]
        akeys[i] = k
        chid[i] = k[0]

    if asmb_xfs is not None:
        # apply transforms
        chid = np.array(chid)
        for i_ch in np.unique(chid):
            idx = chid==i_ch
            xf = torch.tensor(asmb_xfs[ch2xf[i_ch]][1]).float()
            u,r = xf[:3,:3], xf[:3,3]
            xyz[idx] = torch.einsum('ij,aj->ai', u, xyz[idx]) + r[None,None]

    return xyz, mask, seq, chid, akeys

def cif_ligand_to_obmol(xyz, akeys, atoms, bonds):

    mol = openbabel.OBMol()
    for i,k in enumerate(akeys):
        a = mol.NewAtom()
        a.SetAtomicNum(atoms[k].element)
        a.SetVector(float(xyz[i,0]), float(xyz[i,1]), float(xyz[i,2]))

    sm_L = len(akeys)
    bond_feats = torch.zeros((sm_L,sm_L))
    for bond in bonds:
        if bond.a not in akeys or bond.b not in akeys: continue # intended to skip bonds to H's
        i = akeys.index(bond.a)
        j = akeys.index(bond.b)
        bond_feats[i,j] = bond.order if not bond.aromatic else 4
        bond_feats[j,i] = bond_feats[i,j]

        obb = openbabel.OBBond()
        obb.SetBegin(mol.GetAtom(i+1))
        obb.SetEnd(mol.GetAtom(j+1))
        obb.SetBondOrder(bond.order)
        if bond.aromatic:
            obb.SetAromatic()

        mol.AddBond(obb)

    return mol, bond_feats

df = pd.read_csv('ligands_for_tanimoto.csv')
tmp = df.iloc[args.istart:args.istart+args.num]
tmp['LIGAND'] = tmp['LIGAND'].apply(lambda x: eval(x))

ct = 0
t0 = time.time()
for i,row in tmp.iterrows():

    pdb_id = row['PDBID']
    ligand = list(row['LIGAND'])

    chains, asmb, covale, modres = pickle.load(gzip.open(f'/projects/ml/RF2_allatom/rcsb/pkl/{pdb_id[1:3]}/{pdb_id}.pkl.gz'))

    lig_atoms = dict()
    lig_bonds = []
    for i_ch,ch in chains.items():
        for k,v in ch.atoms.items():
            if k[:3] in ligand:
                lig_atoms[k] = v
        for bond in ch.bonds:
            if bond.a[:3] in ligand or bond.b[:3] in ligand:
                lig_bonds.append(bond)
    # for bond in covale:
    #     if bond.a[:3] in ligand and bond.b[:3] in ligand:
    #         lig_bonds.append(bond)

    xyz_sm, mask_sm, msa_sm, chid_sm, akeys = cif_ligand_to_xyz(lig_atoms)

    mol, bond_feats_sm = cif_ligand_to_obmol(xyz_sm, akeys, lig_atoms, lig_bonds)

    obConversion = openbabel.OBConversion()
    obConversion.SetOutFormat("mol2")
    outfn = ligand[0][2]+'.mol2'
    outdir = f'mol2/{outfn[0]}/'
    os.makedirs(outdir, exist_ok=True)
    obConversion.WriteFile(mol, outdir+outfn)

    ct += 1

    if ct % 50 == 0:
        print(ct, time.time() - t0, ligand)
