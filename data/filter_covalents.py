import os, sys, pickle, gzip, argparse, time, glob
import pandas as pd
from curation_utils import *

sys.path.insert(0,'/home/jue/git/rf2a-fd3-ph3/')
sys.path.insert(0,'/home/jue/git/rf2a-fd3-ph3/rf2aa/')
from util import cif_ligand_to_xyz, get_ligand_atoms_bonds

parser = argparse.ArgumentParser()
parser.add_argument("-istart", type=int)
parser.add_argument("-num", type=int)
parser.add_argument("-outdir", default='results_filter_covalents/')
args = parser.parse_args()

def is_weird(covalents):
    """Detects non-biological bonds"""
    has_oxygen_oxygen_bond = any([a1[3][0]=='O' and a2[3][0]=='O' for (a1,a2) in covalents])
    has_fluorine_fluorine_bond = any([a1[3][0]=='F' and a2[3][0]=='F' for (a1,a2) in covalents])
    is_oxy_hydroxy = any([a1[2]=='O' or a2[2]=='O' or \
                          a1[2]=='OH' or a2[2]=='OH' or \
                          a1[2]=='HOH' or a2[2]=='HOH' for (a1,a2) in covalents])
    return has_oxygen_oxygen_bond or has_fluorine_fluorine_bond or is_oxy_hydroxy

def is_clashing(qlig_xyz_valid, xyz_chains, mask_chains, asmb_xforms):
    """Detects if a ligand is within 1A of any protein in its assembly."""
    for xyz, mask in zip(xyz_chains, mask_chains):
        for i_xf, (xf_ch, xf) in enumerate(asmb_xforms):
            if xf_ch != ch.id: continue
            xf = torch.tensor(xf).float()
            u,r = xf[:3,:3], xf[:3,3]
            xyz_xf = torch.einsum('ij,raj->rai', u, xyz) + r[None,None]

            atom_xyz = xyz_xf[:,:NHEAVY][mask[:,:NHEAVY].numpy(),:]
            dist = torch.cdist(qlig_xyz_valid, atom_xyz, compute_mode='donot_use_mm_for_euclid_dist')
            if (dist<1).any():
                return True
    return False

records = []
ct = 0
t0 = time.time()

#df = pd.read_csv('sm_compl_all_goodmsa_filt_atomcounts_partnersfilt_20230126.csv')
df = pd.read_csv(f'input{args.istart}.csv')

for col in ['LIGAND','LIGXF','COVALENT','PARTNERS']:
    if col in df:
        df[col] = df[col].apply(lambda x: eval(x))

pdbid_prev = None
i_a_prev = None
#for i,row in df.iloc[args.istart:args.istart+args.num].iterrows():
for i,row in df.iterrows():
    pdbid = row['CHAINID'].split('_')[0]
    if pdbid!=pdbid_prev:
        chains,asmb,covale,modres = pickle.load(gzip.open(f'/projects/ml/RF2_allatom/rcsb/pkl/{pdbid[1:3]}/{pdbid}.pkl.gz'))

    # remove covalent examples with non-biological bonds
    if len(row['COVALENT'])>0:
        if is_weird(row['COVALENT']):
            print('non-biological protein-ligand bond',row['CHAINID'],row['LIGAND'])
            continue

    asmb_xforms = asmb[str(row['ASSEMBLY'])]
    asmb_xform_chids = [x[0] for x in asmb_xforms]
    asmb_chains = [chains[i_ch] for i_ch in set(asmb_xform_chids)]
    
    if pdbid!=pdbid_prev or i_a!=i_a_prev:
        # featurize all protein chains
        xyz_chains, mask_chains = [], []
        for ch in asmb_chains:
            if ch.type != 'polypeptide(L)': continue
            xyz, mask, seq, chid, resi, unrec_elements = chain_to_xyz(ch)
            xyz_chains.append(xyz)
            mask_chains.append(mask)

    pdbid_prev = pdbid
    i_a_prev = i_a

    # remove examples with clashes between query ligand and protein chains
    qlig_xyz, qlig_mask, qlig_seq, qlig_chid, qlig_resi, qlig_chxf = \
        get_ligand_xyz(asmb_chains, asmb_xforms, row['LIGAND'],
                       seed_ixf=dict(row['LIGXF'])[row['LIGAND'][0][0]])

    qlig_xyz_valid = qlig_xyz[qlig_mask]
    clash = is_clashing(qlig_xyz_valid, xyz_chains, mask_chains, asmb_xforms)
    if clash:
        print('clash between query ligand and protein',row['CHAINID'], row['LIGAND'])
        continue
    
    # filter partners
    new_partners = []
    for p in row['PARTNERS']:
        if p[-1]=='nonpoly':
            # remove covalent partners with non-biological bonds
            bonds = []
            for bond in covale:
                if any([bond.a[:3]==res or bond.b[:3]==res for res in p[0]]):
                    bonds.append((bond.a, bond.b))
            if len(bonds)>0:
                if is_weird(bonds):
                    print('non-biological protein-ligand bond in partner',row['CHAINID'],p)
                    continue
            
            # remove partners with clash to protein
            lig_xyz, lig_mask, lig_seq, lig_chid, lig_resi, lig_chxf = \
                get_ligand_xyz(asmb_chains, asmb_xforms, p[0],
                               seed_ixf=dict(p[1])[p[0][0][0]])
            lig_xyz_valid = lig_xyz[lig_mask]
            clash = is_clashing(lig_xyz_valid, xyz_chains, mask_chains, asmb_xforms)
            if clash:
                print('clash between partner ligand and protein',row['CHAINID'],p)
                continue
        new_partners.append(p)
    
    row['PARTNERS'] = new_partners
    records.append(row)

    ct += 1
    if ct % 50 == 0:
        print(ct, time.time() -t0)

df = pd.DataFrame.from_records(records)

os.makedirs(args.outdir, exist_ok=True)
df.to_csv(args.outdir+f'filtered_covalents{args.istart}.csv')
