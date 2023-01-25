import os, sys, pickle, gzip, argparse, time, glob
import pandas as pd
from curation_utils import *

parser = argparse.ArgumentParser()
parser.add_argument("-istart", type=int)
parser.add_argument("-num", type=int)
parser.add_argument("-outdir", default='results_get_partners/')
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

def filter_partners(partners, chains, xyz_chains, mask_chains, asmb_xforms, covale, row):
    new_partners = []
    for p in partners:
        if p[-1]=='nonpoly':
            # remove covalent partners with non-biological bonds
            bonds = []
            for bond in covale:
                if any([bond.a[:3]==res or bond.b[:3]==res for res in p[0]]):
                    bonds.append((bond.a, bond.b))
            if len(bonds)>0:
                if is_weird(bonds):
                    print('non-biological protein-ligand bond in partner',row['PDBID'],p)
                    continue
            # remove partners with clash to protein
            plig = p[0]
            lig_xyz, lig_mask, lig_seq, lig_chid, lig_resi, lig_chxf = \
                get_ligand_xyz(chains, asmb_xforms, plig,
                               seed_ixf=dict(p[1])[plig[0][0]])

            lig_xyz_valid = lig_xyz[lig_mask]
            clash = is_clashing(lig_xyz_valid, xyz_chains, mask_chains, asmb_xforms)
            if clash:
                print('clash between partner ligand and protein',row['PDBID'],p)
                continue
        new_partners.append(p)
    return new_partners


records = []
ct = 0
start_time = time.time()

df = pd.read_csv('ligands_filt.csv')
df['LIGAND'] = df['LIGAND'].apply(lambda x: eval(x))
df['COVALENT'] = df['COVALENT'].apply(lambda x: eval(x))

records = []
start_time = time.time()
pdbid_prev = None
i_a_prev = None

for i,row in df.iloc[args.istart:args.istart+args.num].iterrows():

    pdbid = row['PDBID']
    if pdbid!=pdbid_prev:
        chains,asmb,covale,modres = pickle.load(gzip.open(f'/projects/ml/RF2_allatom/rcsb/pkl/{pdbid[1:3]}/{pdbid}.pkl.gz'))
        ligands, lig_covale = get_ligands(chains, covale)

    # remove covalent examples with non-biological bonds
    if len(row['COVALENT'])>0:
        if is_weird(row['COVALENT']):
            print('non-biological protein-ligand bond',row['PDBID'],row['LIGAND'])
            continue

    i_a = str(row['ASSEMBLY'])
    asmb_xforms = asmb[i_a]
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

    # assembly must have at least one protein and ligand chain
    if not {'polypeptide(L)','nonpoly'}.issubset(set([ch.type for ch in asmb_chains])):
        continue

    # get query ligand coordinates (one copy of it in this assembly)
    query_ligand = row['LIGAND']
    qlig_xyz, qlig_mask, qlig_seq, qlig_chid, qlig_resi, qlig_chxf = \
        get_ligand_xyz(chains, asmb_xforms, query_ligand)

    # filter out query ligand residues that weren't loaded (all atoms have 0 occupancy)
    query_ligand = [res for res in query_ligand if res[0] in [x[0] for x in qlig_chxf]]
    row['COVALENT'] = [(a,b) for (a,b) in row['COVALENT'] if any([a[:3]==res or b[:3]==res for res in query_ligand])]

    # remove empty ligands
    if qlig_xyz.numel()==0: continue
    qlig_xyz_valid = qlig_xyz[qlig_mask[:,1],1]
    if qlig_xyz_valid.numel()==0: continue

    # remove examples with clashes between query ligand and protein chains
    clash = is_clashing(qlig_xyz_valid, xyz_chains, mask_chains, asmb_xforms)
    if clash:
        print('clash between query ligand and protein',row['PDBID'], row['LIGAND'], row['ASSEMBLY'])
        continue

    # get list of all coordinate-transformed protein chains in assembly that contact this ligand
    prot_na_contacts = get_contacting_chains(asmb_chains, asmb_xforms,
                                     qlig_xyz_valid, qlig_chxf)

    # get list of all coordinate-transformed ligands in assembly that contact this ligand
    lig_contacts = get_contacting_ligands(ligands, chains, asmb_xforms,
                                          query_ligand, qlig_xyz_valid, qlig_chxf)

    prot_contacts = [x for x in prot_na_contacts if x[-1]=='polypeptide(L)' and x[2]>0]
    if len(prot_contacts) == 0:
        continue # no protein <5A of ligand, don't use for training

    # pool all contacting objects, sort from most to least contacts (then lowest to highest min distance)
    partners = sorted(prot_na_contacts+lig_contacts, key=lambda x: (x[2], -x[3]), reverse=True)

    # filter partners
    new_partners = filter_partners(partners, chains, xyz_chains, mask_chains, asmb_xforms, covale, row)

    # save results
    new_row = row.copy()
    new_row['LIGAND'] = query_ligand # update in case we removed some residues with no resolved atoms
    new_row['ASSEMBLY'] = i_a
    new_row['PROT_CHAIN'] = prot_contacts[0][0] # most-contacting protein chain
    new_row['LIGXF'] = qlig_chxf
    new_row['PARTNERS'] = new_partners
    new_row['LIGATOMS'] = qlig_xyz.shape[0]
    new_row['LIGATOMS_RESOLVED'] = qlig_mask.sum().item()

    records.append(new_row)

    if i%50 == 0:
        print(i, time.time()-start_time)

df = pd.DataFrame.from_records(records)

os.makedirs(args.outdir, exist_ok=True)
df.to_csv(args.outdir+f'ligands_partners{args.istart}.csv')
