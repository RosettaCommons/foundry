import os, sys, pickle, gzip, argparse, time, glob
import pandas as pd
from curation_utils import *

parser = argparse.ArgumentParser()
parser.add_argument("-istart", type=int)
parser.add_argument("-num", type=int)
parser.add_argument("-outdir", default='results_get_partners/')
args = parser.parse_args()

records = []
ct = 0
start_time = time.time()

df = pd.read_csv('ligands_filt.csv')
df['LIGAND'] = df['LIGAND'].apply(lambda x: eval(x))

records = []
start_time = time.time()
pdbid_prev = None

for i,row in df.iloc[args.istart:args.istart+args.num].iterrows():
    pdbid = row['PDBID']
    if pdbid!=pdbid_prev:
        chains,asmb,covale,modres = pickle.load(gzip.open(f'/projects/ml/RF2_allatom/rcsb/pkl/{pdbid[1:3]}/{pdbid}.pkl.gz'))
        ligands = get_ligands(chains, covale)
    pdbid_prev = pdbid

    query_ligand = row['LIGAND']

    # all assemblies containing this ligand
    ligand_chids = set([res[0] for res in query_ligand])
    ligand_assemblies = [i_a for i_a in asmb
                         if ligand_chids.issubset(set([x[0] for x in asmb[i_a]]))]

    for i_a in ligand_assemblies:
        asmb_xforms = asmb[i_a]
        asmb_xform_chids = [x[0] for x in asmb_xforms]
        asmb_chains = [chains[i_ch] for i_ch in set(asmb_xform_chids)]

        # assembly must have at least one protein and ligand chain
        if not {'polypeptide(L)','nonpoly'}.issubset(set([ch.type for ch in asmb_chains])):
            continue

        # get query ligand coordinates (one copy of it in this assembly)
        qlig_xyz, qlig_mask, qlig_seq, qlig_chid, qlig_resi, qlig_chxf = \
            get_ligand_xyz(asmb_chains, asmb_xforms, query_ligand)

        if qlig_xyz.numel()==0: continue

        qlig_xyz_valid = qlig_xyz[qlig_mask[:,1],1]
        if qlig_xyz_valid.numel()==0: continue

        # get contacts between this ligand and all transformed protein chains in assembly
        prot_na_contacts = get_contacting_chains(asmb_chains, asmb_xforms,
                                         qlig_xyz_valid, qlig_chxf)

        lig_contacts = get_contacting_ligands(ligands, asmb_chains, asmb_xforms,
                                              query_ligand, qlig_xyz_valid, qlig_chxf)

        prot_contacts = [x for x in prot_na_contacts if x[-1]=='polypeptide(L)' and x[2]>0]
        if len(prot_contacts) == 0:
            continue # no protein near ligand, don't use for training

        # pool all partners, sort from most to least contacts (then lowest to highest min distance)
        contacts = sorted(prot_na_contacts+lig_contacts, key=lambda x: (x[2], -x[3]), reverse=True)

        # save results
        new_row = row.copy()
        new_row['ASSEMBLY'] = i_a
        new_row['PROT_CHAIN'] = prot_contacts[0][0] # most-contacting protein chain
        new_row['LIGXF'] = qlig_chxf
        new_row['PARTNERS'] = contacts
        records.append(new_row)

    if i%50 == 0:
        print(i, time.time()-start_time)

df = pd.DataFrame.from_records(records)

os.makedirs(args.outdir, exist_ok=True)
df.to_csv(args.outdir+f'ligands_partners{args.istart}.csv')
