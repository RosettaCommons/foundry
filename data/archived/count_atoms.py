import os, sys, pickle, gzip, argparse, time, glob
import pandas as pd
from curation_utils import *

sys.path.insert(0,'/home/jue/git/rf2a-fd3/')
sys.path.insert(0,'/home/jue/git/rf2a-fd3/rf2aa/')
from util import cif_ligand_to_xyz, get_ligand_atoms_bonds

parser = argparse.ArgumentParser()
parser.add_argument("-istart", type=int)
parser.add_argument("-num", type=int)
parser.add_argument("-outdir", default='results_count_atoms/')
args = parser.parse_args()

records = []
ct = 0
t0 = time.time()

#df = pd.read_csv('sm_compl_all_goodmsa_filt_20230117.csv')
df = pd.read_csv('sm_compl_all_goodmsa_filt_20230126.csv')

for col in ['LIGAND','LIGXF']:
    if col in df:
        df[col] = df[col].apply(lambda x: eval(x))

pdbid_prev = None
for i,row in df.iloc[args.istart:args.istart+args.num].iterrows():
    pdbid = row['CHAINID'].split('_')[0]
    if pdbid!=pdbid_prev:
        chains,asmb,covale,modres = pickle.load(gzip.open(f'/projects/ml/RF2_allatom/rcsb/pkl/{pdbid[1:3]}/{pdbid}.pkl.gz'))
    pdbid_prev = pdbid

    # coordinate transforms to recreate this bio-assembly
    i_a = str(row['ASSEMBLY'])
    asmb_xfs = asmb[i_a]

    ligand = row['LIGAND']

    # load query ligand (the "focus ligand" for this training example)
    lig_atoms, lig_bonds = get_ligand_atoms_bonds(ligand, chains, covale)
    lig_ch2xf = dict(row['LIGXF'])

    xyz_sm, mask_sm, msa_sm, chid_sm, lig_akeys = cif_ligand_to_xyz(lig_atoms, asmb_xfs, lig_ch2xf)

    row['LIGATOMS'] = xyz_sm.shape[0]
    row['LIGATOMS_RESOLVED'] = mask_sm.sum().item()
    records.append(row)

    ct += 1
    if ct % 50 == 0:
        print(ct, time.time() -t0)

df = pd.DataFrame.from_records(records)

os.makedirs(args.outdir, exist_ok=True)
df.to_csv(args.outdir+f'atom_counts{args.istart}.csv')
