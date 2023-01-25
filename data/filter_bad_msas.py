import os, sys, pickle, gzip, argparse, time, glob
import pandas as pd
from curation_utils import *

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,script_dir+'/../')
sys.path.insert(0,script_dir+'/../rf2aa/')
from data_loader import *
from argparse import Namespace

parser = argparse.ArgumentParser()
parser.add_argument("-istart", type=int)
parser.add_argument("-num", type=int)
parser.add_argument("-outdir", default='results_filter_msas/')
args = parser.parse_args()

records = []
ct = 0
start_time = time.time()

df = pd.read_csv('sm_compl_all_chainid_expanded_20230220.csv')
df['PDBID'] = df['CHAINID'].apply(lambda x: x.split('_')[0])
df['HASH2'] = df['HASH2'].apply(lambda x: f'{int(x):06d}' if pd.notnull(x) else np.nan)
df = df.drop_duplicates(['PDBID','CHAINID2','HASH2'])
pdbids = df['PDBID'].drop_duplicates()
params = set_data_loader_params(Namespace())

records = []
t0 = time.time()
ct = 0
records = []
pdbid_prev = None
for i,item in df.iloc[args.istart:args.istart+args.num].iterrows():

    pdb_chain, pdb_hash = item['CHAINID2'], item['HASH2']
    pdb_id, i_ch_prot = pdb_chain.split('_')

    if pdb_id!=pdbid_prev:
        chains,asmb,covale,modres = pickle.load(gzip.open(f'/projects/ml/RF2_allatom/rcsb/pkl/{pdb_id[1:3]}/{pdb_id}.pkl.gz'))
        ligands, lig_covale = get_ligands(chains, covale)

    # transform doesn't actually matter but we need it for featurizing coords
    i_a = str(item['ASSEMBLY'])
    asmb_xfs = asmb[i_a]
    for ch_xf in asmb_xfs:
        if ch_xf[0] == i_ch_prot:
            break

    # load coords
    ch = chains[i_ch_prot]
    xyz_prot, mask_prot, seq_prot, chid_prot, resi_prot, _ = cif_prot_to_xyz(ch, ch_xf, modres)
    protein_L, nprotatoms, _ = xyz_prot.shape

    # load msa
    if type(pdb_hash) is not str and np.isnan(pdb_hash):
        item['PROT_LEN'] = 0
    else:
        a3mA = get_msa(params['PDB_DIR'] + '/a3m/'+pdb_hash[:3] + '/'+ pdb_hash + '.a3m.gz', pdb_hash)
        item['PROT_LEN'] = xyz_prot.shape[0]
        item['MATCHED'] = a3mA['msa'].shape[1] == item['PROT_LEN']
    records.append(item)

    pdbid_prev = pdb_id

    ct += 1

    if ct % 50 == 0:
        print(ct, time.time() - t0)

df = pd.DataFrame.from_records(records)

os.makedirs(args.outdir, exist_ok=True)
df.to_csv(args.outdir+f'bad_msas{args.istart}.csv')
