import os, sys, pickle, gzip, argparse, time, glob
import pandas as pd
from curation_utils import *
sys.path.insert(0,'/home/jue/git/rf2a-fd3/')
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

df = pd.read_csv('sm_compl_all_chainid_expanded.csv')
df['PDBID'] = df['CHAINID'].apply(lambda x: x.split('_')[0])
df['HASH2'] = df['HASH2'].apply(lambda x: f'{int(x):06d}' if pd.notnull(x) else np.nan)
df = df.drop_duplicates(['PDBID','CHAINID2','HASH2'])
pdbids = df['PDBID'].drop_duplicates()
params = set_data_loader_params(Namespace())

records = []
t0 = time.time()
ct = 0

records = []
for pdb_id in pdbids[args.istart:args.istart+args.num]:
    pdb_fn = params['MOL_DIR']+f'/{pdb_id[1:3]}/{pdb_id}.pkl.gz'
    chains, asmb, covale, modres = pickle.loads(gzip.open(pdb_fn.strip(), "rb").read())

    tmp = df[df['PDBID']==pdb_id]
    for i,item in tmp.iterrows():
        pdb_chain, pdb_hash = item['CHAINID2'], item['HASH2']
        pdb_id, i_ch_prot = pdb_chain.split('_')

        # transform doesn't actually matter but we need it for featurizing coords
        i_a = str(item['ASSEMBLY'])
        asmb_xfs = asmb[i_a]
        for ch_xf in asmb_xfs:
            if ch_xf[0] == i_ch_prot:
                break

        # load coords
        ch = chains[i_ch_prot]
        xyz_prot, mask_prot, seq_prot, chid_prot, resi_prot = cif_prot_to_xyz(ch, ch_xf, modres)
        protein_L, nprotatoms, _ = xyz_prot.shape

        # load msa
        if type(pdb_hash) is not str and np.isnan(pdb_hash):
            item['PROT_LEN'] = 0
        else:
            a3mA = get_msa(params['PDB_DIR'] + '/a3m/'+pdb_hash[:3] + '/'+ pdb_hash + '.a3m.gz', pdb_hash)
            item['PROT_LEN'] = xyz_prot.shape[0]
            item['MATCHED'] = a3mA['msa'].shape[1] == item['PROT_LEN']
        records.append(item)

df = pd.DataFrame.from_records(records)

os.makedirs(args.outdir, exist_ok=True)
df.to_csv(args.outdir+f'bad_msas{args.istart}.csv')
