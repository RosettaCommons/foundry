import os, sys, pickle, gzip, argparse, time, glob
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,script_dir+'/../')
sys.path.insert(0,script_dir+'/../rf2aa/')

from curation_utils import *

parser = argparse.ArgumentParser()
parser.add_argument("-istart", type=int)
parser.add_argument("-num", type=int)
parser.add_argument("-outdir", default='results_get_ligands/')
args = parser.parse_args()

records = []
ct = 0
start_time = time.time()

filenames = [line.strip() for line in open('all_rcsb_cif.txt').readlines()]
filenames = [fn.replace('/databases/rcsb/cif/','/projects/ml/RF2_allatom/rcsb/pkl/').replace('.cif.gz','.pkl.gz') for fn in filenames]

for fn in filenames[args.istart:args.istart+args.num]:
    try:
        pdbid = os.path.basename(fn).replace('.pkl.gz','')
        chains, asmb, covale, modres = pickle.load(gzip.open(fn))

        ligands, lig_covale = get_ligands(chains, covale)

        # add ligands in order of assembly
        # this will make loading & filtering later more efficient
        for i_a in asmb:
            for lig, lcovale in zip(ligands,lig_covale):
                ligand_chids = set([res[0] for res in lig])

                if not ligand_chids.issubset(set([x[0] for x in asmb[i_a]])):
                    continue

                records.append(dict(
                    PDBID=pdbid,
                    LIGAND=lig,
                    ASSEMBLY=i_a,
                    COVALENT=lcovale,
                ))
    except Exception as e:
        print(e)

    ct += 1
    if ct % 100 == 0:
        print(ct, time.time()-start_time)

df = pd.DataFrame.from_records(records)

os.makedirs(args.outdir, exist_ok=True)
df.to_csv(args.outdir+f'ligands{args.istart}.csv')
