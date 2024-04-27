import os, sys, pickle, gzip, argparse
sys.path.insert(0,'/home/jue/git/chemnet/arch.22-10-28/')
sys.path.insert(0,'/home/jue/git/chemnet/arch.22-10-28/pdb/')
import cifutils


parser = argparse.ArgumentParser()
parser.add_argument("-istart", type=int)
parser.add_argument("-num", type=int)
args = parser.parse_args()

Parser = cifutils.CIFParser()

filenames = [line.strip() for line in open('all_rcsb_cif.txt').readlines()]
for fn in filenames[args.istart:args.istart+args.num]:
    outfn = fn.replace('/databases/rcsb/cif/','/projects/ml/RF2_allatom/rcsb/pkl_v2/')
    outfn = outfn.replace('.cif.gz','.pkl.gz')
    #if os.path.exists(outfn): continue

    print(fn)
    try:
        chains,asmb,covale,modres = Parser.parse(fn)
        outdir = os.path.dirname(outfn)
        os.makedirs(outdir, exist_ok=True)
        pickle.dump((chains,asmb,covale,modres),gzip.open(outfn,'wb'))
        print('Saved',outfn)
    except Exception as e:
        print('Failed to parse',fn)
