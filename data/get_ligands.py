import os, sys, pickle, gzip, argparse, time, glob
import pandas as pd
sys.path.insert(0,'/home/jue/git/chemnet/arch.22-10-28/')
sys.path.insert(0,'/home/jue/git/chemnet/arch.22-10-28/pdb/')
import cifutils

parser = argparse.ArgumentParser()
parser.add_argument("-istart", type=int)
parser.add_argument("-num", type=int)
parser.add_argument("-outdir", default='results_get_ligands/')
args = parser.parse_args()

def get_bonded_partners(lig_tuple, bonds):

    partners = set()
    new_bonds = []
    for bond in bonds:
        if bond.a[:3]==lig_tuple:
            partners.add(bond.b[:3])
        elif bond.b[:3]==lig_tuple:
            partners.add(bond.a[:3])
        else:
            new_bonds.append(bond)

    partners = set([p for p in partners if p!=lig_tuple])

    new_partners = []
    for p in partners:
        new_partners.append(get_bonded_partners(p, new_bonds))

    for new_p in new_partners:
        partners.update(new_p)

    return partners


records = []
ct = 0
start_time = time.time()

#filenames = sorted(glob.glob('/projects/ml/RF2_allatom/rcsb/pkl/*/*.pkl.gz'))
filenames = [line.strip() for line in open('all_rcsb_cif.txt').readlines()]
filenames = [fn.replace('/databases/rcsb/cif/','/projects/ml/RF2_allatom/rcsb/pkl/').replace('.cif.gz','.pkl.gz') for fn in filenames]

for fn in filenames[args.istart:args.istart+args.num]:

    pdbid = os.path.basename(fn).replace('.pkl.gz','')
    chains, asmb, covale, modres = pickle.load(gzip.open(fn))

    # collect all ligand residues and potential inter-ligand bonds
    lig_res_s = list(set([x[:3] for ch in chains.values() if ch.type=='nonpoly'
                                for x in ch.atoms if x[2]!='HOH']))
    bonds = []
    for i_ch,ch in chains.items():
        if ch.type=='nonpoly':
            bonds.extend(ch.bonds)
    inter_ligand_bonds = []
    prot_lig_bonds = []
    for bond in covale:
        if chains[bond.a[0]].type=='nonpoly' and chains[bond.b[0]].type=='nonpoly':
            bonds.append(bond)
            inter_ligand_bonds.append(bond)
        if sorted([chains[bond.a[0]].type, chains[bond.b[0]].type])==['nonpoly','polypeptide(L)']:
            prot_lig_bonds.append(bond)

    # make list of bonded ligands (lists of ligand residues)
    ligands = []
    while len(lig_res_s)>0:
        res = lig_res_s[0]
        lig = get_bonded_partners(res, bonds)
        lig.add(res)
        lig = sorted(list(lig))
        lig_res_s = [res for res in lig_res_s if res not in lig]
        ligands.append(lig)

    for lig in ligands:
        records.append(dict(
            PDBID=pdbid,
            LIGAND=lig,
            COVALENT=[(bond.a, bond.b) for bond in prot_lig_bonds],
        ))

    ct += 1
    if ct % 100 == 0:
        print(ct, time.time()-start_time)

df = pd.DataFrame.from_records(records)

os.makedirs(args.outdir, exist_ok=True)
df.to_csv(args.outdir+f'ligands{args.istart}.csv')
