import argparse
import os
import sys
import csv
from dateutil import parser as dateparser

import numpy as np
import torch
import pandas as pd

import pickle
import ast

#from parser import *
from curation_utils import *

# the interface cutoff distance for protein/ligand interactions
MININTDIST = 4.0
# the minimum number of contacts for protein/NA interactions
NCONTACT = 7

# from rohith
biolip_list = ['NUC','ZN','CA','MG','III','MN','FE','CU','SF4','FE2','CO','FES','GOL','NA',  
            'CL','K','CU1','GOL','XE','NO2','EDO','NI','BR','CD','O','CS','NO','TL','HG','UNL','KR',
            'SR','RB','F','AG','AR','U','AU','MO','SE','GD','YB','VX','SM','LI','RE','N','W','OS','HO','PI']
LA_list = ['EDO','PG4','SO4','HEZ','FEO','CL','DMS','ACT','MPD','GOL' ,'NH2','CUA','SIW','PGW','IOD','BR','3NI','ZRW','78M','UNX','nan']
rohith = ['MES','CCN','PO4']
metals = ['LA','NI','3CO','K','CR','ZN','CD','PD','TB','YT3','OS','EU','NA','RB','W','YB','HO3',
          'CE','MN','TL','LI','MN3','AU3','AU','EU3','AL','3NI','FE2','PT','FE','CA','AG','CU1',
          'LU','HG','CO','SR','MG','PB','CS','GA','BA','SM','SB','CU','MO','CU2']
exclude = set(biolip_list+LA_list+rohith)-set(metals)

# input/output files
PB_ORIG = "posebusters_benchmark.csv"
PB_OUT = "posebusters_clean.csv"

def contact_test(prot,lig):
    ''' 
    prot: natoms x 3
    lig: natoms x 3
    '''  
    dists = torch.linalg.norm(prot[:,None,:] - lig[None,:,:], axis=-1)
    mindists,_ = torch.min( dists, axis=-1 )

    return (mindists<MININTDIST).sum()


def get_contacting_ligands(chain,ligands):
    assert chain.type in ['polypeptide(L)', 'polydeoxyribonucleotide', 'polyribonucleotide']
    xyz, mask, seq, chid, resi, unrec_elements = chain_to_xyz(chain)


if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument("-i", default=1, type=int, required=False, help="Parallelize i of j [1]")
    #parser.add_argument("-j", default=1, type=int, required=False, help="Parallelize i of j [1]")
    #args = parser.parse_args()

    orig = pd.read_csv(PB_ORIG)
    ids = list(orig['PDBID'])
    fout = open(PB_OUT, "w")

    records = []
    for ciftag in ids:
        tgt_chn = orig[orig['PDBID']==ciftag]['CHAINID'].item().split('_')[1]

        pklfile = format("/projects/ml/RF2_allatom/rcsb/pkl/%s/%s.pkl.gz"%(ciftag[1:3], ciftag) )
        out = pickle.load(gzip.open(pklfile))
        if len(out) == 4:
            chains, asmb, covale, modres = out
        elif len(out) == 5:
            chains, asmb, covale, meta, modres = out

        xyz, mask, seq, _, _, _ = chain_to_xyz(chains[tgt_chn])
        ligands, lig_covale = get_ligands(chains, covale)

        ligands_prune = []
        for l in ligands:
            for anum,asmb_i in asmb.items():
                chids_in_asmb = set([x[0] for x in asmb_i])
                chids_in_l = set([x[0] for x in l])
                if (not tgt_chn in chids_in_asmb):
                    continue
                if (not chids_in_l.issubset(chids_in_asmb)):
                    continue
                ligs_in_l = set([x[2] for x in l])
                if (ligs_in_l.intersection(exclude) != set()):
                    continue
                lig_xyz, lig_mask, lig_seq, lig_chid, lig_resi, lig_i_ch_xf = get_ligand_xyz(
                    chains, asmb_i, l)

                ncontacts = contact_test(xyz[mask], lig_xyz[lig_mask])

                if ncontacts>=NCONTACT:
                    ligands_prune.append((l, lig_i_ch_xf, int(ncontacts), MININTDIST, 'nonpoly'))

        # final processing
        # 1 add polymers
        final_partner_list = []
        part_orig = orig.loc[orig['PDBID']==ciftag,'PARTNERS'].apply(lambda x: ast.literal_eval(x)).item()
        for p_i in part_orig:
            if (p_i[-1] != 'nonpoly'):
                final_partner_list.append(p_i)

        # 2 add ligand partners
        lig_orig = orig.loc[orig['PDBID']==ciftag,'LIGAND'].apply(lambda x: ast.literal_eval(x)).item()

        for p_i in ligands_prune:
            if p_i[0] != lig_orig:
                final_partner_list.append(p_i)

        print ('b',orig[orig['PDBID']==ciftag]['PARTNERS'].apply(lambda x: ast.literal_eval(x)).item())
        orig.loc[orig['PDBID']==ciftag,'PARTNERS'] = str(final_partner_list)
        print ('a',orig[orig['PDBID']==ciftag]['PARTNERS'].apply(lambda x: ast.literal_eval(x)).item())

        #print (ciftag, part_orig, final_partner_list)

    orig.to_csv(PB_OUT)


