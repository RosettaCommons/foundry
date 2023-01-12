import numpy as np
import scipy
import scipy.spatial
import string
import os,re
from os.path import exists
import random
import util
import gzip
from ffindex import *
import torch
from chemical import NAATOKENS, aa2num, aa2long, atomnum2atomtype, NTOTAL, CHAIN_GAP
from openbabel import openbabel

to1letter = {
    "ALA":'A', "ARG":'R', "ASN":'N', "ASP":'D', "CYS":'C',
    "GLN":'Q', "GLU":'E', "GLY":'G', "HIS":'H', "ILE":'I',
    "LEU":'L', "LYS":'K', "MET":'M', "PHE":'F', "PRO":'P',
    "SER":'S', "THR":'T', "TRP":'W', "TYR":'Y', "VAL":'V',
    "DA":'a', "DC":'c', "DG":'g', "DT":'t',
    "A":'b', "C":'d', "G":'h', "U":'u',
}

def read_template_pdb(L, pdb_fn, target_chain=None):
    # get full sequence from given PDB
    seq_full = list()
    prev_chain=''
    with open(pdb_fn) as fp:
        for line in fp:
            if line[:4] != "ATOM":
                continue
            if line[12:16].strip() != "CA":
                continue
            if line[21] != prev_chain:
                if len(seq_full) > 0:
                    L_s.append(len(seq_full)-offset)
                    offset = len(seq_full)
            prev_chain = line[21]
            aa = line[17:20]
            seq_full.append(aa2num[aa] if aa in aa2num.keys() else 20)

    seq_full = torch.tensor(seq_full).long()

    xyz = torch.full((L, 36, 3), np.nan).float()
    seq = torch.full((L,), 20).long()
    conf = torch.zeros(L,1).float()
    
    with open(pdb_fn) as fp:
        for line in fp:
            if line[:4] != "ATOM":
                continue
            resNo, atom, aa = int(line[22:26]), line[12:16], line[17:20]
            aa_idx = aa2num[aa] if aa in aa2num.keys() else 20
            #
            idx = resNo - 1
            for i_atm, tgtatm in enumerate(aa2long[aa_idx]):
                if tgtatm == atom:
                    xyz[idx, i_atm, :] = torch.tensor([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                    break
            seq[idx] = aa_idx
    
    mask = torch.logical_not(torch.isnan(xyz[:,:3,0])) # (L, 3)
    mask = mask.all(dim=-1)[:,None]
    conf = torch.where(mask, torch.full((L,1),0.1), torch.zeros(L,1)).float()
    seq_1hot = torch.nn.functional.one_hot(seq, num_classes=32).float()
    t1d = torch.cat((seq_1hot, conf), -1)

    #return seq_full[None], ins[None], L_s, xyz[None], t1d[None]
    return xyz[None], t1d[None]

def parse_fasta(filename,  maxseq=10000, rmsa_alphabet=False):
    msa = []
    ins = []

    fstream = open(filename,"r")

    for line in fstream:
        # skip labels
        if line[0] == '>':
            continue
            
        # remove right whitespaces
        line = line.rstrip()

        if len(line) == 0:
            continue

        # remove lowercase letters and append to MSA
        msa.append(line)

        # sequence length
        L = len(msa[-1])

        i = np.zeros((L))
        ins.append(i)

    # convert letters into numbers
    if rmsa_alphabet:
        alphabet = np.array(list("00000000000000000000-000000ACGTN"), dtype='|S1').view(np.uint8)
    else:
        alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-0acgtxbdhuy"), dtype='|S1').view(np.uint8)
    msa = np.array([list(s) for s in msa], dtype='|S1').view(np.uint8)
    for i in range(alphabet.shape[0]):
        msa[msa == alphabet[i]] = i

    ins = np.array(ins, dtype=np.uint8)

    return msa,ins

# parse a fasta alignment IF it exists
# otherwise return single-sequence msa
def parse_fasta_if_exists(seq, filename, maxseq=10000, rmsa_alphabet=False):
    if (exists(filename)):
        return parse_fasta(filename, maxseq, rmsa_alphabet)
    else:
        alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-0acgtxbdhuy"), dtype='|S1').view(np.uint8) # -0 are UNK/mask
        seq = np.array([list(seq)], dtype='|S1').view(np.uint8)
        for i in range(alphabet.shape[0]):
            seq[seq == alphabet[i]] = i

        return (seq, np.zeros_like(seq))

# read A3M and convert letters into
# integers in the 0..20 range,
# also keep track of insertions
def parse_a3m(filename, unzip=True, maxseq=10000):
    msa = []
    ins = []

    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))

    # read file line by line
    if (unzip):
        fstream = gzip.open(filename,"rt")
    else:
        fstream = open(filename,"r")

    for line in fstream:

        # skip labels
        if line[0] == '>':
            continue
            
        # remove right whitespaces
        line = line.rstrip()

        if len(line) == 0:
            continue

        # remove lowercase letters and append to MSA
        msa.append(line.translate(table))

        # sequence length
        L = len(msa[-1])

        # remove insertion at the end
        if (not unzip):
            n_remove = 0
            for c in reversed(line):
                if c.islower():
                    n_remove += 1
                else:
                    break
            line = line[:-n_remove]

        # 0 - match or gap; 1 - insertion
        a = np.array([0 if c.isupper() or c=='-' else 1 for c in line])
        i = np.zeros((L))

        if np.sum(a) > 0:
            # positions of insertions
            pos = np.where(a==1)[0]

            # shift by occurrence
            a = pos - np.arange(pos.shape[0])

            # position of insertions in cleaned sequence
            # and their length
            pos,num = np.unique(a, return_counts=True)

            # append to the matrix of insetions
            i[pos] = num

        ins.append(i)

        if (len(msa) >= maxseq):
            break

    # convert letters into numbers
    alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-"), dtype='|S1').view(np.uint8)
    msa = np.array([list(s) for s in msa], dtype='|S1').view(np.uint8)
    for i in range(alphabet.shape[0]):
        msa[msa == alphabet[i]] = i

    # treat all unknown characters as gaps
    msa[msa > 20] = 20

    ins = np.array(ins, dtype=np.uint8)

    return msa,ins


# read and extract xyz coords of N,Ca,C atoms
# from a PDB file
def parse_pdb(filename, seq=False):
    lines = open(filename,'r').readlines()
    if seq:
        return parse_pdb_lines_w_seq(lines, parse_hetatom=parse_hetatom)
    return parse_pdb_lines(lines)

def parse_pdb_lines_w_seq(lines):

    # indices of residues observed in the structure
    #idx_s = [int(l[22:26]) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]
    res = [(l[22:26],l[17:20]) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]
    idx_s = [int(r[0]) for r in res]
    seq = [aa2num[r[1]] if r[1] in aa2num.keys() else 20 for r in res]

    # 4 BB + up to 10 SC atoms
    xyz = np.full((len(idx_s), NTOTAL, 3), np.nan, dtype=np.float32)
    for l in lines:
        if l[:4] != "ATOM":
            continue
        resNo, atom, aa = int(l[22:26]), l[12:16], l[17:20]
        idx = idx_s.index(resNo)
        for i_atm, tgtatm in enumerate(aa2long[aa2num[aa]]):
            if tgtatm == atom:
                xyz[idx,i_atm,:] = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
                break

    # parse ligand atoms
    offset = max(idx_s)
    res_lig = [l[12:16].strip() for l in lines if l[:6]=="HETATM"]
    res_lig = [(i+offset+CHAIN_GAP,l) for i,l in enumerate(res_lig)]
    idx_s_lig = [int(r[0]) for r in res_lig]
    seq_lig = [aa2num[r[1]] if r[1] in aa2num.keys() else 20 for r in res_lig]

    xyz_s_lig = [[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in lines if l[:6]=='HETATM']

    if len(xyz_s_lig)>0:
        xyz_lig = np.full((len(idx_s_lig), NTOTAL, 3), np.nan, dtype=np.float32)
        xyz_lig[:,1,:] = np.array(xyz_s_lig)
        xyz = np.concatenate([xyz, xyz_lig],axis=0)
        idx_s = idx_s + idx_s_lig
        seq = seq + seq_lig

    # save atom mask
    mask = np.logical_not(np.isnan(xyz[...,0]))
    xyz[np.isnan(xyz[...,0])] = 0.0

    return xyz,mask,np.array(idx_s), np.array(seq)

#'''
def parse_pdb_lines(lines):

    # indices of residues observed in the structure
    idx_s = [int(l[22:26]) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]

    # 4 BB + up to 10 SC atoms
    xyz = np.full((len(idx_s), NTOTAL, 3), np.nan, dtype=np.float32)
    for l in lines:
        if l[:4] != "ATOM":
            continue
        resNo, atom, aa = int(l[22:26]), l[12:16], l[17:20]
        idx = idx_s.index(resNo)
        for i_atm, tgtatm in enumerate(aa2long[aa2num[aa]]):
            if tgtatm == atom:
                xyz[idx,i_atm,:] = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
                break

    # save atom mask
    mask = np.logical_not(np.isnan(xyz[...,0]))
    xyz[np.isnan(xyz[...,0])] = 0.0

    return xyz,mask,np.array(idx_s)


def parse_templates(item, params):

    # init FFindexDB of templates
    ### and extract template IDs
    ### present in the DB
    ffdb = FFindexDB(read_index(params['FFDB']+'_pdb.ffindex'),
                     read_data(params['FFDB']+'_pdb.ffdata'))
    #ffids = set([i.name for i in ffdb.index])

    # process tabulated hhsearch output to get
    # matched positions and positional scores
    infile = params['DIR']+'/hhr/'+item[-2:]+'/'+item+'.atab'
    hits = []
    for l in open(infile, "r").readlines():
        if l[0]=='>':
            key = l[1:].split()[0]
            hits.append([key,[],[]])
        elif "score" in l or "dssp" in l:
            continue
        else:
            hi = l.split()[:5]+[0.0,0.0,0.0]
            hits[-1][1].append([int(hi[0]),int(hi[1])])
            hits[-1][2].append([float(hi[2]),float(hi[3]),float(hi[4])])

    # get per-hit statistics from an .hhr file
    # (!!! assume that .hhr and .atab have the same hits !!!)
    # [Probab, E-value, Score, Aligned_cols, 
    # Identities, Similarity, Sum_probs, Template_Neff]
    lines = open(infile[:-4]+'hhr', "r").readlines()
    pos = [i+1 for i,l in enumerate(lines) if l[0]=='>']
    for i,posi in enumerate(pos):
        hits[i].append([float(s) for s in re.sub('[=%]',' ',lines[posi]).split()[1::2]])
        
    # parse templates from FFDB
    for hi in hits:
        #if hi[0] not in ffids:
        #    continue
        entry = get_entry_by_name(hi[0], ffdb.index)
        if entry == None:
            continue
        data = read_entry_lines(entry, ffdb.data)
        hi += list(parse_pdb_lines(data))

    # process hits
    counter = 0
    xyz,qmap,mask,f0d,f1d,ids = [],[],[],[],[],[]
    for data in hits:
        if len(data)<7:
            continue
        
        qi,ti = np.array(data[1]).T
        _,sel1,sel2 = np.intersect1d(ti, data[6], return_indices=True)
        ncol = sel1.shape[0]
        if ncol < 10:
            continue
        
        ids.append(data[0])
        f0d.append(data[3])
        f1d.append(np.array(data[2])[sel1])
        xyz.append(data[4][sel2])
        mask.append(data[5][sel2])
        qmap.append(np.stack([qi[sel1]-1,[counter]*ncol],axis=-1))
        counter += 1

    xyz = np.vstack(xyz).astype(np.float32)
    mask = np.vstack(mask).astype(np.bool)
    qmap = np.vstack(qmap).astype(np.long)
    f0d = np.vstack(f0d).astype(np.float32)
    f1d = np.vstack(f1d).astype(np.float32)
    ids = ids
        
    return xyz,mask,qmap,f0d,f1d,ids

def parse_templates_raw(ffdb, hhr_fn, atab_fn):
    # process tabulated hhsearch output to get
    # matched positions and positional scores
    hits = []
    for l in open(atab_fn, "r").readlines():
        if l[0]=='>':
            key = l[1:].split()[0]
            hits.append([key,[],[]])
        elif "score" in l or "dssp" in l:
            continue
        else:
            hi = l.split()[:5]+[0.0,0.0,0.0]
            hits[-1][1].append([int(hi[0]),int(hi[1])])
            hits[-1][2].append([float(hi[2]),float(hi[3]),float(hi[4])])

    # get per-hit statistics from an .hhr file
    # (!!! assume that .hhr and .atab have the same hits !!!)
    # [Probab, E-value, Score, Aligned_cols, 
    # Identities, Similarity, Sum_probs, Template_Neff]
    lines = open(hhr_fn, "r").readlines()
    pos = [i+1 for i,l in enumerate(lines) if l[0]=='>']
    for i,posi in enumerate(pos):
        hits[i].append([float(s) for s in re.sub('[=%]',' ',lines[posi]).split()[1::2]])
        
    # parse templates from FFDB
    for hi in hits:
        #if hi[0] not in ffids:
        #    continue
        entry = get_entry_by_name(hi[0], ffdb.index)
        if entry == None:
            continue
        data = read_entry_lines(entry, ffdb.data)
        hi += list(parse_pdb_lines_w_seq(data))

    # process hits
    counter = 0
    xyz,qmap,mask,f0d,f1d,ids,seq = [],[],[],[],[],[],[]
    for data in hits:
        if len(data)<7:
            continue
        
        qi,ti = np.array(data[1]).T
        _,sel1,sel2 = np.intersect1d(ti, data[6], return_indices=True)
        ncol = sel1.shape[0]
        if ncol < 10:
            continue
        
        ids.append(data[0])
        f0d.append(data[3])
        f1d.append(np.array(data[2])[sel1])
        xyz.append(data[4][sel2])
        mask.append(data[5][sel2])
        seq.append(data[-1][sel2])
        qmap.append(np.stack([qi[sel1]-1,[counter]*ncol],axis=-1))
        counter += 1
    
    xyz = np.vstack(xyz).astype(np.float32)
    qmap = np.vstack(qmap).astype(np.long)
    f1d = np.vstack(f1d).astype(np.float32)
    seq = np.hstack(seq).astype(np.long)
    ids = ids

    return torch.from_numpy(xyz), torch.from_numpy(qmap), \
           torch.from_numpy(f1d), torch.from_numpy(seq), ids

def read_templates(qlen, ffdb, hhr_fn, atab_fn, n_templ=10):
    xyz_t, qmap, t1d, seq, ids = parse_templates_raw(ffdb, hhr_fn, atab_fn)
    npick = min(n_templ, len(ids))
    if npick < 1: # no templates
        xyz = torch.full((1,qlen,27,3),np.nan).float()
        t1d = torch.nn.functional.one_hot(torch.full((1, qlen), 20).long(), num_classes=21).float() # all gaps
        t1d = torch.cat((t1d, torch.zeros((1,qlen,1)).float()), -1)
        return xyz, t1d

    sample = torch.arange(npick)
    #
    xyz = torch.full((npick, qlen, 27, 3), np.nan).float()
    f1d = torch.full((npick, qlen), 20).long()
    f1d_val = torch.zeros((npick, qlen, 1)).float()
    #
    for i, nt in enumerate(sample):
        sel = torch.where(qmap[:,1] == nt)[0]
        pos = qmap[sel, 0]
        xyz[i, pos] = xyz_t[sel]
        f1d[i, pos] = seq[sel]
        f1d_val[i,pos] = t1d[sel, 2].unsqueeze(-1)
    
    f1d = torch.nn.functional.one_hot(f1d, num_classes=21).float()
    f1d = torch.cat((f1d, f1d_val), dim=-1)

    return xyz, f1d

def parse_mol(filename, filetype="mol2", string=False):

    obConversion = openbabel.OBConversion()
    obConversion.SetInFormat(filetype)

    obmol = openbabel.OBMol()
    if string:
        obConversion.ReadString(obmol,filename)
    else:
        obConversion.ReadFile(obmol,filename)

    obmol.DeleteHydrogens()
    # the above sometimes fails to get all the hydrogens
    i = 1
    while i < obmol.NumAtoms()+1:
        if obmol.GetAtom(i).GetAtomicNum()==1:
            obmol.DeleteAtom(obmol.GetAtom(i))
        else:
            i += 1

    atomtypes = [atomnum2atomtype.get(obmol.GetAtom(i).GetAtomicNum(), 'ATM') for i in range(1, obmol.NumAtoms()+1)]
    msa = torch.tensor([aa2num[x] for x in atomtypes])
    ins = torch.zeros_like(msa)

    atom_coords = torch.tensor([[obmol.GetAtom(i).x(),obmol.GetAtom(i).y(), obmol.GetAtom(i).z()] 
                                for i in range(1, obmol.NumAtoms()+1)]).unsqueeze(0) # (1, natoms, 3)
    mask = torch.full(atom_coords.shape[:-1], True) # (1, natoms)

    try:
        automorphs = openbabel.vvpairUIntUInt()
        openbabel.FindAutomorphisms(obmol,automorphs)
        
        automorphs = torch.tensor(automorphs)
        n_symmetry = automorphs.shape[0]

        atom_coords = atom_coords.repeat(n_symmetry,1,1)
        mask = mask.repeat(n_symmetry,1)

        atom_coords = torch.scatter(atom_coords, 1, automorphs[:,:,0:1].repeat(1,1,3),
                                    torch.gather(atom_coords,1,automorphs[:,:,1:2].repeat(1,1,3)))
        mask = torch.scatter(mask, 1, automorphs[:,:,0],
                             torch.gather(mask, 1, automorphs[:,:,1]))

    except Exception as e:
        print(f"ERROR: automorphs for {filename} yielded invalid tensor")

    return obmol, msa, ins, atom_coords, mask

