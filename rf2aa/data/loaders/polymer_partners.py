import torch
import os
import numpy as np
import warnings
from typing import Any, Dict, List, Tuple
from itertools import permutations
from rf2aa.data.parsers import parse_a3m, parse_mixed_fasta, parse_fasta
from rf2aa.chemical import ChemicalData as ChemData
from rf2aa.data.data_loader import (
    TemplFeaturize,
    blank_template,
    merge_hetero_templates,
)
from rf2aa.util import (
    random_rot_trans,
    cif_poly_to_xyz,
    map_identical_poly_chains,
    cartprodcat,
)


def sequence_to_msa(seq_poly: torch.Tensor) -> Dict[str, Any]:
    msa = seq_poly[None]
    ins = torch.zeros_like(msa)
    taxid = np.array(["query"])
    is_paired = torch.ones(msa.shape[0]).bool()
    a3m_new = {
        "msa": msa,
        "ins": ins,
        "taxid": taxid,
        "is_paired": is_paired,
    }
    return a3m_new


def remove_all_gap_seqs(a3m):
    """Removes sequences that are all gaps from an MSA represented as `a3m` dictionary"""
    idx_seq_keep = ~(a3m["msa"] == ChemData().UNKINDEX).all(dim=1)
    a3m["msa"] = a3m["msa"][idx_seq_keep]
    a3m["ins"] = a3m["ins"][idx_seq_keep]
    return a3m


def get_matched_indices(a3m, taxids_shared):
    """
    Given an msa with taxids, finds the indices of the sequences in the MSA that have the taxids
    in taxids_shared. Returns the indices in sorted lexicographic order first by taxid and then by
    how well the sequence matches the query.
    """
    mask = np.isin(a3m["taxid"], taxids_shared)
    msa_indices = np.where(mask)[0]

    taxids = a3m["taxid"][mask]
    num_matches = (a3m["msa"][mask] == a3m["msa"][0:1]).sum(dim=1)
    sort_indices = np.lexsort((num_matches, taxids))[::-1]

    matched_indices = msa_indices[sort_indices]
    return matched_indices


def dfill(a):
    n = a.size
    b = np.concatenate([[0], np.where(a[:-1] != a[1:])[0] + 1, [n]])
    return np.arange(n)[b[:-1]].repeat(np.diff(b))


def argunsort(s):
    n = s.size
    u = np.empty(n, dtype=np.int64)
    u[s] = np.arange(n)
    return u


def cumcount(a):
    """
    https://stackoverflow.com/questions/40602269/how-to-use-numpy-to-get-the-cumulative-count-by-unique-values-in-linear-time
    """
    n = a.size
    s = a.argsort(kind="mergesort")
    i = argunsort(s)
    b = a[s]
    return (np.arange(n) - dfill(b))[i]


def remove_extraneous_taxid_copies(a3m_a, a3m_b, i_paired_a, i_paired_b):
    """
    For index sets i_paired_a and i_paired_b, this function removes indices
    in i_paired_a and i_paired_b until the taxids in both match not only in taxid
    but also in repeat number. For example, if:

    i_paired_a = [0, 1, 2, 3, 4]
    i_paired_b = [0, 1, 2, 3, 4]
    taxids_a = ["a", "a", "a", "b", "b"]
    taxids_b = ["a", "a", "b", "b", "b"]

    then this function returns:
    ([0, 1, 3, 4], [0, 1, 2, 3])
    """
    taxids_a = a3m_a["taxid"][i_paired_a]
    taxids_b = a3m_b["taxid"][i_paired_b]

    counts_a = np.char.add("_", cumcount(taxids_a).astype(str))
    counts_b = np.char.add("_", cumcount(taxids_b).astype(str))

    taxids_a_with_counts = np.char.add(taxids_a, counts_a)
    taxids_b_with_counts = np.char.add(taxids_b, counts_b)

    discard_mask_a = np.isin(taxids_a_with_counts, taxids_b_with_counts)
    discard_mask_b = np.isin(taxids_b_with_counts, taxids_a_with_counts)
    i_paired_a = i_paired_a[discard_mask_a]
    i_paired_b = i_paired_b[discard_mask_b]
    return i_paired_a, i_paired_b


def get_paired(a3mA, a3mB, taxids_shared):
    """
    Fully vectorized implementation of the following.
    Given a set of taxids that are shared between two MSAs, this
    function:
        1. Finds the indices of the sequences in the MSAs that
            have those taxids, and returns them in sorted lexicographic order
            first by taxid and then by how well the sequence matches the query.
        3. Removes extraneous copies of taxids in the indices found in step 1, such that
            both index sets have the same taxids with the same multiplicity.

    Note that all of the operations done are stable, so the order of the sequences in the final
    indices will be paired, first by taxid and then by how well the sequence matches the query.
    """
    i_pairedA = get_matched_indices(a3mA, taxids_shared)
    i_pairedB = get_matched_indices(a3mB, taxids_shared)

    i_pairedA, i_pairedB = remove_extraneous_taxid_copies(
        a3mA, a3mB, i_pairedA, i_pairedB
    )

    return i_pairedA, i_pairedB


def join_msas_by_taxid(a3mA, a3mB, idx_overlap=None):
    """Joins (or "pairs") 2 MSAs by matching sequences with the same
    taxonomic ID. If more than 1 sequence exists in both MSAs with the same tax
    ID, only the sequence with the highest sequence identity to the query (1st
    sequence in MSA) will be paired.

    Sequences that aren't paired will be padded and added to the bottom of the
    joined MSA.  If a subregion of the input MSAs overlap (represent the same
    chain), the subregion residue indices can be given as `idx_overlap`, and
    the overlap region of the unpaired sequences will be included in the joined
    MSA.

    Parameters
    ----------
    a3mA : dict
        First MSA to be joined, with keys `msa` (N_seq, L_seq), `ins` (N_seq,
        L_seq), `taxid` (N_seq,), and optionally `is_paired` (N_seq,), a
        boolean tensor indicating whether each sequence is fully paired. Can be
        a multi-MSA (contain >2 sub-MSAs).
    a3mB : dict
        2nd MSA to be joined, with keys `msa`, `ins`, `taxid`, and optionally
        `is_paired`. Can be a multi-MSA ONLY if not overlapping with 1st MSA.
    idx_overlap : tuple or list (optional)
        Start and end indices of overlap region in 1st MSA, followed by the
        same in 2nd MSA.

    Returns
    -------
    a3m : dict
        Paired MSA, with keys `msa`, `ins`, `taxid` and `is_paired`.
    """
    # preprocess overlap region
    L_A, L_B = a3mA["msa"].shape[1], a3mB["msa"].shape[1]
    if idx_overlap is not None:
        i1A, i2A, i1B, i2B = idx_overlap
        i1B_new, i2B_new = (
            (0, i1B) if i2B == L_B else (i2B, L_B)
        )  # MSA B residues that don't overlap MSA A
        assert (i1B == 0) or (i2B == a3mB["msa"].shape[1]), (
            "When overlapping with 1st MSA, 2nd MSA must comprise at most 2 sub-MSAs "
            "(i.e. residue range should include 0 or a3mB['msa'].shape[1])"
        )
    else:
        i1B_new, i2B_new = (0, L_B)

    # pair sequences
    taxids_shared = a3mA["taxid"][np.isin(a3mA["taxid"], a3mB["taxid"])]
    i_pairedA, i_pairedB = get_paired(a3mA, a3mB, taxids_shared)

    # unpaired sequences
    i_unpairedA = np.setdiff1d(np.arange(a3mA["msa"].shape[0]), i_pairedA)
    i_unpairedB = np.setdiff1d(np.arange(a3mB["msa"].shape[0]), i_pairedB)
    N_paired, N_unpairedA, N_unpairedB = (
        len(i_pairedA),
        len(i_unpairedA),
        len(i_unpairedB),
    )

    # handle overlap region
    # if msa A consists of sub-MSAs 1,2,3 and msa B of 2,4 (i.e overlap region is 2),
    # this diagram shows how the variables below make up the final multi-MSA
    # (* denotes nongaps, - denotes gaps)
    #  1 2 3 4
    # |*|*|*|*|   msa_paired
    # |*|*|*|-|   msaA_unpaired
    # |-|*|-|*|   msaB_unpaired
    if idx_overlap is not None:
        assert (
            (a3mA["msa"][i_pairedA, i1A:i2A] == a3mB["msa"][i_pairedB, i1B:i2B])
            | (a3mA["msa"][i_pairedA, i1A:i2A] == ChemData().UNKINDEX)
        ).all(), "Paired MSAs should be identical (or 1st MSA should be all gaps) in overlap region"

        # overlap region gets sequences from 2nd MSA bc sometimes 1st MSA will be all gaps here
        msa_paired = torch.cat(
            [
                a3mA["msa"][i_pairedA, :i1A],
                a3mB["msa"][i_pairedB, i1B:i2B],
                a3mA["msa"][i_pairedA, i2A:],
                a3mB["msa"][i_pairedB, i1B_new:i2B_new],
            ],
            dim=1,
        )
        msaA_unpaired = torch.cat(
            [
                a3mA["msa"][i_unpairedA],
                torch.full((N_unpairedA, i2B_new - i1B_new), ChemData().UNKINDEX),
            ],
            dim=1,
        )
        msaB_unpaired = torch.cat(
            [
                torch.full((N_unpairedB, i1A), ChemData().UNKINDEX),
                a3mB["msa"][i_unpairedB, i1B:i2B],
                torch.full((N_unpairedB, L_A - i2A), ChemData().UNKINDEX),
                a3mB["msa"][i_unpairedB, i1B_new:i2B_new],
            ],
            dim=1,
        )
    else:
        # no overlap region, simple offset pad & stack
        # this code is actually a special case of "if" block above, but writing
        # this out explicitly here to make the logic more clear
        msa_paired = torch.cat(
            [a3mA["msa"][i_pairedA], a3mB["msa"][i_pairedB, i1B_new:i2B_new]], dim=1
        )
        msaA_unpaired = torch.cat(
            [
                a3mA["msa"][i_unpairedA],
                torch.full((N_unpairedA, L_B), ChemData().UNKINDEX),
            ],
            dim=1,
        )  # pad with gaps
        msaB_unpaired = torch.cat(
            [
                torch.full((N_unpairedB, L_A), ChemData().UNKINDEX),
                a3mB["msa"][i_unpairedB],
            ],
            dim=1,
        )  # pad with gaps

    # stack paired & unpaired
    msa = torch.cat([msa_paired, msaA_unpaired, msaB_unpaired], dim=0)
    taxids = np.concatenate(
        [
            a3mA["taxid"][i_pairedA],
            a3mA["taxid"][i_unpairedA],
            a3mB["taxid"][i_unpairedB],
        ]
    )

    # label "fully paired" sequences (a row of MSA that was never padded with gaps)
    # output seq is fully paired if seqs A & B both started out as paired and were paired to
    # each other on tax ID.
    # NOTE: there is a rare edge case that is ignored here for simplicity: if
    # pMSA 0+1 and 1+2 are joined and then joined to 2+3, a seq that exists in
    # 0+1 and 2+3 but NOT 1+2 will become fully paired on the last join but
    # will not be labeled as such here
    is_pairedA = (
        a3mA["is_paired"]
        if "is_paired" in a3mA
        else torch.ones((a3mA["msa"].shape[0],)).bool()
    )
    is_pairedB = (
        a3mB["is_paired"]
        if "is_paired" in a3mB
        else torch.ones((a3mB["msa"].shape[0],)).bool()
    )
    is_paired = torch.cat(
        [
            is_pairedA[i_pairedA] & is_pairedB[i_pairedB],
            torch.zeros((N_unpairedA + N_unpairedB,)).bool(),
        ]
    )

    # insertion features in paired MSAs are assumed to be zero
    a3m = dict(msa=msa, ins=torch.zeros_like(msa), taxid=taxids, is_paired=is_paired)
    return a3m


def choose_multimsa_clusters(msa_seq_is_paired, params):
    """Returns indices of fully-paired sequences in a multi-MSA to use as seed
    clusters during MSA featurization.
    """
    frac_paired = msa_seq_is_paired.float().mean()
    if (
        frac_paired > 0.25
    ):  # enough fully paired sequences, just let MSAFeaturize choose randomly
        return None
    else:
        # ensure that half of the clusters are fully-paired sequences,
        # and let the rest be chosen randomly
        N_seed = params["MAXLAT"] // 2
        msa_seed_clus = []
        for i_cycle in range(params["MAXCYCLE"]):
            idx_paired = torch.where(msa_seq_is_paired)[0]
            msa_seed_clus.append(idx_paired[torch.randperm(len(idx_paired))][:N_seed])
        return msa_seed_clus


def load_minimal_multi_msa(hash_list, taxid_list, Ls, seqs_unique, types_unique, hash2chid, params):
    """Load a multi-MSA, which is a MSA that is paired across more than 2
    chains. This loads the MSA for unique chains. Use 'expand_multi_msa` to
    duplicate portions of the MSA for homo-oligomer repeated chains.

    Given a list of unique MSA hashes, loads all MSAs (using paired MSAs where
    it can) and pairs sequences across as many sub-MSAs as possible by matching
    taxonomic ID. For details on how pairing is done, see
    `join_msas_by_taxid()`

    Parameters
    ----------
    hash_list : list of str
        Hashes of MSAs to load and join. Must not contain duplicates.
    taxid_list : list of str
        Taxonomic IDs of query sequences of each input MSA.
    Ls : list of int
        Lengths of the chains corresponding to the hashes.
    seqs_unique: list of tensors
        Tensors representing polymer sequences. Used as a fall back if the MSA cannot be loaded
        or if the MSA is somehow corrupt.
    types_unique: list of str
        The type of the given chain. One of "polypeptide(L)", "polydeoxyribonucleotide", "polyribonucleotide".
    
    Returns
    -------
    a3m_out : dict
        Multi-MSA with all input MSAs. Keys: `msa`,`ins` [torch.Tensor (N_seq, L)],
        `taxid` [np.array (Nseq,)], `is_paired` [torch.Tensor (N_seq,)]
    hashes_out : list of str
        Hashes of MSAs in the order that they are joined in `a3m_out`.
        Contains the same elements as the input `hash_list` but may be in a
        different order.
    Ls_out : list of int
        Lengths of each chain in `a3m_out`
    """
    assert len(hash_list) == len(set(hash_list)), "Input MSA hashes must be unique"

    # the lists below are constructed such that `a3m_list[i_a3m]` is a multi-MSA
    # comprising sub-MSAs whose indices in the input lists are
    # `i_in = idx_list_groups[i_a3m][i_submsa]`, i.e. the sub-MSA hashes are
    # `hash_list[i_in]` and lengths are `Ls[i_in]`.
    # Each sub-MSA spans a region of its multi-MSA `a3m_list[i_a3m][:,i_start:i_end]`,
    # where `(i_start,i_end) = res_range_groups[i_a3m][i_submsa]`
    a3m_list = []  # list of multi-MSAs
    idx_list_groups = (
        []
    )  # list of lists of indices of input chains making up each multi-MSA
    res_range_groups = (
        []
    )  # list of lists of start and end residues of each sub-MSA in multi-MSA

    # iterate through all pairs of hashes and look for paired MSAs (pMSAs)
    # NOTE: in the below, if pMSAs are loaded for hashes 0+1 and then 2+3, and
    # later a pMSA is found for 0+2, the last MSA will not be loaded. The 0+1
    # and 2+3 pMSAs will still be joined on taxID at the end, but sequences
    # only present in the 0+2 pMSA pMSAs will be missed. this is probably very
    # rare and so is ignored here for simplicity.
    N = len(hash_list)
    for i1, i2 in permutations(range(N), 2):

        idx_list = [
            x for group in idx_list_groups for x in group
        ]  # flattened list of loaded hashes
        if i1 in idx_list and i2 in idx_list:
            continue  # already loaded
        if i1 == "" or i2 == "":
            continue  # no taxID means no pMSA

        h1 = hash_list[i1]
        h2 = hash_list[i2]
        if h1 is None or h2 is None:
            continue

        length_1 = Ls[i1]
        length_2 = Ls[i2]

        types_1 = types_unique[i1]
        types_2 = types_unique[i2]

        paired_msa = None
        paired_msa_file = None

        if types_1 == "polypeptide(L)" and types_2 == "polypeptide(L)" and taxid_list[i1] == taxid_list[i2]:
            paired_msa_file = f"{params['COMPL_DIR']}/pMSA/{h1[:3]}/{h2[:3]}/{h1}_{h2}.a3m.gz"
            if os.path.exists(paired_msa_file):
                msa, ins, taxid = parse_a3m(paired_msa_file, paired=True)
                paired_msa = {
                    "msa": msa,
                    "ins": ins,
                    "taxid": taxid,
                }
        elif types_1 == "polypeptide(L)" and types_2 == "polyribonucleotide":
            for chid_1 in hash2chid[h1]:
                # Nucleic acids have unique hash -> chid mappings
                chid_2 = hash2chid[h2][0]
                na_chain = chid_2.split("_")[1]

                paired_msa_file = f"{params['NA_DIR']}/msas/{chid_1[1:3]}/{chid_1[:4]}/{chid_1}_{na_chain}_paired.a3m"
                if os.path.exists(paired_msa_file):
                    msa, ins = parse_mixed_fasta(paired_msa_file)
                    taxid = ["query"] + ["0"] * (msa.shape[0] - 1)
                    taxid = np.array(taxid)
                    paired_msa = {
                        "msa": msa,
                        "ins": ins,
                        "taxid": taxid,
                    }
                    break

        if paired_msa is not None:
            msa = paired_msa["msa"]
            ins = paired_msa["ins"]
            taxid = paired_msa["taxid"]

            # Skip MSAs that are not the correct length
            # This usually occurs because the sequence was processed differently
            # in the CIF file and the sequence file used to generate the MSA.
            # In particular, it can happen because of modified residues being duplicated
            # in the sequence file.
            if msa.shape[1] != length_1 + length_2:
                warnings.warn(
                    f"Paired MSA {paired_msa_file} has length {msa.shape[1]} but expected length {length_1 + length_2}. Skipping."
                )
                continue
            
            a3m_new = dict(
                msa=torch.tensor(msa),
                ins=torch.tensor(ins),
                taxid=taxid,
                is_paired=torch.ones(msa.shape[0]).bool(),
            )
            res_range1 = (0, Ls[i1])
            res_range2 = (Ls[i1], msa.shape[1])

            # both hashes are new, add paired MSA to list
            if i1 not in idx_list and i2 not in idx_list:
                a3m_list.append(a3m_new)
                idx_list_groups.append([i1, i2])
                res_range_groups.append([res_range1, res_range2])

            # one of the hashes is already in a multi-MSA
            # find that multi-MSA and join the new pMSA to it
            elif i1 in idx_list:
                # which multi-MSA & sub-MSA has the hash with index `i1`?
                i_a3m = np.where([i1 in group for group in idx_list_groups])[0][0]
                i_submsa = np.where(np.array(idx_list_groups[i_a3m]) == i1)[0][0]

                idx_overlap = res_range_groups[i_a3m][i_submsa] + res_range1
                a3m_list[i_a3m] = join_msas_by_taxid(
                    a3m_list[i_a3m], a3m_new, idx_overlap
                )

                idx_list_groups[i_a3m].append(i2)
                L = res_range_groups[i_a3m][-1][1]  # length of current multi-MSA
                L_new = res_range2[1] - res_range2[0]
                res_range_groups[i_a3m].append((L, L + L_new))

            elif i2 in idx_list:
                # which multi-MSA & sub-MSA has the hash with index `i2`?
                i_a3m = np.where([i2 in group for group in idx_list_groups])[0][0]
                i_submsa = np.where(np.array(idx_list_groups[i_a3m]) == i2)[0][0]

                idx_overlap = res_range_groups[i_a3m][i_submsa] + res_range2
                a3m_list[i_a3m] = join_msas_by_taxid(
                    a3m_list[i_a3m], a3m_new, idx_overlap
                )

                idx_list_groups[i_a3m].append(i1)
                L = res_range_groups[i_a3m][-1][1]  # length of current multi-MSA
                L_new = res_range1[1] - res_range1[0]
                res_range_groups[i_a3m].append((L, L + L_new))

    # add unpaired MSAs
    # ungroup hash indices now, since we're done making multi-MSAs
    idx_list = [x for group in idx_list_groups for x in group]
    for i in range(N):
        if i not in idx_list:
            a3m_new = None
            hash_i = hash_list[i]
            length_i = Ls[i]
            type_i = types_unique[i]

            if type_i == "polypeptide(L)":
                msa_file = f"{params['PDB_DIR']}/a3m/{hash_i[:3]}/{hash_i}.a3m.gz"
                if os.path.exists(msa_file):
                    msa, ins, taxid = parse_a3m(msa_file)

                    # Another length check, see above for explanation.
                    # If the length is not correct, we use simply load
                    # the sequence in single sequence mode.
                    if msa.shape[1] == length_i:
                        a3m_new = dict(
                            msa=torch.tensor(msa),
                            ins=torch.tensor(ins),
                            taxid=taxid,
                            is_paired=torch.ones(msa.shape[0]).bool(),
                        )
                    else:
                        warnings.warn(
                            f"MSA {msa_file} has length {msa.shape[1]} but expected length {length_i}. Loading single sequence instead."
                        )
                else:
                    warnings.warn(f"MSA {msa_file} not found. Loading single sequence instead.")
            elif type_i == "polyribonucleotide":
                msa_file = f"{params['NA_DIR']}/torch/{hash_i[1:3]}/{hash_i}_0.afa"
                if os.path.exists(msa_file):
                    msa, ins = parse_fasta(msa_file, rmsa_alphabet=True)

                    if msa.shape[1] == length_i:
                        taxid = ["query"] + ["0"] * (msa.shape[0] - 1)
                        taxid = np.array(taxid)
                        a3m_new = dict(
                            msa=torch.tensor(msa),
                            ins=torch.tensor(ins),
                            taxid=taxid,
                            is_paired=torch.ones(msa.shape[0]).bool(),
                        )
                    else:
                        warnings.warn(
                            f"MSA {msa_file} has length {msa.shape[1]} but expected length {length_i}. Loading single sequence instead."
                        )
                else:
                    warnings.warn(f"MSA {msa_file} not found. Loading single sequence instead.")

            if a3m_new is None:
                a3m_new = sequence_to_msa(seqs_unique[i])

            a3m_list.append(a3m_new)
            idx_list.append(i)

    Ls_out = [Ls[i] for i in idx_list]
    hashes_out = [hash_list[i] for i in idx_list]

    # join multi-MSAs & unpaired MSAs
    a3m_out = a3m_list[0]
    for i in range(1, len(a3m_list)):
        a3m_out = join_msas_by_taxid(a3m_out, a3m_list[i])

    return a3m_out, hashes_out, Ls_out


def expand_multi_msa(a3m, hashes_in, hashes_out, Ls_in, Ls_out, params):
    """Expands a multi-MSA of unique chains into an MSA of a
    hetero-homo-oligomer in which some chains appear more than once. The query
    sequences (1st sequence of MSA) are concatenated directly along the
    residue dimention. The remaining sequences are offset-tiled (i.e. "padded &
    stacked") so that exact repeat sequences aren't paired.

    For example, if the original multi-MSA contains unique chains 1,2,3 but
    the final chain order is 1,2,1,3,3,1, this function will output an MSA like
    (where - denotes a block of gap characters):

        1 2 - 3 - -
        - - 1 - 3 -
        - - - - - 1

    Parameters
    ----------
    a3m : dict
        Contains torch.Tensors `msa` and `ins` (N_seq, L) and np.array `taxid` (Nseq,),
        representing the multi-MSA of unique chains.
    hashes_in : list of str
        Unique MSA hashes used in `a3m`.
    hashes_out : list of str
        Non-unique MSA hashes desired in expanded MSA.
    Ls_in : list of int
        Lengths of each chain in `a3m`
    Ls_out : list of int
        Lengths of each chain desired in expanded MSA.
    params : dict
        Data loading parameters

    Returns
    -------
    a3m : dict
        Contains torch.Tensors `msa` and `ins` of expanded MSA. No
        taxids because no further joining needs to be done.
    """
    assert len(hashes_out) == len(Ls_out)
    assert set(hashes_in) == set(hashes_out)
    assert a3m["msa"].shape[1] == sum(Ls_in)

    # figure out which oligomeric repeat is represented by each hash in `hashes_out`
    # each new repeat will be offset in sequence dimension of final MSA
    counts = dict()
    n_copy = []  # n-th copy of this hash in `hashes`
    for h in hashes_out:
        if h in counts:
            counts[h] += 1
        else:
            counts[h] = 1
        n_copy.append(counts[h])

    # num sequences in source & destination MSAs
    N_in = a3m["msa"].shape[0]
    N_out = (N_in - 1) * max(n_copy) + 1  # concatenate query seqs, pad&stack the rest

    # source MSA
    msa_in, ins_in = a3m["msa"], a3m["ins"]

    # initialize destination MSA to gap characters
    msa_out = torch.full((N_out, sum(Ls_out)), ChemData().UNKINDEX)
    ins_out = torch.full((N_out, sum(Ls_out)), 0)

    # for each destination chain
    for i_out, h_out in enumerate(hashes_out):
        # identify index of source chain
        i_in = np.where(np.array(hashes_in) == h_out)[0][0]

        # residue indexes
        i1_res_in = sum(Ls_in[:i_in])
        i2_res_in = sum(Ls_in[: i_in + 1])
        i1_res_out = sum(Ls_out[:i_out])
        i2_res_out = sum(Ls_out[: i_out + 1])

        # copy over query sequence
        msa_out[0, i1_res_out:i2_res_out] = msa_in[0, i1_res_in:i2_res_in]
        ins_out[0, i1_res_out:i2_res_out] = ins_in[0, i1_res_in:i2_res_in]

        # offset non-query sequences along sequence dimension based on repeat number of a given hash
        i1_seq_out = 1 + (n_copy[i_out] - 1) * (N_in - 1)
        i2_seq_out = 1 + n_copy[i_out] * (N_in - 1)
        # copy over non-query sequences
        msa_out[i1_seq_out:i2_seq_out, i1_res_out:i2_res_out] = msa_in[
            1:, i1_res_in:i2_res_in
        ]
        ins_out[i1_seq_out:i2_seq_out, i1_res_out:i2_res_out] = ins_in[
            1:, i1_res_in:i2_res_in
        ]

    # only 1st oligomeric repeat can be fully paired
    is_paired_out = torch.cat([a3m["is_paired"], torch.zeros((N_out - N_in,)).bool()])

    a3m_out = dict(msa=msa_out, ins=ins_out, is_paired=is_paired_out)
    a3m_out = remove_all_gap_seqs(a3m_out)

    return a3m_out


def load_multi_msa(chain_ids, chain_types, Ls, seq_poly, chid2hash, chid2taxid, params):
    """Loads multi-MSA for an arbitrary number of polymer chains. Tries to
    locate paired MSAs and pair sequences across all chains by taxonomic ID.
    Unpaired sequences are padded and stacked on the bottom.
    """
    # get MSA hashes (used to locate a3m files) and taxonomic IDs (used to determine pairing)
    hashes = []
    hashes_unique = []
    taxids_unique = []
    Ls_unique = []
    seqs_unique = []
    types_unique = []
    hash2chid = {}

    offset = 0
    for chid, chain_type, L_ in zip(chain_ids, chain_types, Ls):
        # The "default" value here just needs to be a uniquely identifying string
        # that has no associated MSA so that the MSA loading code will fall back
        # to loading the sequence in single sequence mode.
        if chain_type == "polypeptide(L)":
            hash = chid2hash.get(chid, f"{chid}_single_sequence")
            taxid = chid2taxid.get(chid, f"{chid}_single_sequence")
        else:
            # Nucleic acids have unique hash -> chid mappings
            hash = chid
            taxid = chid

        hashes.append(hash)
        seq = seq_poly[offset : offset + L_]
        offset += L_

        if hash not in hash2chid:
            hash2chid[hash] = []
        hash2chid[hash].append(chid)
        
        if hash not in hashes_unique:
            hashes_unique.append(hash)
            taxids_unique.append(taxid)
            Ls_unique.append(L_)
            seqs_unique.append(seq)
            types_unique.append(chain_type)
    
    # loads multi-MSA for unique chains
    a3m_poly, hashes_unique, Ls_unique = load_minimal_multi_msa(
        hashes_unique, taxids_unique, Ls_unique, seqs_unique, types_unique, hash2chid, params
    )

    # expands multi-MSA to repeat chains of homo-oligomers
    a3m_poly = expand_multi_msa(a3m_poly, hashes_unique, hashes, Ls_unique, Ls, params)

    return a3m_poly

def featurize_asmb_poly(
    pdb_id,
    partners,
    params,
    chains,
    asmb_xfs,
    modres,
    chid2hash,
    pick_top=True,
    random_noise=5.0,
):
    """Loads multiple polymer chains from parsed CIF assembly into tensors.
    Outputs will contain chains roughly in the order that they appear in
    `partners` (decreasing number of contacts to query ligand), except that
    chains with different letters but the same sequence (homo-oligomers) are
    placed contiguously in the residue dimension. All homo-oligomer chain swaps
    are enumerated and stored in the leading dimension ("permutation
    dimension"). Chain swap permutations of different sets of homo-oligomers
    are combined by a cartesian product (e.g. a complex with 2 copies of chain
    A and 3 copies of chain B, where A and B have distinct sequences, will have
    2 (# chain swaps of A) * 6 (# chain swaps of B) = 12 total chain-swap
    permutations.

    Parameters
    ----------
    pdb_id : string
        PDB accession of example. Used to load the pre-parsed CIF data.
    partners : list of 5-tuples (partner, transform_index, num_contacts, min_dist, partner_type)
        Polymer chains to featurize. All elements should have one of the following
        partner types: "polypeptide(L)", "polydeoxyribonucleotide", "polyribonucleotide".
        `partner` contains the chain letter.
        `transform_index` is an integer index of the coordinate transform for
        each partner chain.
    params : dict
        Parameters for the data loader
    chains : dict
        Dictionary mapping chain letters to cifutils.Chain objects representing
        the chains in a PDB entry.
    asmb_xfs : list of 2-tuples (chain_id, torch.Tensor(4,4))
        Coordinate transforms for the current assembly
    modres : dict
        Maps modified residue names to their canonical equivalents. Any
        modified residue will be converted to its standard equivalent and
        coordinates for atoms with matching names will be saved.
    chid2hash : dict
        Maps chain ids (<pdbid>_<chain_letter>) to hash strings used to name homology
        template and MSA files. If None, no templates are loaded.
    num_polymer_chains : number of polymer chains to include in the assembly, if set to None
                        all neighboring polymer chains will be loaded
    Returns
    -------
    xyz_poly : tensor (N_chain_permutation, L_total, N_atoms, 3)
        Atom coordinates of all the polymer chains
    mask_poly : tensor (N_chain_permutation, L_total, N_atoms)
        Boolean mask for whether an atom exists in `xyz_poly`
    seq_poly : tensor (L_total,)
        Integer-coded sequence of the polymer chains
    ch_label_poly : tensor (L_total,)
        Integer-coded chain identity for each residue. Differs from chain letter
        in that different-lettered chains with the same sequence will have the
        same integer code
    xyz_t_poly : tensor (N_templates, L_total, N_atoms, 3)
        Atom coordinates of the templates
    f1d_t_poly : tensor (N_templates, L_total, N_t1d_features)
        1D template features
    mask_t_poly : tensor (N_templates, L_total, N_atoms)
        Boolean mask for whether template atoms exist
    Ls_poly : list (N_chains,)
        Length of each polymer chain
    ch_letters : list (N_chains,)
        Chain letter for each chain
    mod_residues_to_atomize : list
        List of tuples `((chain_letter, residue_num, residue_name),
        (chain_letter, xform_index))` representing chemically modified residues
        that should be atomized.
    """
    # assign number to each unique polymer sequence, irrespective of chain letter
    chnum2chlet = map_identical_poly_chains(partners, chains, modres)
    valid_partner_types = ["polypeptide(L)", "polydeoxyribonucleotide", "polyribonucleotide"]
    valid_partners = [
        p for p in partners if p[-1] in valid_partner_types
    ]
    if not isinstance(chid2hash, dict):
        raise ValueError("chid2hash must be provided to load polymer partners.")

    # polymer true coords
    xyz_poly, mask_poly, ch_label_poly, seq_poly = [], [], [], []
    xyz_t_poly, f1d_t_poly, mask_t_poly, tplt_ids = [], [], [], []
    ch_letters, Ls_poly = [], []
    chain_types = []
    for chnum, chlet_set in chnum2chlet.items():
        # every location of this chain
        partners_ch = [
            p for p in valid_partners if p[0] in chlet_set
        ]
        chain_type = partners_ch[0][-1]

        N_mer = len(partners_ch)
        xyz_chxf, mask_chxf, seq_chxf, mod_residues_to_atomize = [], [], [], []
        for p in partners_ch:
            xyz_, mask_, seq_, _, _, residues_to_atomize = cif_poly_to_xyz(
                chains[p[0]], asmb_xfs[p[1]], modres
            )
            residues_to_atomize = [
                (residue, (residue[0], p[1])) for residue in residues_to_atomize
            ]
            xyz_chxf.append(xyz_)  # (L, N_atoms, 3)
            mask_chxf.append(mask_)
            seq_chxf.append(seq_)
            mod_residues_to_atomize.extend(residues_to_atomize)
            Ls_poly.append(xyz_.shape[0])
            ch_letters.append(p[0])
            chain_types.append(chain_type)

        # concatenate all locations, repeat for every permutation of locations
        xyz_ch, mask_ch, seq_ch = [], [], []
        for idx in permutations(range(len(xyz_chxf))):
            xyz_ch.append(torch.cat([xyz_chxf[i] for i in idx], dim=0))
            mask_ch.append(torch.cat([mask_chxf[i] for i in idx], dim=0))
        xyz_ch = torch.stack(xyz_ch, dim=0)  # (perm(N_mer), L*N_mer, N_atoms, 3)
        mask_ch = torch.stack(mask_ch, dim=0)  # (perm(N_mer), L*N_mer, N_atoms)

        seq_ch = torch.cat(seq_chxf, dim=0)

        # save results for each chain
        xyz_poly.append(xyz_ch)
        mask_poly.append(mask_ch)
        seq_poly.append(seq_ch)
        ch_label_poly.append(torch.full((xyz_ch.shape[1],), chnum))
        chnum += 1

        # Load templates. 
        # 
        ntempl = np.random.randint(params["MINTPLT"], params["MAXTPLT"] + 1)
        chain_id = pdb_id + "_" + list(chlet_set)[0] # chlet_set all have same hash
        
        if ntempl < 1 or chain_type != "polypeptide(L)" or chain_id not in chid2hash:
            xyz_t_ch, f1d_t_ch, mask_t_ch, tplt_ids_ch = blank_template(
                n_tmpl=1, L=xyz_ch.shape[1], random_noise=random_noise
            )
        else:
            pdb_hash = chid2hash[chain_id]
            tplt = torch.load(
                params["PDB_DIR"]
                + "/torch/hhr/"
                + pdb_hash[:3]
                + "/"
                + pdb_hash
                + ".pt"
            )
            xyz_t_, f1d_t_, mask_t_, tplt_ids_ = TemplFeaturize(
                tplt,
                Ls_poly[-1],
                params,
                npick=ntempl,
                offset=0,
                pick_top=pick_top,
                random_noise=random_noise,
            )
            xyz_t_ch = torch.cat(
                [xyz_t_] + [random_rot_trans(xyz_t_) for i in range(N_mer - 1)], dim=1
            )  # (ntempl, L*N_mer, natm, 3)
            f1d_t_ch = torch.cat([f1d_t_] * N_mer, dim=1)  # (ntempl, L*N_mer, 21)
            mask_t_ch = torch.cat([mask_t_] * N_mer, dim=1)  # (ntempl, L*N_mer, natm)
            tplt_ids_ch = np.concatenate(
                [
                    tplt_ids_,
                ],
                axis=0,
            )  # (ntempl) -- don't need to concatenate on the length dimension

        xyz_t_poly.append(xyz_t_ch)
        f1d_t_poly.append(f1d_t_ch)
        mask_t_poly.append(mask_t_ch)
        tplt_ids.append(tplt_ids_ch)

    # cartesian product over each chain's location permutations
    xyz_poly = cartprodcat(
        xyz_poly
    )  # (prod_i(N_perm_i), sum_i(L_i*N_mer_i), N_atoms, 3)
    mask_poly = cartprodcat(
        mask_poly
    )  # (prod_i(N_perm_i), sum_i(L_i*N_mer_i), N_atoms)

    xyz_t_poly, f1d_t_poly, mask_t_poly, tplt_ids = merge_hetero_templates(
        xyz_t_poly, f1d_t_poly, mask_t_poly, tplt_ids, Ls_poly
    )

    ch_label_poly = torch.cat(ch_label_poly, dim=0)
    seq_poly = torch.cat(seq_poly, dim=0)

    return (
        xyz_poly,
        mask_poly.bool(),
        seq_poly,
        ch_label_poly,
        xyz_t_poly,
        f1d_t_poly,
        mask_t_poly,
        Ls_poly,
        ch_letters,
        chain_types,
        mod_residues_to_atomize,
        tplt_ids,
    )


def get_empty_polymer_partners() -> Dict[str, Any]:
    a3m_poly = {
        "msa": torch.zeros((0, 0), dtype=torch.long),
        "ins": torch.zeros((0, 0), dtype=torch.long),
        "is_paired": torch.zeros((0, ), dtype=torch.bool),
    }
    poly_outs = {
        "xyz_poly": torch.zeros((1, 0, ChemData().NTOTAL, 3), dtype=torch.float32),
        "mask_poly": torch.zeros((1, 0, ChemData().NTOTAL), dtype=torch.bool),
        "seq_poly": torch.zeros((0, ), dtype=torch.long),
        "ch_label_poly": torch.zeros((0, ), dtype=torch.long),
        "xyz_t_poly": torch.zeros((1, 0, ChemData().NTOTAL, 3), dtype=torch.float32),
        "f1d_t_poly": torch.zeros((1, 0, ChemData().NAATOKENS), dtype=torch.float32),
        "mask_t_poly": torch.zeros((1, 0, ChemData().NTOTAL), dtype=torch.bool),
        "Ls_poly": [],
        "ch_letters": [],
        "mod_residues_to_atomize": [],
        "tplt_ids": [],
        "a3m_poly": a3m_poly,
        "seed_msa_clus": None,
    }
    return poly_outs


def load_polymer_partners(
    poly_partners: List[Tuple[Any, ...]],
    params: Dict[str, Any],
    pdb_id: str,
    cif_outs: Dict[str, Any],
    chid2hash: Dict[str, str] = {},
    chid2taxid: Dict[str, str] = {},
    pick_top: bool = True,
    random_noise: float = 5.0,
) -> Dict[str, Any]:
    if len(poly_partners) == 0:
        return get_empty_polymer_partners()
    
    # load polymer chains
    try:
        (
            xyz_poly,
            mask_poly,
            seq_poly,
            ch_label_poly,
            xyz_t_poly,
            f1d_t_poly,
            mask_t_poly,
            Ls_poly,
            ch_letters,
            chain_types,
            mod_residues_to_atomize,
            tplt_ids,
        ) = featurize_asmb_poly(
            pdb_id,
            poly_partners,
            params,
            cif_outs["chains"],
            cif_outs["asmb_xfs"],
            cif_outs["modres"],
            chid2hash,
            pick_top=pick_top,
            random_noise=random_noise,
        )
    except Exception as e:
        import pdb; pdb.set_trace()
    # keep 1st template and random sample of others for params['MAXTPLT'] total
    if xyz_t_poly.shape[0] > params["MAXTPLT"]:
        sel = np.concatenate(
            [
                [0],
                np.random.permutation(xyz_t_poly.shape[0] - 1)[: params["MAXTPLT"] - 1]
                + 1,
            ]
        )
        xyz_t_poly = xyz_t_poly[sel]
        mask_t_poly = mask_t_poly[sel]
        f1d_t_poly = f1d_t_poly[sel]

    chain_ids = [pdb_id + "_" + chlet for chlet in ch_letters]
    a3m_poly = load_multi_msa(chain_ids, chain_types, Ls_poly, seq_poly, chid2hash, chid2taxid, params)
    seed_msa_clus = choose_multimsa_clusters(a3m_poly["is_paired"][1:], params)
    poly_outs = {
        "xyz_poly": xyz_poly,
        "mask_poly": mask_poly,
        "seq_poly": seq_poly,
        "ch_label_poly": ch_label_poly,
        "xyz_t_poly": xyz_t_poly,
        "f1d_t_poly": f1d_t_poly,
        "mask_t_poly": mask_t_poly,
        "Ls_poly": Ls_poly,
        "ch_letters": ch_letters,
        "mod_residues_to_atomize": mod_residues_to_atomize,
        "tplt_ids": tplt_ids,
        "a3m_poly": a3m_poly,
        "seed_msa_clus": seed_msa_clus,
    }
    return poly_outs
