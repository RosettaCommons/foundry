import pytest
import torch
from typing import Tuple, Dict, List, Optional

from rf2aa.data.data_loader import _load_df
from rf2aa.data.compose_dataset import default_dataloader_params
from rf2aa.data.loaders.polymer_partners import load_multi_msa
from rf2aa.data.parsers import parse_a3m, parse_fasta
from rf2aa.chemical import initialize_chemdata

initialize_chemdata()


def load_chid_mappings() -> Tuple[Dict[str, str], Dict[str, str]]:
    pdb_metadata = _load_df(default_dataloader_params["PDB_METADATA"])
    chid2hash = dict(zip(pdb_metadata.CHAINID, pdb_metadata.HASH))
    tmp = pdb_metadata.dropna(subset=["TAXID"])
    chid2taxid = dict(zip(tmp.CHAINID, tmp.TAXID))
    return chid2hash, chid2taxid


chid2hash, chid2taxid = load_chid_mappings()

inputs = [
    (
        "5gam",
        ["C", "B"],
        [735, 1008],
        ["polypeptide(L)", "polypeptide(L)"],
        True,
        False,
        "/projects/ml/RoseTTAComplex/pMSA/050/077/050649_077859.a3m.gz",
    ),
    (
        "5gam",
        ["C", "B"],
        [735, 1008],
        ["polypeptide(L)", "polypeptide(L)"],
        False,
        True,
        None,
    ),
    (
        "5gam",
        ["C"],
        [1008],
        ["polypeptide(L)"],
        False,
        False,
        "/projects/ml/TrRosetta/PDB-2021AUG02/a3m/050/050649.a3m.gz",
    ),
    (
        "5gam",
        ["C"],
        [1008],
        ["polypeptide(L)"],
        False,
        True,
        None,
    ),
    (
        "5gam",
        ["A"],
        [178],
        ["polyribonucleotide"],
        False,
        False,
        "/projects/ml/nucleic/torch/ga/5gam_A_0.afa",
    ),
    (
        "5gam",
        ["Z"], # Chain z does not exist, so this should default to the primary sequence
        [178],
        ["polyribonucleotide"],
        False,
        True,
        None,
    ),
]


@pytest.mark.parametrize(
    "pdb_id, chain_letters, Ls, chain_types, paired, single_sequence_mode, true_file",
    inputs,
)
def test_loading(
    pdb_id: str,
    chain_letters: List[str],
    Ls: List[int],
    chain_types: List[str],
    paired: bool,
    single_sequence_mode: bool,
    true_file: Optional[str],
):
    chain_ids = [f"{pdb_id}_{chain_letter}" for chain_letter in chain_letters]
    seq_poly = torch.arange(0, sum(Ls), dtype=torch.long)

    if single_sequence_mode:
        a3m = load_multi_msa(
            chain_ids,
            chain_types,
            Ls,
            seq_poly,
            {},
            {},
            default_dataloader_params,
        )
    else:
        a3m = load_multi_msa(
            chain_ids,
            chain_types,
            Ls,
            seq_poly,
            chid2hash,
            chid2taxid,
            default_dataloader_params,
        )

    if single_sequence_mode:
        assert a3m["msa"].shape[0] == 1
        assert a3m["ins"].shape[0] == 1
        assert a3m["is_paired"].shape[0] == 1

        assert (seq_poly[None] == a3m["msa"]).all().item()
        assert (a3m["ins"] == 0).all().item()
    else:
        if "a3m" in true_file:
            msa, ins, _ = parse_a3m(true_file, paired=paired)
        else:
            msa, ins = parse_fasta(true_file, rmsa_alphabet=True)
        msa = torch.from_numpy(msa)
        ins = torch.from_numpy(ins)

        assert (a3m["msa"] == msa).all().item()
        assert (a3m["ins"] == ins).all().item()
        assert a3m["is_paired"].all().item()

if __name__ == "__main__":
    test_loading(*inputs[5])
    