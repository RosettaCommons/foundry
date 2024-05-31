import pandas as pd
from typing import Dict, Any
from rf2aa.data.loaders.rcsb_loader import get_cif_metadata, loader_sm_compl_assembly


def get_partner_from_chainid(
    chainid, assembly_transform_index_dictionary, chain_id_type_dictionary
):
    chain_letter = chainid.split("_")[1]
    transform_index = assembly_transform_index_dictionary[chain_letter]
    chain_type = chain_id_type_dictionary[chain_letter]
    partner = (chain_letter, transform_index, 0, 0, chain_type)
    return partner


# This function should work for na_compl, dna, rna, compl and pdb examples, theoretically.
# Someone with more knowledge of those datasets should probably test this.
# It basically just assumes that CHAINID is a colon-separated list of chain ids.
# Note that it also assumes that we are only dealing with the first bio-assembly, which
# may be an assumption that we want to change in the future.
def spoofed_loader(
    item,
    params,
    chid2hash={},
    chid2taxid={},
    **kwargs,
):
    ids_list = item["CHAINID"].split(":")
    pdb_id = ids_list[0].split("_")[0]

    cif_outs = get_cif_metadata(pdb_id, "1", params)
    assembly_transform_index_dictionary = {
        k: i for i, (k, _) in enumerate(cif_outs["asmb"]["1"])
    }
    chain_id_type_dictionary = {k: v.type for k, v in cif_outs["chains"].items()}

    partners = [
        get_partner_from_chainid(
            chid, assembly_transform_index_dictionary, chain_id_type_dictionary
        )
        for chid in ids_list
    ]

    spoofed_sm_compl_item = {
        "CHAINID": ids_list[0],
        "ASSEMBLY": "1",
        "COVALENT": [],
        "PARTNERS": partners,
    }
    return loader_sm_compl_assembly(
        spoofed_sm_compl_item,
        params,
        chid2hash,
        chid2taxid,
        cif_outs=cif_outs,
        **kwargs,
    )
