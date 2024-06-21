import pandas as pd
from typing import Dict, Any
from icecream import ic
from rf2aa.data.loaders.rcsb_loader import get_cif_metadata, loader_sm_compl_assembly


def get_partner_from_chainid(
    chainid, assembly_transform_index_dictionary, chain_id_type_dictionary
):
    chain_letter = chainid.split("_")[1]
    transform_index = assembly_transform_index_dictionary[chain_letter]
    chain_type = chain_id_type_dictionary[chain_letter]
    partner = (chain_letter, transform_index, 0, 0, chain_type)
    return partner

def choose_assembly(cif_outs, ids_list):
    chletter_list = [x.split("_")[1] for x in ids_list]
    for assembly in cif_outs["asmb"]:
        chains_in_asmb = [x[0] for x in cif_outs["asmb"][assembly]]
        if set(chletter_list).issubset(chains_in_asmb):
            return assembly
    raise ValueError(f"Could not find assembly for {ids_list}")

def spoofed_loader(
    item,
    params,
    chid2hash={},
    chid2taxid={},
    **kwargs,
):
    try:
        ids_list = item["CHAINID"].split(":")
        pdb_id = ids_list[0].split("_")[0]
        cif_outs = get_cif_metadata(pdb_id, "1", params)
        asmb = choose_assembly(cif_outs, ids_list)
        assembly_transform_index_dictionary = {
            k: i for i, (k, _) in enumerate(cif_outs["asmb"][asmb])
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
            "ASSEMBLY": asmb,
            "COVALENT": [],
            "PARTNERS": partners,
        }
    except Exception as e:
        # print exception so that whole traceback is visible    
        print(f"Error in spoofed_loader: {repr(e)}")
        ic(f"{item}")
        from rf2aa.tests.test_conditions import sm_compl_item
        spoofed_sm_compl_item = sm_compl_item
    import pdb; pdb.set_trace()
    return loader_sm_compl_assembly(
        spoofed_sm_compl_item,
        params,
        chid2hash,
        chid2taxid,
        cif_outs=cif_outs,
        **kwargs,
    )
