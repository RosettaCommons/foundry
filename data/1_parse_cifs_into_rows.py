"""
This first script creates the original ligand lists with additional assembly metadata.
"""
import sys
import rf2aa.cifutils as cifutils
import submitit
import numpy as np
import argparse
import pandas as pd
import pickle
import gzip
import json
from datetime import datetime
from functools import partial
from tqdm import tqdm
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from rf2aa.chemical import initialize_chemdata
from custom_executor import get_executor
from criterion_utils import (
    get_ligand_name_count_dictionary,
    get_criterion,
    get_soi_ligands_from_pdb_id,
    get_ligand_name_from_query_ligand,
)
from curation_utils import (
    preparse_all_chains,
    preprocess_asmb,
    get_ligands,
    get_partners,
    has_non_biological_bonds,
    strip_ligand,
    get_coordinates_from_chain_list,
    query_ligand_is_bad,
    get_primary_protein_chain,
    get_partner_length,
    filter_ligands,
    cif_coords_match_msa_len,
    get_chid2hash,
)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract data from PDB")
    parser.add_argument(
        "--include_nucleic_acids",
        action="store_true",
        help="Whether or not to include nucleic acids in the partners list.",
    )
    parser.add_argument(
        "--close_dist_prot",
        type=float,
        default=30.0,
        help="The distance at which a protein has to have at least min_close_prot atoms within such that it is included in the partners list.",
    )
    parser.add_argument(
        "--min_close_prot",
        type=int,
        default=1,
        help="The minimum number of atoms within close_dist_prot for a protein to be included in the partners list.",
    )
    parser.add_argument(
        "--contact_dist_prot",
        type=float,
        default=10.0,
        help="The distance that is considered a `contact` between a ligand and a protein.",
    )
    parser.add_argument(
        "--max_prot_partners",
        type=int,
        default=10,
        help="The maximum number of protein partners to include in the output.",
    )
    parser.add_argument(
        "--contact_dist_lig",
        type=float,
        default=5.0,
        help="The distance that is considered a `contact` between two ligands.",
    )
    parser.add_argument(
        "--close_dist_lig",
        type=float,
        default=30.0,
        help="The distance at which a ligand has to have at ALL atoms within such that it is included in the partners list.",
    )
    parser.add_argument(
        "--max_lig_partners",
        type=int,
        default=10,
        help="The maximum number of ligand partners to include in the output.",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/projects/ml/RF2_allatom/rcsb/pkl/",
        help="Where to parse cif files from.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/projects/ml/RF2_allatom/sm_compl_20240412/",
        help="Where to save the output csv files.",
    )
    parser.add_argument(
        "--max_unique_copies_per_cif",
        type=int,
        default=25,
        help="The maximum number of unique copies of a ligand to consider in a given cif file.",
    )
    parser.add_argument(
        "--coordinating_distance",
        type=float,
        default=2.6,
        help="The distance at which a ligand is considered to be `coordinated` by min_coordinating residue atoms.",
    )
    parser.add_argument(
        "--min_coordinating",
        type=int,
        default=3,
        help="The minimum number of atoms within coordinating_distance for a ligand to be considered `coordinated` by a residue atom.",
    )
    parser.add_argument(
        "--ignore_ligand_list",
        type=str,
        default="HOH,DOD,EDO,PEG,GOL",
        help="Ligands to not consider during cif parsing.",
    )
    parser.add_argument(
        "--max_assembly_protein_chains",
        type=int,
        default=200,
        help="If an assembly has more protein chains than this, it is skipped. Mostly for speed reasons. You can set this to be very high if you want to include all assemblies.",
    )
    return parser


def get_cif_file(cif_file: str) -> Tuple[Any, ...]:
    return pickle.load(gzip.open(cif_file))


def get_rows(
    pdb_id: str,
    chid2hash: Optional[Dict[str, int]] = None,
    base_dir: Path = Path("/projects/ml/RF2_allatom/rcsb/pkl/"),
    safe_catch_exceptions: bool = True,
    **kwargs,
) -> List[Dict[str, Any]]:
    if chid2hash is None:
        chid2hash = get_chid2hash()
    
    cif_file = base_dir / f"{pdb_id[1:3]}/{pdb_id}.pkl.gz"
    cif_file_outs = get_cif_file(cif_file)
    chains, asmb, covale = cif_file_outs[:3]
    asmb = preprocess_asmb(asmb)

    ignore_ligands = kwargs.get("ignore_ligand_list", "HOH,DOD,EDO,PEG,GOL")
    ignore_ligands = ignore_ligands.strip().split(",")
    ignore_ligands = [c.strip() for c in ignore_ligands]
    ligands, lig_covale_list = get_ligands(chains, covale, ignore_ligands=ignore_ligands)
    ligands, lig_covale_list = filter_ligands(
        ligands,
        lig_covale_list,
        max_unique_copies_per_cif=kwargs.get("max_unique_copies_per_cif", 20),
    )

    if len(ligands) == 0:
        return []


    chain_xyz_dict = preparse_all_chains(chains, ignore_ligands=ignore_ligands)
    soi_ligand_set = get_soi_ligands_from_pdb_id(pdb_id)
    records = []
    for assembly_index in asmb:
        assembly_chain_ids = set([x[0] for x in asmb[assembly_index]])
        assembly_chains = [chains[i_ch] for i_ch in assembly_chain_ids]
        assembly_protein_chains = [
            chain for chain in assembly_chains if chain.type == "polypeptide(L)"
        ]
        if len(assembly_protein_chains) > kwargs.get("max_assembly_protein_chains", 98):
            print(f"Skipping assembly {assembly_index} in {pdb_id} with {len(assembly_protein_chains)} protein chains.")
            continue

        num_chains_in_assembly = len(assembly_chains)
        num_protein_chains_in_assembly = len(assembly_protein_chains)

        assembly_partner_chains = assembly_protein_chains if kwargs.get("include_nucleic_acids", False) else assembly_chains
        
        assembly_chain_dict = {chain.id: chain for chain in assembly_chains}
        try:
            xyz_chains, mask_chains, chain_letters = get_coordinates_from_chain_list(
                assembly_partner_chains, chain_xyz_dict=chain_xyz_dict, 
            )
        except Exception as e:
            if safe_catch_exceptions:
                print(
                    f"Error in get_coordinates_from_chain_list: {pdb_id}, {assembly_index}"
                )
                continue
            else:
                raise e

        assembly_transforms = asmb[assembly_index]

        assembly_ligand_name_counts = get_ligand_name_count_dictionary(
            assembly_chain_dict, covale
        )
        for query_ligand, query_lig_covale in zip(ligands, lig_covale_list):
            query_ligand_chids = set([res[0] for res in query_ligand])
            if not all(
                [
                    ligand_chid in assembly_chain_ids
                    for ligand_chid in query_ligand_chids
                ]
            ):
                continue

            if has_non_biological_bonds(query_lig_covale):
                continue

            try:
                (
                    query_ligand,
                    qlig_xyz,
                    qlig_xyz_valid,
                    qlig_mask,
                    qlig_seq,
                    qlig_chid,
                    qlig_resi,
                    qlig_chxf,
                ) = strip_ligand(chains, assembly_transforms, query_ligand, chain_xyz_dict=chain_xyz_dict)
            except Exception as e:
                if safe_catch_exceptions:
                    print(f"Error in strip_ligand: {pdb_id}, {assembly_index}")
                    continue
                else:
                    raise e
            query_ligand_name = get_ligand_name_from_query_ligand(query_ligand)

            if query_ligand_is_bad(
                qlig_xyz,
                qlig_xyz_valid,
                xyz_chains,
                mask_chains,
                chain_letters,
                assembly_transforms,
            ):
                continue

            try:
                partners = get_partners(
                    query_ligand,
                    qlig_xyz_valid,
                    qlig_chxf,
                    chains,
                    assembly_partner_chains,
                    assembly_transforms,
                    covale,
                    ligands,
                    xyz_chains,
                    mask_chains,
                    chain_letters,
                    chain_xyz_dict=chain_xyz_dict,
                    **kwargs,
                )
            except Exception as e:
                if safe_catch_exceptions:
                    print(f"Error in get_partners: {pdb_id}, {assembly_index}")
                    continue
                else:
                    raise e

            query_ligand_length = qlig_xyz.shape[0]
            partner_lengths = [
                get_partner_length(chains, partner) for partner in partners
            ]
            total_assembly_length = sum(partner_lengths) + query_ligand_length

            primary_protein_chain = get_primary_protein_chain(partners)
            if primary_protein_chain is None:
                continue

            try:
                filter_criterion = get_criterion(
                    query_ligand,
                    primary_protein_chain,
                    chains,
                    assembly_transforms,
                    covale,
                    qlig_chxf,
                    query_ligand_name,
                    soi_ligand_set,
                    **kwargs,
                )
            except Exception as e:
                if safe_catch_exceptions:
                    print(f"Error in get_criterion: {pdb_id}, {assembly_index}")
                    continue
                else:
                    raise e

            chain_id = f"{pdb_id}_{primary_protein_chain}"
            msa_matching = False
            if chain_id in chid2hash:
                hash = chid2hash[chain_id]
                try:
                    msa_matching = cif_coords_match_msa_len(
                        hash,
                        chains[primary_protein_chain],
                        chain_xyz_dict=chain_xyz_dict,
                    )
                except Exception as e:
                    if safe_catch_exceptions:
                        print(f"Error in cif_coords_match_msa_len: {pdb_id}, {assembly_index}")
                        continue
                    else:
                        raise e

            record = {
                "PDBID": pdb_id,
                "CHAINID": chain_id,
                "LIGAND": query_ligand,
                "ASSEMBLY": assembly_index,
                "COVALENT": query_lig_covale,
                "PARTNERS": partners,
                "LIGXF": qlig_chxf,
                "LIGATOMS": qlig_xyz.shape[0],
                "LIGATOMS_RESOLVED": qlig_xyz_valid.shape[0],
                "LIGAND_ID": query_ligand[0][2],
                "PROT_CHAIN": primary_protein_chain,
                "COMPLEX_LEN": total_assembly_length,
                "NUM_IDENTICAL_COPIES_QLIG": assembly_ligand_name_counts.get(query_ligand_name, 1),
                "NUM_CHAINS_IN_ASSEMBLY": num_chains_in_assembly,
                "NUM_PROT_CHAINS_IN_ASSEMBLY": num_protein_chains_in_assembly,
                "MSA_MATCHING": msa_matching,
            }
            record.update(filter_criterion)
            records.append(record)
    return records


def get_all_pdb_ids(
    base_dir: Path = Path("/projects/ml/RF2_allatom/rcsb/pkl/"),
) -> List[str]:
    files = base_dir.glob("*/*.pkl.gz")
    pdb_ids = [file.name.split(".")[0] for file in files]
    return pdb_ids


def run_fn(shard: int = 0, num_shards: int = 1000, args: argparse.Namespace = None):
    sys.modules["cifutils"] = cifutils
    initialize_chemdata()
    args.base_dir = Path(args.base_dir)
    args.out_dir = Path(args.out_dir)

    pdb_ids = get_all_pdb_ids(base_dir=args.base_dir)
    rng = np.random.default_rng(42)
    rng.shuffle(pdb_ids)

    chid2hash = get_chid2hash()

    pdb_ids = pdb_ids[shard::num_shards]
    records = []
    bar = tqdm(pdb_ids)
    for pdb_id in bar:
        bar.set_description(pdb_id)
        records += get_rows(pdb_id, chid2hash=chid2hash, **args.__dict__)
    df = pd.DataFrame.from_records(records)
    df.to_pickle(args.out_dir / f"df_{shard:04d}.pkl")


def get_single(pdb_id: str):
    sys.modules["cifutils"] = cifutils
    initialize_chemdata()
    parser = create_parser()
    args = parser.parse_args()
    args.base_dir = Path(args.base_dir)
    return get_rows(pdb_id, safe_catch_exceptions=False, **args.__dict__)


def main() -> List[submitit.Job]:
    parser = create_parser()
    args = parser.parse_args()
    submit_fn = partial(run_fn, args=args)

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        existing_files = list(out_dir.glob("df_*.pkl"))
        existing_shards = [
            int(file.name.split("_")[1].split(".")[0]) for file in existing_files
        ]
        remaining_shards = list(set(range(1000)) - set(existing_shards))
        print(f"Remaining shards: {remaining_shards}")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)    
        remaining_shards = range(1000)
        print("No existing files found. Running on all shards.")

    if len(remaining_shards) == 0:
        print("All shards already processed.")
        return []

    arg_dict = args.__dict__
    arg_dict["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    args_json_string = json.dumps(arg_dict, indent=4)
    with open(out_dir / "args.json", "w") as f:
        f.write(args_json_string)

    executor = get_executor(
        job_name="recurate_step_1",
    )
    jobs = executor.map_array(submit_fn, remaining_shards)
    print(f"Submitted {len(jobs)} jobs: {[job.job_id for job in jobs[:3]]}...")
    return jobs


if __name__ == "__main__":
    # main()
    print(pd.DataFrame.from_records(get_single("5cgq")))
