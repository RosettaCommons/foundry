"""
This second script combines and filters the resultant shards from step_1.
"""

import argparse
import pandas as pd
import numpy as np
import re
from pathlib import Path
from curation_utils import get_master_df


SHARD_COLS = [
    "PDBID",
    "CHAINID",
    "LIGAND",
    "ASSEMBLY",
    "COVALENT",
    "PARTNERS",
    "LIGXF",
    "LIGATOMS",
    "LIGATOMS_RESOLVED",
    "LIGAND_ID",
    "PROT_CHAIN",
    "COMPLEX_LEN",
    "NUM_IDENTICAL_COPIES_QLIG",
    "NUM_CHAINS_IN_ASSEMBLY",
    "NUM_PROT_CHAINS_IN_ASSEMBLY",
    "MSA_MATCHING",
    "QLIG_POLAR_CONTACTS",
    "QLIG_FRACTION_HULL",
    "QLIG_DIAMETER",
    "QLIG_IS_SOI",
    "QLIG_IS_COORDINATED",
    "QLIG_IS_METAL",
]

MASTER_COLS = [
    "CHAINID",
    "DEPOSITION",
    "RESOLUTION",
    "HASH",
    "CLUSTER",
    "SEQUENCE",
    "LEN_EXIST",
]


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/projects/ml/RF2_allatom/sm_compl_20240412/",
        help="Output directory for the combined and filtered data",
    )
    parser.add_argument(
        "--remove_shards_after_merging",
        action="store_true",
        help="Remove the shards after merging",
    )
    return parser


def get_pdb_bind_df() -> pd.DataFrame:
    pdb_bind_csv_file = Path("pdb_bind.csv")
    if not pdb_bind_csv_file.exists():
        records = []
        lig_re = re.compile("\((.*?)\)")
        with open("/home/jue/dev/rfaa/index/INDEX_general_PL.2020") as f:
            lines = f.readlines()
            for line in lines:
                if not line.startswith("#"):
                    tokens = line.split("//")
                    desc = tokens[1]
                    ligand = lig_re.findall(desc)[0]
                    tokens = tokens[0].split()
                    records.append(
                        dict(
                            pdb=tokens[0],
                            resolution=tokens[1],
                            year=tokens[2],
                            affinity=tokens[3],
                            ligand=ligand,
                            description=desc,
                        )
                    )
        pdb_bind_df = pd.DataFrame.from_records(records)
        pdb_bind_df.to_csv(pdb_bind_csv_file)
    else:
        pdb_bind_df = pd.read_csv(pdb_bind_csv_file, na_filter=None)
    return pdb_bind_df


def get_biolip_df():
    biolip_csv_file = Path("biolip.csv")
    if not biolip_csv_file.exists():
        df_s = []
        for datestr in pd.date_range(start="2013-3-06", end="2022-03-30", freq="7D"):
            fn = f"/home/jue/dev/rfaa/data_curation_20221207/biolip/BioLiP_{str(datestr).split()[0]}.txt"

            dat = open(fn).read()
            if len(dat) == 0 or "404 Not Found" in dat:
                continue

            tmp = pd.read_csv(fn, delimiter="\t", header=None)
            tmp.columns = [
                "pdb_id",
                "pdb_chain",
                "resolution",
                "binding_site_id",
                "ligand_id",
                "ligand_chain",
                "ligand_serial_number",
                "bs_res",
                "bs_res_renum",
                "cat_site_res",
                "cat_site_res",
                "EC",
                "GO",
                "binding_affinity",
                "binding_affinity_MOAD",
                "binding_affinity_pdbbind",
                "binding_affinity_bindingDB",
                "uniprot",
                "pubmed",
                "receptor_seq",
            ]
            df_s.append(tmp)
        biolip_df = pd.concat(df_s)
        biolip_df.to_csv(biolip_csv_file)
    else:
        biolip_df = pd.read_csv(biolip_csv_file, na_filter=None)
    return biolip_df


def get_pdb_bind_entries() -> pd.DataFrame:
    pdb_bind_df = get_pdb_bind_df()
    pdbbind_entries = (
        pdb_bind_df[["pdb", "ligand"]]
        .rename(columns={"pdb": "PDBID", "ligand": "LIGAND_ID"})
        .drop_duplicates()
    )
    pdbbind_entries["IN_PDBBIND"] = True
    return pdbbind_entries


def get_biolip_entries() -> pd.DataFrame:
    biolip_df = get_biolip_df()
    biolip_entries = (
        biolip_df[["pdb_id", "ligand_id"]]
        .rename(columns={"pdb_id": "PDBID", "ligand_id": "LIGAND_ID"})
        .drop_duplicates()
    )
    biolip_entries["IN_BIOLIP"] = True
    return biolip_entries


def get_casf_astex_clusters(master_df: pd.DataFrame) -> np.ndarray:
    astex_casf_ids = [
        line.strip()
        for line in open(
            "/home/aivan/work/results/2022Jul23/list.astex_diverse"
        ).readlines()
    ]
    astex_casf_ids += [
        line.strip()
        for line in open(
            "/home/aivan/work/results/2022Jul23/list.astex_nonnat"
        ).readlines()
    ]
    astex_casf_ids += [
        line.strip()
        for line in open("/home/aivan/work/results/2022Jul23/list.casf2016").readlines()
    ]

    master_df["PDBID"] = master_df["CHAINID"].apply(lambda x: x.split("_")[0])
    astex_casf_ids_df = master_df[master_df["PDBID"].isin(astex_casf_ids)]
    astex_casf_protein_clusters = astex_casf_ids_df["CLUSTER"].drop_duplicates().values
    return astex_casf_protein_clusters


def get_gpcr_clusters(master_df: pd.DataFrame) -> np.ndarray:
    gpcr_ids = [
        "3eml",
        "3pbl",
        "3oeu",
        "3oe0",
        "3oe6",
        "3oe8",
        "3oe9",
        "4ib4",
        "4iar",
        "4iaq",
        "4jkv",
        "4n4w",
        "4qim",
        "4qin",
        "7vug",
        "7vuh",
        "7vgx",
    ]
    gpcr_ids_df = master_df[master_df["PDBID"].isin(gpcr_ids)]
    gpcr_protein_clusters = gpcr_ids_df["CLUSTER"].drop_duplicates().values
    return gpcr_protein_clusters


def main():
    parser = get_parser()
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    if not out_dir.exists():
        print("No shards found!")
        return

    shard_files = list(out_dir.glob("df_*.pkl"))
    if len(shard_files) == 0:
        print("No shards found!")
        return

    shards = [pd.read_pickle(shard_file)[SHARD_COLS] for shard_file in shard_files]
    df = pd.concat(shards)
    master_df = get_master_df()
    master_df_filtered = master_df[MASTER_COLS].drop_duplicates("CHAINID")
    df = df.merge(master_df_filtered, on="CHAINID", how="inner")

    pdbbind_entries = get_pdb_bind_entries()
    biolip_entries = get_biolip_entries()
    astex_casf_protein_clusters = get_casf_astex_clusters(master_df)
    gpcr_protein_clusters = get_gpcr_clusters(master_df)

    df = df.merge(biolip_entries, on=["PDBID", "LIGAND_ID"], how="left")
    df.loc[df["IN_BIOLIP"].isna(), "IN_BIOLIP"] = False

    df = df.merge(pdbbind_entries, on=["PDBID", "LIGAND_ID"], how="left")
    df.loc[df["IN_PDBBIND"].isna(), "IN_PDBBIND"] = False

    df["IN_ASTEX_CASF"] = df["CLUSTER"].isin(astex_casf_protein_clusters)
    df["IN_GPCR_CLUSTERS"] = df["CLUSTER"].isin(gpcr_protein_clusters)

    if args.remove_shards_after_merging:
        for shard_file in shard_files:
            shard_file.unlink()

    df.to_pickle(out_dir / "sm_compl.pkl")


if __name__:
    main()
