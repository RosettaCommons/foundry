import pandas as pd
import argparse
import hydra
from tqdm import tqdm
from openbabel import pybel
from typing import Dict
from pathlib import Path


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/projects/ml/RF2_allatom/sm_compl_20240412/",
        help="Output directory for the combined and filtered data",
    )
    return parser


def filter_nonexistent_partners(df):
    from rf2aa.data.data_loader import _load_df
    from rf2aa.data.compose_dataset import default_dataloader_params

    pdb_metadata = _load_df(default_dataloader_params['PDB_METADATA'])
    chid2hash = dict(zip(pdb_metadata.CHAINID, pdb_metadata.HASH))
    updated_partners_list = []

    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        pdb_id = row["PDBID"]
        partners = row["PARTNERS"]
        protein_partners = [p for p in partners if p[-1] == "polypeptide(L)"]
        non_protein_partners = [p for p in partners if p[-1] != "polypeptide(L)"]
        protein_partners = [
            p for p in protein_partners if f"{pdb_id}_{p[0]}" in chid2hash
        ]
        updated_partners = protein_partners + non_protein_partners
        updated_partners_list.append(updated_partners)
    df["PARTNERS"] = updated_partners_list

    df = df.loc[df["CHAINID"].isin(chid2hash)]
    return df


def main():
    parser = get_parser()
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    df = out_dir / "sm_compl.pkl"
    if not df.exists():
        raise FileNotFoundError(f"File {df} not found")
    df = pd.read_pickle(df)
    df = filter_nonexistent_partners(df)
    df.to_pickle(out_dir / "sm_compl.pkl")


if __name__ == "__main__":
    main()
