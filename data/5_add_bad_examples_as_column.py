import pandas as pd
import argparse
from pathlib import Path


def create_parser() -> argparse.ArgumentParser():
    parser = argparse.ArgumentParser(description="Add example exception as a column.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/projects/ml/RF2_allatom/sm_compl_20240412/",
        help="Path to the output directory",
    )
    return parser


def main(csv_file: str = "sm_compl.pkl"):
    parser = create_parser()
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    shard_out_dir = out_dir / "load_outs"
    is_valid_dfs = [
        pd.read_csv(shard_file) for shard_file in shard_out_dir.glob("shard_*.csv")
    ]
    is_valid_df = pd.concat(is_valid_dfs)
    
    csv_file = out_dir / csv_file
    df = pd.read_pickle(csv_file)
    invalid_indices = is_valid_df.loc[~is_valid_df["LOADED_WITHOUT_ERROR"], "index"]

    df["LOADED_WITHOUT_ERROR"] = True
    df.loc[invalid_indices, "LOADED_WITHOUT_ERROR"] = False
    df.to_pickle(csv_file)


if __name__ == "__main__":
    main()
