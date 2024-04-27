import argparse
import submitit
import warnings
import pandas as pd
import signal
import traceback
from torch.utils.data import default_collate
from typing import List, Tuple
from functools import partial
from pathlib import Path

from tqdm import tqdm
from custom_executor import get_executor
from rf2aa.data.compose_dataset import set_data_loader_params, default_dataloader_params
from rf2aa.data.data_loader import (
    _load_df,
    loader_sm_compl_assembly,
)
from rf2aa.chemical import initialize_chemdata
from rf2aa.set_seed import seed_all
from rf2aa.tools.debug_item import check_inputs


class Timeout:
    def __init__(self, seconds: int = 60, error_message: str = "Timeout error"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def create_parser() -> argparse.ArgumentParser():
    parser = argparse.ArgumentParser(description="Load all examples")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/projects/ml/RF2_allatom/sm_compl_20240412/",
        help="Path to the output directory",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1024,
        help="Number of shards to split the dataset into",
    )
    parser.add_argument(
        "--max_seconds_per_entry",
        type=int,
        default=60,
        help="Maximum seconds to spend on each entry",
    )
    parser.add_argument(
        "--reload_only_bad_shards",
        action="store_true",
        help="Try reloading items that have failed once due to time out or other errors.",
    )
    parser.add_argument(
        "--sub_dir_in",
        type=str,
        default="load_outs",
        help="Subdirectory to load which items failed and reload them.",
    )
    parser.add_argument(
        "--sub_dir_out",
        type=str,
        default="load_outs",
        help="Subdirectory to store the output files",
    )
    return parser


def run_indices(
    indices: List[int],
    df: pd.DataFrame,
    loader_params: dict,
    chid2hash: dict,
    chid2taxid: dict,
    max_seconds_per_entry: int = 60,
    verbose: bool = False,
    seeds: List[int] = None,
) -> Tuple[List[str], List[str]]:
    exceptions = []
    loaded_okay = []
    for i, index in enumerate(tqdm(indices)):
        row = df.loc[index].to_dict()
        if seeds is not None:
            seed_all(seeds[i])

        if verbose:
            print(f"Loading row {index}: {row}")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with Timeout(seconds=max_seconds_per_entry):
                    inputs = loader_sm_compl_assembly(
                        row,
                        loader_params,
                        chid2hash,
                        chid2taxid,
                        num_protein_chains=100,
                        num_ligand_chains=100,
                    )
                    inputs = default_collate([inputs])
                    check_inputs(inputs)
        except Exception as e:
            if verbose:
                print(f"Exception for row {index}")
                traceback.print_exc()
            exceptions.append(repr(e))
            loaded_okay.append(False)
            continue

        if verbose:
            print(f"Loaded row {index} without error")
        exceptions.append(None)
        loaded_okay.append(True)
    return exceptions, loaded_okay


def get_init_params():
    initialize_chemdata()
    loader_params = set_data_loader_params(default_dataloader_params)
    loader_params["MSA_LIMIT"] = 2
    loader_params["CROP"] = 1024
    loader_params["ligands_to_remove"] = []
    loader_params["OMIT_PERMUTATE"] = True
    loader_params["MAXNSYMM"] = 1
    loader_params["RADIAL_CROP"] = True
    loader_params["min_metal_contacts"] = 0
    loader_params["p_msa_mask"] = 0.0

    pdb_metadata = _load_df(loader_params["PDB_METADATA"])
    chid2hash = dict(zip(pdb_metadata.CHAINID, pdb_metadata.HASH))
    tmp = pdb_metadata.dropna(subset=["TAXID"])
    chid2taxid = dict(zip(tmp.CHAINID, tmp.TAXID))
    return loader_params, chid2hash, chid2taxid


def run_shard(
    shard: int = 0,
    num_shards: int = 1024,
    out_dir: str = "/projects/ml/RF2_allatom/sm_compl_20240412/",
    csv_file: str = "sm_compl.pkl",
    max_seconds_per_entry: int = 60,
    sub_dir_out: str = "load_outs",
    sub_dir_in: str = "load_outs",
    reload_only_bad_shards: bool = False,
):
    loader_params, chid2hash, chid2taxid = get_init_params()

    out_dir = Path(out_dir)
    csv_file = out_dir / csv_file
    if csv_file.suffix == ".csv":
        df = pd.read_csv(csv_file)
    else:
        df = pd.read_pickle(csv_file)

    shard_out_dir = out_dir / sub_dir_out
    shard_out_dir.mkdir(exist_ok=True, parents=True)

    if reload_only_bad_shards:
        assert (
            sub_dir_in != sub_dir_out
        ), "sub_dir_in and sub_dir_out must be different if reload_only_bad_shards is True."

        shard_in_dir = out_dir / sub_dir_in
        is_valid_dfs = [
            pd.read_csv(shard_file) for shard_file in shard_in_dir.glob("shard_*.csv")
        ]
        is_valid_df = pd.concat(is_valid_dfs)
        indices = is_valid_df.dropna()["index"].tolist()
    else:
        indices = df.index

    shard_indices = indices[shard::num_shards]
    exceptions, loaded_okay = run_indices(
        shard_indices,
        df,
        loader_params,
        chid2hash,
        chid2taxid,
        max_seconds_per_entry=max_seconds_per_entry,
    )
    out_df = pd.DataFrame(
        {
            "index": shard_indices,
            "exception": exceptions,
            "LOADED_WITHOUT_ERROR": loaded_okay,
        }
    )
    out_df.to_csv(shard_out_dir / f"shard_{shard:04d}.csv", index=False)


def main() -> List[submitit.Job]:
    executor = get_executor(
        job_name="recurate_step_4",
        time=180,
        array_parallelism=64,
    )
    parser = create_parser()
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    shard_out_dir = out_dir / args.sub_dir_out
    existing_shards = []
    if shard_out_dir.exists():
        existing_files = shard_out_dir.glob("shard_*.csv")
        existing_shards = [int(f.stem.split("_")[1]) for f in existing_files]

    shards_to_run = set(range(args.num_shards)) - set(existing_shards)

    if len(shards_to_run) == 0:
        print("No shards to run in step 4.")
        return []
    else:
        print(f"Running {len(shards_to_run)} shards in step 4.")

    run_fn = partial(
        run_shard,
        num_shards=args.num_shards,
        out_dir=args.out_dir,
        max_seconds_per_entry=args.max_seconds_per_entry,
        sub_dir_out=args.sub_dir_out,
        sub_dir_in=args.sub_dir_in,
        reload_only_bad_shards=args.reload_only_bad_shards,
    )
    jobs = executor.map_array(run_fn, shards_to_run)
    print(
        f"Submitted {len(jobs)} jobs, with ids: {jobs[0].job_id}, ... {jobs[-1].job_id}"
    )
    return jobs


if __name__ == "__main__":
    main()
