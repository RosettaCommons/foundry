import os
import numpy as np
import subprocess
import torch
import torch.multiprocessing as mp
import argparse
from pprint import pprint
from pathlib import Path

from rf2aa.submitit_utils import add_slurm_args, create_executor
from rf2aa.arguments import get_args
from factory import trainer_factory


def call_fn(args, dataset_param, model_param, loader_param, loss_param):
    master_port = str(np.random.randint(10000, 100000))
    master_address = (
        subprocess.check_output(
            ['scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1'], shell=True
        )
        .decode()
        .strip()
    )
    print(f"Setting MASTER_PORT={master_port} and MASTER_ADDR={master_address}")
    os.environ["MASTER_PORT"] = master_port
    os.environ["MASTER_ADDR"] = master_address

    print("============== INPUT ARGUMENTS ==============")
    pprint(vars(args))
    print("=============================================")

    mp.freeze_support()
    trainer_object = trainer_factory(args, dataset_param, model_param, loader_param, loss_param)
    trainer_object.run_model_training(torch.cuda.device_count())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=["train", "eval"],
        help="Set to eval to run evaluation only",
    )
    parser = add_slurm_args(parser)
    print("Reading in arguments...")
    args, dataset_param, model_param, loader_param, loss_param = get_args(parser)
    print("Done reading in arguments.")

    assert (
        args.local or not args.interactive
    ), "When submiting via submitit you either have to launch it locally, or not in interactive mode."

    if args.local:
        call_fn(args, dataset_param, model_param, loader_param, loss_param)
    else:
        if args.mode == "train":
            log_folder = Path(args.slurm_log_path) / args.model_name / "training_log/"
        else:
            if args.initialize_model_from_checkpoint is not None:
                model_restore_path = Path(args.initialize_model_from_checkpoint)
                model_name = model_restore_path.stem
            else:
                model_name = "no_checkpoint"

            log_folder = Path(args.slurm_log_path) / args.model_name / f"eval_{model_name}/"
        log_folder.parent.mkdir(parents=True, exist_ok=True)

        job_name = f"{args.model_name}_training"
        executor = create_executor(args, log_folder, job_name)
        job = executor.submit(
            call_fn, args, dataset_param, model_param, loader_param, loss_param
        )
        print(
            f"Submitted job {job.job_id} with name {job_name} and log folder {log_folder}"
        )


if __name__ == "__main__":
    main()
