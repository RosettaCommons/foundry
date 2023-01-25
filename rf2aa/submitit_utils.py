import os
import argparse
import submitit
from typing import Dict, Optional


def add_slurm_args(parser: Optional[argparse.ArgumentParser] = None, prefix: str = "-") -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser(
            description="Submits a job to the digs cluster via submitit"
        )
    parser.add_argument(
        f"{prefix}slurm_log_path",
        default="/home/psturm/RF2-allatom/slurm_logs",
        type=str,
        help="Path where slurm logs will go."
    )
    parser.add_argument(
        f"{prefix}local",
        action="store_true",
        help="Set to true to train locally rather than submitting to slurm",
    )
    parser.add_argument(
        f"{prefix}slurm_partition",
        type=str,
        default="gpu",
        help="Slurm partition to run job on",
    )
    parser.add_argument(
        f"{prefix}gpu_type",
        type=str,
        default="a6000",
        help="Which gpus to run on, slurm gres constraint",
    )
    parser.add_argument(
        f"{prefix}cpu_memory",
        type=int,
        default=64,
        help="Amount of cpu job memory to request for slurm submission",
    )
    parser.add_argument(
        f"{prefix}cpus_per_task",
        type=int,
        default=4,
        help="Number of cpu cores to request for slurm submission",
    )
    parser.add_argument(
        f"{prefix}timeout_min",
        type=int,
        default=10000,
        help="Maximum number of minutes for slurm job to run",
    )
    parser.add_argument(
        f"{prefix}max_slurm_jobs_at_once",
        type=int,
        default=16,
        help="Maximum number of array jobs to run at once",
    )
    parser.add_argument(
        f"{prefix}num_gpus",
        type=int,
        default=4,
        help="Number of GPUs to train on"
    )
    parser.add_argument(
        f"{prefix}nodes",
        type=int,
        default=1,
        help="Number of nodes to submit to"
    )
    return parser


def create_executor(args: Dict, log_folder: str, job_name: str) -> submitit.AutoExecutor:
    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(
        slurm_partition=args.slurm_partition,
        slurm_mem=f"{args.cpu_memory}gb",
        slurm_job_name=job_name,
        cpus_per_task=args.cpus_per_task,
        slurm_ntasks_per_node=args.num_gpus,
        slurm_array_parallelism=args.max_slurm_jobs_at_once,
        nodes=args.nodes,
        timeout_min=args.timeout_min,
    )
    if args.gpu_type != "none":
        executor.update_parameters(
            slurm_gres=f"gpu:{args.gpu_type}:{args.num_gpus}",
        )

    return executor
