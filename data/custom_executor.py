import submitit
import shlex
import sys
from pathlib import Path
from typing import Union


class CustomSlurmExecutor(submitit.SlurmExecutor):
    def __init__(
        self,
        folder: Union[Path, str],
        max_num_timeout: int = 3,
        python: str = "apptainer run /software/containers/versions/SE3nv/SE3nv-20240415.sif",
    ) -> None:
        super().__init__(folder, max_num_timeout)
        self.python = python

    @property
    def _submitit_command_str(self) -> str:
        python = self.python or shlex.quote(sys.executable)
        return " ".join(
            [python, "-u -m submitit.core._submit", shlex.quote(str(self.folder))]
        )


def get_executor(
    job_name: str = "recurate_pdb",
    partition: str = "cpu",
    time: int = 60,
    array_parallelism: int = 32,
    cpus_per_task: int = 2,
    log_folder: str = "logs",
    mem_gb: int = 32,
) -> CustomSlurmExecutor:
    executor = CustomSlurmExecutor(folder=log_folder)
    executor.update_parameters(
        partition=partition,
        mem=f"{mem_gb}gb",
        job_name=job_name,
        cpus_per_task=cpus_per_task,
        ntasks_per_node=1,
        nodes=1,
        time=time,
        array_parallelism=array_parallelism,
    )
    return executor
