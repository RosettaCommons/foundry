# '''
# Tailored dataset wrappers for design tasks
# '''

import json
import os
import textwrap
from os import PathLike
from typing import Any, Dict, List

import yaml
from atomworks.ml.datasets import MolecularDataset
from atomworks.ml.transforms.base import Compose, Transform
from omegaconf import DictConfig, OmegaConf
from rfd3.inference.input_parsing import (
    DesignInputSpecification,
)
from rfd3.utils.inference import ensure_input_is_abspath
from torch.utils.data import (
    DataLoader,
    SequentialSampler,
)

from foundry.utils.datasets import assemble_distributed_loader
from foundry.utils.ddp import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)
all_ranks_logger = RankedLogger(__name__, rank_zero_only=False)


class ContigJsonDataset(MolecularDataset):
    """
    Enables loading of JSON files containing contig data for benchmark design tasks,
    or the passing of examples through analogously-structured hydra configs.
    """

    def __init__(
        self,
        *,
        data: PathLike | Dict[str, dict | DesignInputSpecification],
        cif_parser_args: dict | None,
        transform: Transform | Compose | None,
        name: str | None,
        subset_to_keys: List[str] | None,
        eval_every_n: int,
    ):
        """
        Args:
            - data: path to the JSON file containing the contig data
            - cif_parser_args: arguments for the CIF parser
            - transform: transform to apply to the data
            - name: name of the dataset
            - subset_to_keys: list of keys to subset the data to
            - evaluate_every_n: how many times should this dataset be evaluated?
        """

        if isinstance(data, (PathLike, str)):
            self.json_path = data
            original_data = self._load_from_path(data)
        elif isinstance(data, DictConfig):
            self.json_path = None
            original_data = OmegaConf.to_object(data)
        else:
            self.json_path = None
            original_data = data

        # These will have already been added at inference time, but this block is useful for validation.
        if "global_args" in original_data:
            global_args = original_data.pop("global_args")
            for k, v in original_data.items():
                original_data[k].update(global_args)

        self._data = original_data

        if subset_to_keys is not None:
            assert (
                len(subset_to_keys) > 0
            ), "subset_to_keys must be a non-empty list of keys."
            self._data = {k: v for k, v in self._data.items() if k in subset_to_keys}
        self._check_json_keys()

        # ...basic assignments
        self.name = name if name is not None else "json-dataset"
        self.transform = transform

        self.cif_parser_args = cif_parser_args
        self.eval_every_n = eval_every_n

        if len(self) > 1_000:
            logger.warning(
                "ContigJsonDataset contains more than 1,000 entries. This may lead to performance issues."
            )
        elif len(self) == 0:
            raise ValueError(
                "ContigJsonDataset is empty, data: {}. Names: {}".format(
                    data, self.names
                )
            )

        l = 46
        fmt_names = textwrap.fill(
            ", ".join(self.names), width=l
        )  # .replace('\n', '+\n+ ')
        logger.info(
            f"\n+{l * '-'}+\n"
            f"Dataset {self.name}:\n"
            f"  - Found {len(self):,} examples:\n"
            f"{fmt_names}\n"
            f"\n+{l * '-'}+\n"
        )

    @staticmethod
    def _load_from_path(data):
        """Load data from a JSON or YAML file."""
        assert os.path.exists(data), f"Input file {data} does not exist."
        with open(data, "r") as f:
            if data.endswith(".json"):
                data = json.load(f)
            elif data.endswith(".yaml"):
                data = yaml.safe_load(f)
            else:
                raise ValueError(f"Input file {data} must be a JSON or YAML file.")
        return data

    def _check_json_keys(self):
        """Check if the JSON keys are valid."""
        for k, data in self.data.items():
            if not isinstance(data, (dict, DesignInputSpecification)):
                raise ValueError("Each item in the JSON data must be a dictionary.")

    @property
    def data(self):
        """Expose underlying dataframe as property to discourage changing it (can lead to unexpected behavior with torch ConcatDatasets)."""
        return self._data

    @property
    def names(self) -> List[str]:
        return list(self.data.keys())

    def __len__(self) -> int:
        """Pass through the length of the wrapped dataset."""
        return len(self.names)

    def __contains__(self, example_id: str) -> bool:
        """Pass through the contains method of the wrapped dataset."""
        return example_id in self.names

    def id_to_idx(self, example_id: str) -> int:
        """Pass through the id_to_idx method of the wrapped dataset."""
        return self.names.index(example_id)

    def idx_to_id(self, idx: int) -> str:
        """Pass through the idx_to_id method of the wrapped dataset."""
        return self.names[idx]

    def __getitem__(self, idx: int) -> Any:
        """Pass through the getitem method of the wrapped dataset."""
        example_id = self.idx_to_id(idx)
        spec = self.data[example_id]

        # if 'input' in metadata and not abspath, prepend the source json directory to the file path
        if not isinstance(spec, DesignInputSpecification):
            spec = ensure_input_is_abspath(spec, self.json_path)
            spec["cif_parser_args"] = self.cif_parser_args
            spec = DesignInputSpecification.safe_init(**spec)

        # Create pipeline input
        data = spec.to_pipeline_input(example_id=example_id)

        # Apply transforms and return
        data = self.transform(data)
        return data


def assemble_distributed_inference_loader_from_json(
    *, rank: int, world_size: int, **dataset_kwargs
) -> DataLoader:
    """
    Assemble a distributed inference DataLoader from JSONs.
    example:
        data={
            "backbone_0": {**args},
            "backbone_1": {**args}
        }
    """
    dataset = ContigJsonDataset(**dataset_kwargs)
    sampler = SequentialSampler(dataset)
    return assemble_distributed_loader(
        dataset=dataset,
        sampler=sampler,
        rank=rank,
        world_size=world_size,
    )
