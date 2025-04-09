# '''
# Tailored dataset wrappers for design tasks
# '''

import json
import os
import tempfile
import textwrap
import time
from os import PathLike
from typing import Any, List

from cifutils.utils.io_utils import to_cif_file
from datahub.datasets.datasets import BaseDataset
from datahub.transforms.base import Compose, Transform, TransformedDict

from modelhub.utils.ddp import RankedLogger
from projects.aa_design.inference.input_parsing import (
    create_atom_array_from_design_specification,
    load_,
)
from projects.aa_design.transforms.masks import Mask

logger = RankedLogger(__name__, rank_zero_only=True)

class ContigJsonDataset(BaseDataset): # datahub.datasets.datasets
    '''
    Enables loading of JSON files containing contig data for benchmark design tasks.
    '''
    # allowed_json_keys = ['name', 'length']
    def __init__(self,
        *,
        data: PathLike,
        cif_parser_args: dict | None = None,
        transform: Transform | Compose | None = None, 
        name: str | None = None,
        subset_to_keys: List[str] | None = None,
    ):
    
        if isinstance(data, (PathLike, str)):
            self.json_path = data
            self._data = self._load_from_path(data)
        else:
            raise ValueError("Please specify path to a valid JSON file")
        
        if subset_to_keys is not None:
            assert len(subset_to_keys) > 0, "subset_to_keys must be a non-empty list of keys."
            self._data = {k: v for k, v in self._data.items() if k in subset_to_keys}
        self._check_json_keys()

        # ...basic assignments
        self.name = name if name is not None else "json-dataset"
        self.transform = transform
        
        self.cif_parser_args = cif_parser_args

        if len(self) > 1_000:
            logger.warning("ContigJsonDataset contains more than 1,000 entries. This may lead to performance issues.")
        elif len(self) == 0:
            raise ValueError("ContigJsonDataset is empty, file path: {}. Names: {}".format(data, self.names))

        l=46
        fmt_names = textwrap.fill(', '.join(self.names), width=l)#.replace('\n', '+\n+ ')
        logger.info(
            f"\n+{l*'-'}+\n"
            f"Dataset {self.name}:\n"
            f"  - Found {len(self):,} examples:\n"
            f"{fmt_names}\n"
            f"\n+{l*'-'}+\n"
        )

    @staticmethod
    def _load_from_path(data):
        """Load data from a JSON file."""
        assert os.path.exists(data), f"Input json file {data} does not exist."
        with open(data, 'r') as f:
            data = json.load(f)
        return data

    def _check_json_keys(self):
        """Check if the JSON keys are valid."""
        for k, data in self.data.items():
            if not isinstance(data, dict):
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
        metadata=self.data[example_id]
        metadata['example_id'] = example_id

        # if 'input' in metadata and not abspath, prepend the source json directory to the file path
        if 'input' in metadata and not os.path.isabs(metadata['input']) and self.json_path is not None:
            metadata['input'] = os.path.join(os.path.dirname(self.json_path), metadata['input'])

        _start_parse_time = time.time()
        data = load_or_create_example_from_metadata_dict(metadata, cif_parser_args=self.cif_parser_args)
        _stop_parse_time = time.time()

        data = TransformedDict(data)
        data.__transform_history__.append(
            dict(
                name="load_or_create_example_from_metadata_dict",
                instance=hex(id(load_or_create_example_from_metadata_dict)),
                start_time=_start_parse_time,
                end_time=_stop_parse_time,
                processing_time=_stop_parse_time - _start_parse_time,
            )
        )
        data = self.transform(data)
        return data

def load_or_create_example_from_metadata_dict(  # see datahub.datasets.parsers.base.load_example_from_metadata_row
    metadata_dict: dict,
    *,
    cif_parser_args: dict,
) -> dict:
    """
    Load or create an example from a metadata dictionary.
    If the file path is not provided in the metadata dictionary, create a spoofed CIF file based on the length.
    Args:
        metadata_dict (dict): The metadata dictionary containing information about the example.
        cif_parser_args (dict, optional): Additional arguments for the CIF parser. Defaults to None.

    Returns:
        dict: A dictionary containing the parsed row data and additional loaded CIF data.
    """
    
    # Create atom array for pipeline
    atom_array = create_atom_array_from_design_specification(cif_parser_args=cif_parser_args, **metadata_dict)
    
    # Spoof a file and reload for consistency with inference

    extra_fields = list(set(cif_parser_args.get('extra_fields', []) + Mask.required_annotations))
    tmp_file = tempfile.NamedTemporaryFile(suffix='.cif', delete=False)
    to_cif_file(atom_array, tmp_file.name, id=metadata_dict.get('example_id', 'unknown'), 
        extra_fields=extra_fields,)
    data = load_(tmp_file.name, cif_parser_args=cif_parser_args)
    
    data = metadata_dict | data
    return data
