import os
import socket
import time
from pathlib import Path
from typing import Any

import numpy as np
from atomworks.common import exists
from atomworks.enums import ChainType
from atomworks.ml.datasets import StructuralDatasetWrapper, logger
from atomworks.ml.datasets.parsers import (
    MetadataRowParser,
    load_example_from_metadata_row,
)
from atomworks.ml.transforms._checks import (
    check_contains_keys,
    check_is_instance,
    check_nonzero_length,
)
from atomworks.ml.transforms.base import Transform, TransformedDict
from atomworks.ml.transforms.msa._msa_loading_utils import load_msa_data_from_path
from atomworks.ml.utils.rng import capture_rng_states
from biotite.structure import AtomArray, concatenate


# input data wrapper that allows multiple input files separated by ':'
#   data is loaded as concatentation of all inputs
class MultiInputDatasetWrapper(StructuralDatasetWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx: int) -> Any:
        # Capture example ID & current rng state (for reproducibility & debugging)
        if hasattr(self, "idx_to_id"):
            # ...if the dataset has a custom idx_to_id method, use it (e.g., for a PandasDataset)
            example_id = self.idx_to_id(idx)
        else:
            # ...otherwise, fallback to a the `id_column` or a string representation of the index
            example_id = (
                self.dataset[idx][self.id_column] if self.id_column else f"row_{idx}"
            )

        # Get process id and hostname (for debugging)
        logger.debug(
            f"({socket.gethostname()}:{os.getpid()}) Processing example ID: {example_id}"
        )

        # Load the row, using the __getitem__ method of the dataset
        row = self.dataset[idx]
        pdb_path = row["pdb_path"].split(":")

        # Process the row into a transform-ready dictionary with the given CIF and dataset parsers
        # We require the "data" dictionary output from `load_example_from_metadata_row` to contain, at a minimum:
        #   (a) An "id" key, which uniquely identifies the example within the dataframe; and,
        #   (b) The "path" key, which is the path to the CIF file
        _start_parse_time = time.time()
        data = None
        assert len(pdb_path) <= 2

        for pdb_i in pdb_path:
            row_i = {"example_id": row["example_id"], "path": pdb_i}
            data_i = load_example_from_metadata_row(
                row_i, self.dataset_parser, cif_parser_args=self.cif_parser_args
            )

            if data is None:
                data = data_i
            else:
                data_i["atom_array"].pn_unit_id = np.full(
                    len(data_i["atom_array"]), "B_1"
                )  # unique pn unit id
                data_i["atom_array"].pn_unit_iid = np.full(
                    len(data_i["atom_array"]), "B_1"
                )  # unique pn unit iid
                data_i["atom_array"].chain_id = np.full(
                    len(data_i["atom_array"]), "B"
                )  # unique chain id
                data_i["atom_array"].chain_iid = np.full(
                    len(data_i["atom_array"]), "B"
                )  # unique chain iid
                data["atom_array"] = concatenate(
                    [data["atom_array"], data_i["atom_array"]]
                )
                data["atom_array_stack"] = concatenate(
                    [data["atom_array_stack"], data_i["atom_array_stack"]]
                )
                data["chain_info"]["B"] = data_i["chain_info"]["A"]

        # 'example_id', 'path', 'assembly_id', 'query_pn_unit_iids',
        data["path"] = row["pdb_path"]
        data["msa_path"] = Path(row["msa_path"])  # save msa
        _stop_parse_time = time.time()

        # Manually add timing for cif-parsing
        data = TransformedDict(data)
        data.__transform_history__.append(
            dict(
                name="load_example_from_metadata_row",
                instance=hex(id(load_example_from_metadata_row)),
                start_time=_start_parse_time,
                end_time=_stop_parse_time,
                processing_time=_stop_parse_time - _start_parse_time,
            )
        )

        # Apply the transformation pipeline to the data
        if exists(self.transform):
            try:
                rng_state_dict = capture_rng_states(include_cuda=False)
                data = self.transform(data)
            except KeyboardInterrupt as e:
                raise e
            except Exception as e:
                # Log the error and save the failed example to disk (optional)
                logger.info(f"Error processing row {idx} ({example_id}): {e}")

                if exists(self.save_failed_examples_to_dir):
                    save_failed_example_to_disk(
                        example_id=example_id,
                        error_msg=e,
                        rng_state_dict=rng_state_dict,
                        data={},  # We do not save the data, since it may be large.
                        fail_dir=self.save_failed_examples_to_dir,
                    )
                raise e

        return data


class MultidomainDFParser(MetadataRowParser):
    """Parser for Qian's multidomain data"""

    def __init__(
        self,
        example_id_colname: str = "example_id",
        path_colname: str = "path",
    ):
        self.example_id_colname = example_id_colname
        self.path_colname = path_colname

    def _parse(self, row: dict) -> dict[str, Any]:
        query_pn_unit_iids = None
        assembly_id = "1"

        return {
            "example_id": row[self.example_id_colname],
            "path": Path(row[self.path_colname]),
            "assembly_id": assembly_id,
            "query_pn_unit_iids": query_pn_unit_iids,
            "extra_info": row,
        }


class LoadPairedMSAs(Transform):
    """
    LoadPairedMSAs adds paired MSAs from disk, overwriting previously paired MSA data.
    """

    def check_input(self, data: dict[str, Any]):
        check_contains_keys(data, ["atom_array", "msa_path"])
        check_is_instance(data, "atom_array", AtomArray)
        check_nonzero_length(data, "atom_array")

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array = data["atom_array"]
        msa_file_path = data["msa_path"]
        chain_type = data["chain_info"]["A"]["chain_type"]
        max_msa_sequences = 10000

        msa_data = load_msa_data_from_path(
            msa_file_path=msa_file_path,
            chain_type=chain_type,
            max_msa_sequences=max_msa_sequences,
        )

        # split into chains
        start_idx = 0
        allpolymerchains = np.unique(
            atom_array.chain_id[
                np.isin(atom_array.chain_type, ChainType.get_polymers())
            ]
        )

        data["polymer_msas_by_chain_id"] = {}  # nuke old version
        for chain_id in allpolymerchains:
            sequence = data["chain_info"][chain_id][
                "processed_entity_non_canonical_sequence"
            ]
            stop_idx = start_idx + len(sequence)

            data["polymer_msas_by_chain_id"][chain_id] = {}

            # trim all msa info to this chain only
            for mkey in msa_data.keys():
                data["polymer_msas_by_chain_id"][chain_id][mkey] = msa_data[mkey][
                    ..., start_idx:stop_idx
                ]

            # mock msa_is_padded_mask (all 0s)
            data["polymer_msas_by_chain_id"][chain_id]["msa_is_padded_mask"] = np.zeros(
                data["polymer_msas_by_chain_id"][chain_id]["msa"].shape, dtype=bool
            )

            start_idx = stop_idx

        return data
