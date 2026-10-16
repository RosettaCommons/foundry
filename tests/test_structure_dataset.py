"""Exercise the configuration adapter with real CIF loading and metadata."""

from pathlib import Path

import pandas as pd
import pytest
from atomworks.ml.datasets import PandasDataset
from atomworks.ml.datasets.parsers import (
    GenericDFParser,
    InterfacesDFParser,
    PNUnitsDFParser,
)

from foundry.datasets.structure import StructuralDatasetWrapper


@pytest.mark.parametrize("kind", ["generic", "pn_unit", "interface"])
def test_structure_dataset_loads_real_cif_and_preserves_metadata(kind):
    path = Path(__file__).parents[1] / "models/rf3/tests/data/glke.cif"
    row = {
        "example_id": "example",
        "path": str(path),
        "assembly_id": "1",
        "pdb_id": "glke",
        "q_pn_unit_iid": "A_1",
        "pn_unit_1_iid": "A_1",
        "pn_unit_2_iid": "B_1",
        "all_pn_unit_iids_after_processing": ["A_1"],
        "score_tag": "held-out",
    }
    if kind == "generic":
        parser = GenericDFParser(
            pn_unit_iid_colnames="q_pn_unit_iid", assembly_id_colname="assembly_id"
        )
    else:
        cls = PNUnitsDFParser if kind == "pn_unit" else InterfacesDFParser
        parser = cls(
            base_dir=path.parent,
            file_extension=".cif",
            path_template="{base_dir}/{pdb_id}{file_extension}",
        )
    metadata = PandasDataset(data=pd.DataFrame([row]), name="portable")
    dataset = StructuralDatasetWrapper(
        metadata, parser, cif_parser_args={"cache_dir": None}
    )
    item = dataset[0]
    assert len(dataset) == 1
    assert dataset.metadata is metadata.metadata
    assert item["example_id"] == "example"
    assert item["dataset_name"] == "portable"
    assert item["extra_info"]["all_pn_unit_iids_after_processing"] == ["A_1"]
    assert item["extra_info"]["score_tag"] == "held-out"
    assert item["query_pn_unit_iids"] == (
        ["A_1", "B_1"] if kind == "interface" else ["A_1"]
    )
    assert len(item["atom_array"]) > 0
    assert set(item["atom_array"].res_name) == {"GLY", "LEU", "LYS", "GLU"}


def test_structure_dataset_propagates_missing_file():
    metadata = PandasDataset(
        data=pd.DataFrame(
            [{"example_id": "missing", "path": "/nonexistent/foundry-test.cif"}]
        ),
        name="portable",
    )
    dataset = StructuralDatasetWrapper(metadata, GenericDFParser())
    with pytest.raises(FileNotFoundError):
        dataset[0]
