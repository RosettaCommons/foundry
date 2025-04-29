import tempfile
from os import PathLike
from pathlib import Path

import hydra
import numpy as np
import pytest
from cifutils import parse
from hydra import compose, initialize

from modelhub.utils.inference import build_file_paths_for_prediction

current_file_directory = Path(__file__).parent


@pytest.mark.parametrize(
    "file_path",
    [
        "data/nested_examples",
        "data/multiple_examples_from_json.json",
    ],
)
def test_build_file_paths_for_prediction(file_path: PathLike, tmp_path: Path):
    """Use the inference pipeline to build and parse inputs for prediction."""
    file_path = current_file_directory / Path(file_path)

    # Call the function with the file path and temporary directory
    paths = build_file_paths_for_prediction(file_path, tmp_path)

    # Iterate over the returned paths and parse them, ensuring the the outputs are reasonable
    for path in paths:
        output = parse(path)
        assert output is not None
        assert len(output["assemblies"]["1"][0]) > 0


@pytest.mark.parametrize(
    "inference_engine",
    ["af3"],
)
@pytest.mark.parametrize(
    "inputs",
    ["tests/data/5vht_from_file.cif"],
)
@pytest.mark.parametrize("template_selection_syntax", ["A1-71"])
@pytest.mark.slow
def test_inference_engine(
    inference_engine: Path, inputs: PathLike, template_selection_syntax: str
):
    INFERENCE_ENGINE_CONFIG = "../configs/"
    with initialize(config_path=INFERENCE_ENGINE_CONFIG):
        cfg = compose(
            config_name="inference",
            overrides=[
                f"inference_engine={inference_engine}",
                f"inputs={inputs}",
            ],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)

        inference_engine = hydra.utils.instantiate(
            cfg, temp_dir=temp_dir, _convert_="partial"
        )
    out = inference_engine.parse_from_path(inputs)
    atom_array = (
        out["assemblies"]["1"][0] if "assemblies" in out else out["asym_unit"][0]
    )
    assert atom_array is not None

    atom_array_untemplated = inference_engine.prepare_atom_array(atom_array)
    assert (
        "is_input_file_templated" in atom_array_untemplated.get_annotation_categories()
    )
    assert np.sum(atom_array_untemplated.get_annotation("is_input_file_templated")) == 0

    atom_array_templated = inference_engine.prepare_atom_array(
        atom_array, template_selection_syntax=template_selection_syntax
    )
    assert "is_input_file_templated" in atom_array_templated.get_annotation_categories()
    assert np.sum(atom_array_templated.get_annotation("is_input_file_templated")) > 0
