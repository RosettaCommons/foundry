import tempfile
from os import PathLike
from pathlib import Path

import hydra
import numpy as np
import pytest
from hydra import compose, initialize

from atomworks.io import parse
from modelhub.utils.inference import (
    apply_conformer_and_template_selections,
    build_file_paths_for_prediction,
)

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
@pytest.mark.parametrize("template_selection", ["A"])
@pytest.mark.parametrize("ground_truth_conformer_selection", ["*/PBF"])
@pytest.mark.slow
@pytest.mark.skip(reason="TEST STILL BROKEN")
def test_inference_engine(
    inference_engine: Path,
    inputs: PathLike,
    template_selection: str,
    ground_truth_conformer_selection: str,
):
    # TODO: TEST STILL BROKEN
    with initialize(config_path="../configs"):
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

    atom_array_untemplated = apply_conformer_and_template_selections(atom_array)
    assert (
        "is_input_file_templated" in atom_array_untemplated.get_annotation_categories()
    )
    assert np.sum(atom_array_untemplated.get_annotation("is_input_file_templated")) == 0

    atom_array_templated = apply_conformer_and_template_selections(
        atom_array,
        template_selection=template_selection,
        ground_truth_conformer_selection=ground_truth_conformer_selection,
    )
    assert "is_input_file_templated" in atom_array_templated.get_annotation_categories()
    assert np.sum(atom_array_templated.get_annotation("is_input_file_templated")) > 0

    # TODO: Make this actually test the ground truth conformer policy; make template selection actuall work;
    # also dont rely on the is_input_file_templated annotation instead just handle the case where it doesnt exist
    # correctly
