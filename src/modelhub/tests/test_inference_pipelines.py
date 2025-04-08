from os import PathLike
from pathlib import Path

import pytest
from cifutils import parse

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
    
