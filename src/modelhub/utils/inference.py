import json
import pickle
from os import PathLike
from pathlib import Path

from cifutils.tools.inference import (
    build_msa_paths_by_chain_id_from_component_list,
    components_to_atom_array,
)
from cifutils.utils.io_utils import to_cif_file

from modelhub.utils.io import (
    CIF_LIKE_EXTENSIONS,
    DICTIONARY_LIKE_EXTENSIONS,
    create_example_id_extractor,
    find_files_with_extension,
)


def _spoof_cif_from_dictionary(item: dict, temp_dir: PathLike) -> Path:
    """Unpacks a dictionary to create a CIF file from its components.

    Args:
        item (dict): A dictionary containing 'name' and either 'components' or 'sequences', optionally 'bonds'.
        temp_dir (Path): Path to the temporary directory for storing CIF files.

    Returns:
        Path: The path to the created CIF file, saved in the temporary directory.

    Raises:
        ValueError: If 'name' or neither 'components' nor 'sequences' are present in the dictionary.
    """
    # Validate the dictionary structure ("name" is required, either "components" or "sequences" is required)
    assert "name" in item, "The input dictionary must contain a 'name' key."
    assert (
        "components" in item or "sequences" in item
    ), "The input dictionary must contain either 'components' or 'sequences' keys."

    # Use sequences if components not present
    if "components" not in item and "sequences" in item:
        # Rename sequences to components
        item["components"] = [{"sequence": seq} for seq in item.pop("sequences")]

    # Build components
    atom_array, component_list = components_to_atom_array(
        item["components"], return_components=True, bonds=item.get("bonds", None)
    )
    msa_paths_by_chain_id = build_msa_paths_by_chain_id_from_component_list(
        component_list
    )

    # Create a temporary CIF file from the JSON data
    cif_path = Path(temp_dir) / f"{item['name']}.cif"
    save_path = to_cif_file(
        atom_array,
        cif_path,
        extra_categories={"msa_paths_by_chain_id": msa_paths_by_chain_id}
        if msa_paths_by_chain_id
        else None,
        file_type="cif",  # Not zipped for efficiency (as it's a temporary directory anyways)
    )

    return Path(save_path)


def build_file_paths_for_prediction(
    input: PathLike | list[PathLike],
    temp_dir: PathLike,
    existing_outputs_dir: PathLike | None = None,
) -> list[Path]:
    """Prepare files for prediction based on the input paths.

    Input path may be dictionary-like format (e.g., JSON, YAML, Pickle), CIF/PDB files, or a directory containing these files.
    Processes directories to find supported file types and converts dictionary-like formats to CIF files.

    Args:
        input (PathLike): Input paths (JSON, YAML, Pickle, or CIF/PDB) or a directory containing these files.
        temp_dir (Path): Path to the temporary directory for storing CIF files.
        existing_outputs_dir(Path): Directory for existing outputs (optional). If provided, we not predict files with matching example_ids.

    Returns:
        list[Path]: List of file paths for prediction.
    """
    # Collect all files from inputs, handling directories, individual files, and lists of directories/files
    input_paths = [input] if not isinstance(input, list) else input

    example_id_extractor = create_example_id_extractor(CIF_LIKE_EXTENSIONS)

    existing_example_ids = None
    if existing_outputs_dir:
        existing_example_ids = set(
            example_id_extractor(path)
            for path in find_files_with_extension(
                existing_outputs_dir, CIF_LIKE_EXTENSIONS
            )
        )

    paths_to_raw_input_files = []
    for _path in input_paths:
        if Path(_path).is_dir():
            paths_to_raw_input_files.extend(
                find_files_with_extension(
                    _path, DICTIONARY_LIKE_EXTENSIONS | CIF_LIKE_EXTENSIONS
                )
            )
        else:
            paths_to_raw_input_files.append(Path(_path))

    paths_to_cif_like_files = []
    for _path in paths_to_raw_input_files:
        if _path.name.endswith(tuple(DICTIONARY_LIKE_EXTENSIONS)):
            # Spoof CIF files from dictionary-like formats
            with open(_path, "rb" if _path.suffix == ".pkl" else "r") as file:
                # Load data based on file extension
                if _path.suffix == ".json":
                    data = json.load(file)
                elif _path.suffix in {".yaml", ".yml"}:
                    raise NotImplementedError("YAML files are not yet supported.")
                elif _path.suffix == ".pkl":
                    data = pickle.load(file)

                if isinstance(data, dict):
                    data = [
                        data
                    ]  # Convert single dictionary to list for uniform processing

                for item in data:
                    paths_to_cif_like_files.append(
                        _spoof_cif_from_dictionary(item, temp_dir)
                    )
        elif _path.name.endswith(tuple(CIF_LIKE_EXTENSIONS)):
            # Directly use CIF-like files
            paths_to_cif_like_files.append(_path)
        else:
            raise ValueError(
                f"Unsupported file extension: {_path.suffix} (path: {_path}; paths: {paths_to_raw_input_files})."
            )

    # Filter out existing example_ids if provided
    if existing_example_ids:
        paths_to_cif_like_files = [
            path
            for path in paths_to_cif_like_files
            if example_id_extractor(path) not in existing_example_ids
        ]

    return paths_to_cif_like_files
