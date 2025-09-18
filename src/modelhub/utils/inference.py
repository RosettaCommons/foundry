import json
import logging
import pickle
from os import PathLike
from pathlib import Path
from typing import Iterable

import numpy as np
from atomworks.common import as_list
from atomworks.enums import GroundTruthConformerPolicy
from atomworks.io.tools.inference import (
    build_msa_paths_by_chain_id_from_component_list,
    components_to_atom_array,
)
from atomworks.io.utils.io_utils import to_cif_file
from atomworks.io.utils.selection import AtomSelectionStack
from biotite.structure import AtomArray

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

    if item.get("msa_paths") and isinstance(item.get("msa_paths"), dict):
        for chain_id, msa_path in item.get("msa_paths").items():
            msa_paths_by_chain_id[chain_id] = msa_path

    extra_categories = {}
    if item.get("template_selection"):
        extra_categories["template_selection"] = {
            "template_selection": item.get("template_selection"),
        }
    if item.get("ground_truth_conformer_selection"):
        extra_categories["ground_truth_conformer_selection"] = {
            "ground_truth_conformer_selection": item.get(
                "ground_truth_conformer_selection"
            ),
        }

    if msa_paths_by_chain_id:
        extra_categories["msa_paths_by_chain_id"] = msa_paths_by_chain_id

    # Create a temporary CIF file from the JSON data
    cif_path = Path(temp_dir) / f"{item['name']}.cif"
    save_path = to_cif_file(
        atom_array,
        cif_path,
        extra_categories=extra_categories,
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


def apply_atom_selection_mask(
    atom_array: AtomArray, selection_list: Iterable[str]
) -> np.ndarray:
    """Return a combined boolean mask for a list of AtomSelectionStack queries.

    Args:
        atom_array: AtomArray to select from.
        selection_list: Iterable of AtomSelectionStack queries (e.g., "*/LIG", "A1-10").

    Returns:
        A boolean numpy array of shape (num_atoms,) where True indicates a selected atom.
    """
    selection_mask = np.zeros(len(atom_array), dtype=bool)
    for selection in selection_list:
        if not selection:
            continue
        try:
            selector = AtomSelectionStack.from_query(selection)
            mask = selector.get_mask(atom_array)
            selection_mask = selection_mask | mask
        except Exception as exc:  # Defensive: keep going if one selection fails
            logging.warning(
                "Failed to parse selection '%s': %s. Skipping.", selection, exc
            )
    return selection_mask


def apply_template_selection(
    atom_array: AtomArray, template_selection: list[str] | str | None
) -> AtomArray:
    """Apply token-level template selection to `atom_array` with OR semantics.

    If the `is_input_file_templated` annotation already exists, this function ORs
    the new selection with the existing annotation. Otherwise, it creates it.

    Args:
        atom_array: AtomArray to annotate.
        template_selection: Selection string(s). Single strings are converted to lists. If None/empty, no-op.

    Returns:
        The same AtomArray with `is_input_file_templated` updated.
    """
    # Convert to list if needed
    template_selection_list = as_list(template_selection) if template_selection else []

    if not template_selection_list:
        # Ensure the annotation exists even if no selection provided
        if "is_input_file_templated" not in atom_array.get_annotation_categories():
            atom_array.set_annotation(
                "is_input_file_templated", np.zeros(len(atom_array), dtype=bool)
            )
        return atom_array

    # Build new mask
    selection_mask = apply_atom_selection_mask(atom_array, template_selection_list)
    logging.info(
        "Selected %d atoms for token-level templating with %d syntaxes",
        int(np.sum(selection_mask)),
        len([s for s in template_selection_list if s]),
    )

    # OR with existing annotation if present
    if "is_input_file_templated" in atom_array.get_annotation_categories():
        existing = atom_array.get_annotation("is_input_file_templated").astype(bool)
        selection_mask = existing | selection_mask
    atom_array.set_annotation("is_input_file_templated", selection_mask)
    return atom_array


def apply_ground_truth_conformer_selection(
    atom_array: AtomArray, ground_truth_conformer_selection: list[str] | str | None
) -> AtomArray:
    """Apply ground-truth conformer policy selection with union semantics.

    Behavior:
    - Creates `ground_truth_conformer_policy` if missing and initializes to IGNORE.
    - For selected atoms, sets policy to at least ADD without downgrading any
      existing policy (e.g., preserves REPLACE if present).

    Args:
        atom_array: AtomArray to annotate.
        ground_truth_conformer_selection: Selection string(s). Single strings are converted to lists. If None/empty, no-op.

    Returns:
        The same AtomArray with `ground_truth_conformer_policy` updated.
    """
    # Convert to list if needed
    ground_truth_conformer_selection_list = (
        as_list(ground_truth_conformer_selection)
        if ground_truth_conformer_selection
        else []
    )

    if not ground_truth_conformer_selection_list:
        if (
            "ground_truth_conformer_policy"
            not in atom_array.get_annotation_categories()
        ):
            atom_array.set_annotation(
                "ground_truth_conformer_policy",
                np.full(
                    len(atom_array), GroundTruthConformerPolicy.IGNORE, dtype=np.int8
                ),
            )
        return atom_array

    # Ensure annotation exists
    if "ground_truth_conformer_policy" not in atom_array.get_annotation_categories():
        atom_array.set_annotation(
            "ground_truth_conformer_policy",
            np.full(len(atom_array), GroundTruthConformerPolicy.IGNORE, dtype=np.int8),
        )

    selection_mask = apply_atom_selection_mask(
        atom_array, ground_truth_conformer_selection_list
    )
    logging.info(
        "Selected %d atoms for ground-truth conformer policy with %d syntaxes",
        int(np.sum(selection_mask)),
        len([s for s in ground_truth_conformer_selection_list if s]),
    )

    existing = atom_array.get_annotation("ground_truth_conformer_policy")
    existing[selection_mask] = GroundTruthConformerPolicy.ADD
    atom_array.set_annotation("ground_truth_conformer_policy", existing)

    return atom_array


def apply_conformer_and_template_selections(
    atom_array: AtomArray,
    template_selection: list[str] | str | None = None,
    ground_truth_conformer_selection: list[str] | str | None = None,
) -> AtomArray:
    """Apply template and conformer selections and basic preprocessing.

    This function replaces the former class method `prepare_atom_array`.

    - Applies `apply_template_selection` then `apply_ground_truth_conformer_selection`.
    - Replaces NaN coordinates with -1 for safety.

    Args:
        atom_array: AtomArray to prepare.
        template_selection: Template selection string(s). Single strings are converted to lists.
        ground_truth_conformer_selection: Ground-truth conformer selection string(s). Single strings are converted to lists.

    Returns:
        The same AtomArray with `is_input_file_templated` and `ground_truth_conformer_policy` updated.
    """
    atom_array = apply_template_selection(atom_array, template_selection)
    atom_array = apply_ground_truth_conformer_selection(
        atom_array, ground_truth_conformer_selection
    )
    # Safety: avoid unexpected behavior downstream
    atom_array.coord[np.isnan(atom_array.coord)] = -1
    return atom_array
