import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from atomworks.common import as_list
from atomworks.enums import GroundTruthConformerPolicy
from atomworks.io import parse
from atomworks.io.parser import parse_atom_array
from atomworks.io.tools.inference import (
    build_msa_paths_by_chain_id_from_component_list,
    components_to_atom_array,
)
from atomworks.io.transforms.categories import category_to_dict
from atomworks.io.utils.selection import AtomSelectionStack
from atomworks.ml.transforms.atom_array import add_global_token_id_annotation
from biotite.structure import AtomArray
from rf3.utils.io import (
    CIF_LIKE_EXTENSIONS,
    DICTIONARY_LIKE_EXTENSIONS,
    get_sharded_output_path,
)
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def _resolve_override(override_value, source_value, param_name: str, example_id: str):
    """Resolve CLI override vs source value with warning."""
    if override_value is not None and source_value:
        logger.warning(f"CLI {param_name} overriding source value for {example_id}")
        return override_value
    return override_value if override_value is not None else source_value


def extract_example_id_from_path(path: Path) -> str:
    """Extract example ID from file path."""
    path_str = str(path.name)
    # Check for known extensions (longer matches first to handle .cif.gz before .gz)
    for ext in sorted(CIF_LIKE_EXTENSIONS | {".json"}, key=len, reverse=True):
        if path_str.endswith(ext):
            return path_str[: -len(ext)]
    # Fallback to simple stem
    return path.stem


def extract_example_ids_from_json(path: Path) -> list[str]:
    """Extract example IDs from a JSON file containing one or more examples."""
    with open(path, "r") as f:
        data = json.load(f)
    return [ex["name"] for ex in data]


@dataclass
class InferenceInput:
    """Input specification for RF3 inference."""

    atom_array: AtomArray
    chain_info: dict
    example_id: str
    template_selection: list[str] | None = None
    ground_truth_conformer_selection: list[str] | None = None
    cyclic_chains: list[str] | None = None

    @classmethod
    def from_cif_path(
        cls,
        path: PathLike,
        example_id: str | None = None,
        template_selection: list[str] | str | None = None,
        ground_truth_conformer_selection: list[str] | str | None = None,
    ) -> "InferenceInput":
        """Load from CIF/PDB file.

        Args:
          path: Path to CIF/PDB file.
          example_id: Example ID. Defaults to filename stem.
          template_selection: Template selection override.
          ground_truth_conformer_selection: Conformer selection override.

        Returns:
          InferenceInput object.
        """
        parsed = parse(path, hydrogen_policy="remove", keep_cif_block=True)

        atom_array = (
            parsed["assemblies"]["1"][0]
            if "assemblies" in parsed
            else parsed["asym_unit"][0]
        )

        example_id = example_id or extract_example_id_from_path(Path(path))

        # Extract from CIF
        cif_template_sel = None
        cif_conformer_sel = None
        if "cif_block" in parsed:
            template_dict = category_to_dict(parsed["cif_block"], "template_selection")
            if template_dict:
                cif_template_sel = list(template_dict.get("template_selection", []))

            conformer_dict = category_to_dict(
                parsed["cif_block"], "ground_truth_conformer_selection"
            )
            if conformer_dict:
                cif_conformer_sel = list(
                    conformer_dict.get("ground_truth_conformer_selection", [])
                )

        # Resolve overrides (CLI priority)
        final_template_sel = _resolve_override(
            template_selection, cif_template_sel, "template_selection", example_id
        )
        final_conformer_sel = _resolve_override(
            ground_truth_conformer_selection,
            cif_conformer_sel,
            "ground_truth_conformer_selection",
            example_id,
        )

        return cls(
            atom_array=atom_array,
            chain_info=parsed["chain_info"],
            example_id=example_id,
            template_selection=final_template_sel,
            ground_truth_conformer_selection=final_conformer_sel,
        )

    @classmethod
    def from_json_dict(
        cls,
        data: dict,
        template_selection: list[str] | str | None = None,
        ground_truth_conformer_selection: list[str] | str | None = None,
    ) -> "InferenceInput":
        """Create from JSON dict with components.

        CLI args override JSON metadata.

        Args:
          data: JSON dictionary with components.
          template_selection: Template selection override.
          ground_truth_conformer_selection: Conformer selection override.

        Returns:
          InferenceInput object.
        """
        # Build atom_array from components
        atom_array, component_list = components_to_atom_array(
            data["components"],
            bonds=data.get("bonds"),
            return_components=True,
        )

        parsed = parse_atom_array(
            atom_array,
            build_assembly="_spoof",
            hydrogen_policy="keep",
        )

        chain_info = parsed.get("chain_info", {})
        atom_array = (
            parsed["assemblies"]["1"][0]
            if "assemblies" in parsed
            else parsed["asym_unit"][0]
        )

        # Merge MSA paths into chain_info
        msa_paths_by_chain_id = build_msa_paths_by_chain_id_from_component_list(
            component_list
        )
        if data.get("msa_paths") and isinstance(data.get("msa_paths"), dict):
            msa_paths_by_chain_id.update(data.get("msa_paths"))

        for chain_id, msa_path in msa_paths_by_chain_id.items():
            if chain_id in chain_info:
                chain_info[chain_id]["msa_path"] = msa_path

        # Resolve overrides (CLI priority)
        final_template_sel = _resolve_override(
            template_selection,
            data.get("template_selection"),
            "template_selection",
            data["name"],
        )
        final_conformer_sel = _resolve_override(
            ground_truth_conformer_selection,
            data.get("ground_truth_conformer_selection"),
            "ground_truth_conformer_selection",
            data["name"],
        )

        return cls(
            atom_array=atom_array,
            chain_info=chain_info,
            example_id=data["name"],
            template_selection=final_template_sel,
            ground_truth_conformer_selection=final_conformer_sel,
        )

    @classmethod
    def from_atom_array(
        cls,
        atom_array: AtomArray,
        chain_info: dict | None = None,
        example_id: str | None = None,
        template_selection: list[str] | str | None = None,
        ground_truth_conformer_selection: list[str] | str | None = None,
    ) -> "InferenceInput":
        """Create from AtomArray.

        Args:
          atom_array: Input AtomArray.
          chain_info: Chain info dict. Defaults to extracted from atom_array.
          example_id: Example ID. Defaults to generated ID.
          template_selection: Template selection.
          ground_truth_conformer_selection: Conformer selection.

        Returns:
          InferenceInput object.
        """
        # Use parse_atom_array
        parsed = parse_atom_array(
            atom_array,
            build_assembly="_spoof",
            hydrogen_policy="keep",
            extra_fields="all",
        )

        extracted_chain_info = parsed.get("chain_info", {})

        # Merge with provided chain_info (provided takes priority)
        if chain_info is not None:
            for chain_id, chain_data in chain_info.items():
                if chain_id in extracted_chain_info:
                    extracted_chain_info[chain_id].update(chain_data)
                else:
                    extracted_chain_info[chain_id] = chain_data

        final_atom_array = (
            parsed["assemblies"]["1"][0]
            if "assemblies" in parsed
            else parsed["asym_unit"][0]
        )

        return cls(
            atom_array=final_atom_array,
            chain_info=extracted_chain_info,
            example_id=example_id or f"inference_{id(atom_array)}",
            template_selection=template_selection,
            ground_truth_conformer_selection=ground_truth_conformer_selection,
        )

    def to_pipeline_input(self) -> dict:
        """Apply transformations and return input for Transform pipeline.

        Returns:
          Pipeline input dict with example_id, atom_array, and chain_info.
        """
        atom_array = self.atom_array.copy()

        # Apply template and conformer selections
        atom_array = apply_conformer_and_template_selections(
            atom_array,
            template_selection=self.template_selection,
            ground_truth_conformer_selection=self.ground_truth_conformer_selection,
        )

        if self.cyclic_chains:
            atom_array = cyclize_atom_array(atom_array, self.cyclic_chains)

        return {
            "example_id": self.example_id,
            "atom_array": atom_array,
            "chain_info": self.chain_info,
        }


def _process_single_path(
    path: Path,
    existing_outputs_dir: Path | None,
    sharding_pattern: str | None,
    template_selection: list[str] | str | None,
    ground_truth_conformer_selection: list[str] | str | None,
) -> list[InferenceInput]:
    """Worker function to process a single input file path.

    This function is defined at module level to be picklable for multiprocessing.

    Args:
      path: Path to a single input file.
      existing_outputs_dir: If set, skip examples with existing outputs.
      sharding_pattern: Sharding pattern for output paths.
      template_selection: Override for template selection.
      ground_truth_conformer_selection: Override for conformer selection.

    Returns:
      List of InferenceInput objects (may be empty if file is skipped).
    """

    def example_exists(example_id: str) -> bool:
        """Check if example already has predictions (sharding-aware)."""
        if not existing_outputs_dir:
            return False
        example_dir = get_sharded_output_path(
            example_id, existing_outputs_dir, sharding_pattern
        )
        return (example_dir / f"{example_id}_metrics.csv").exists()

    inference_inputs = []

    if path.suffix == ".json":
        # Load JSON and convert each entry
        with open(path, "r") as f:
            data = json.load(f)

        # Normalize to list
        if isinstance(data, dict):
            data = [data]

        for item in data:
            example_id = item["name"]
            if not example_exists(example_id):
                inference_inputs.append(
                    InferenceInput.from_json_dict(
                        item,
                        template_selection=template_selection,
                        ground_truth_conformer_selection=ground_truth_conformer_selection,
                    )
                )

    elif any(path.name.endswith(ext) for ext in CIF_LIKE_EXTENSIONS):
        # CIF/PDB file
        example_id = extract_example_id_from_path(path)
        if not example_exists(example_id):
            inference_inputs.append(
                InferenceInput.from_cif_path(
                    path,
                    example_id=example_id,
                    template_selection=template_selection,
                    ground_truth_conformer_selection=ground_truth_conformer_selection,
                )
            )
    else:
        raise ValueError(
            f"Unsupported file type: {path.suffix} (path: {path}). "
            f"Supported: {CIF_LIKE_EXTENSIONS | DICTIONARY_LIKE_EXTENSIONS}"
        )

    return inference_inputs


def prepare_inference_inputs_from_paths(
    inputs: PathLike | list[PathLike],
    existing_outputs_dir: PathLike | None = None,
    sharding_pattern: str | None = None,
    template_selection: list[str] | str | None = None,
    ground_truth_conformer_selection: list[str] | str | None = None,
) -> list[InferenceInput]:
    """Load InferenceInput objects from file paths.

    Handles CIF, PDB, and JSON files. Filters out existing outputs if requested.
    Uses multiprocessing to parallelize file loading across all available CPUs.

    Args:
      inputs: File path(s) or directory path(s).
      existing_outputs_dir: If set, skip examples with existing outputs.
      sharding_pattern: Sharding pattern for output paths.
      template_selection: Override for template selection (applied to all inputs).
      ground_truth_conformer_selection: Override for conformer selection (applied to all inputs).

    Returns:
      List of InferenceInput objects.
    """
    input_paths = as_list(inputs)

    # Collect all raw input files (reusing logic from build_file_paths_for_prediction)
    paths_to_raw_input_files = []
    for _path in input_paths:
        if Path(_path).is_dir():
            # Scan directory for supported file types (JSON + CIF-like)
            for file_type in CIF_LIKE_EXTENSIONS | DICTIONARY_LIKE_EXTENSIONS:
                paths_to_raw_input_files.extend(Path(_path).glob(f"*{file_type}"))
        else:
            paths_to_raw_input_files.append(Path(_path))

    # Determine number of CPUs to use
    num_cpus = min(os.cpu_count() or 1, len(paths_to_raw_input_files))
    logger.info(
        f"Processing {len(paths_to_raw_input_files)} files using {num_cpus} CPUs"
    )

    # Convert existing_outputs_dir to Path if needed
    existing_outputs_dir_path = (
        Path(existing_outputs_dir) if existing_outputs_dir else None
    )

    # Process files in parallel using all available CPUs
    inference_inputs = []
    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        # Submit all tasks
        futures = [
            executor.submit(
                _process_single_path,
                path,
                existing_outputs_dir_path,
                sharding_pattern,
                template_selection,
                ground_truth_conformer_selection,
            )
            for path in paths_to_raw_input_files
        ]

        # Collect results as they complete
        for future in futures:
            result = future.result()
            inference_inputs.extend(result)

    logger.info(f"Loaded {len(inference_inputs)} inference inputs")
    return inference_inputs


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


def cyclize_atom_array(atom_array: AtomArray, cyclic_chains: list[str]) -> AtomArray:
    """Cyclize the atom array by positioining the termini properly if not already done.

    Behavior:
    - Positions the last carbon atom in the chain to be 1.3 Angstroms away from the first nitrogen atom if they are not already close.
    - Adds a bond between the termini for proper cif output.

    Args:
        atom_array: AtomArray to cyclize.
        cyclic_chains: List of chain IDs to cyclize.

    Returns:
        The same AtomArray with the specified chains cyclized.
    """
    for chain in cyclic_chains:
        # Find the first nitrogen atom in the chain
        nitrogen_mask = (atom_array.chain_id == chain) & (atom_array.atom_name == "N")
        nitrogen_mask_indices = np.where(nitrogen_mask)[0]
        first_nitrogen_index = nitrogen_mask_indices[0]
        nitrogen_coord = atom_array.coord[first_nitrogen_index]

        # move the last carbon atom in the chain to be 1.3 Angstroms away from the nitrogen
        carbon_mask = (atom_array.chain_id == chain) & (atom_array.atom_name == "C")
        carbon_mask_indices = np.where(carbon_mask)[0]
        last_carbon_index = carbon_mask_indices[-1]
        # check if the last carbon is already close to the nitrogen
        termini_distance = np.linalg.norm(
            atom_array.coord[last_carbon_index] - nitrogen_coord
        )
        if not (termini_distance < 1.5 and termini_distance > 0.5):
            atom_array.coord[last_carbon_index] = nitrogen_coord + np.array(
                [1.3, 0.0, 0.0]
            )

        # add a bond between the nitrogen and carbon so output cif has a connection
        atom_array.bonds.add_bond(first_nitrogen_index, last_carbon_index)
        atom_array.bonds.add_bond(last_carbon_index, first_nitrogen_index)

    return atom_array


class InferenceInputDataset(Dataset):
    """
    Dataset for inference inputs. Also has a length key telling you the number of tokens in each example for LoadBalancedDistributedSampler.

    To calculate the length of each example, we need to add the token_id annotation to the atom_array. If it doesn't exist yet, we add it,
    calculate the length, and then remove it since the downstream pipeline may not be expecting it. That means the num_tokens key may not ultimately
    be the same as what's actually used in the model, but this is a close enough approximation for load balancing.

    Args:
      inference_inputs: List of InferenceInput objects to wrap in a Dataset.
    """

    def __init__(self, inference_inputs: list[InferenceInput]):
        self.inference_inputs = inference_inputs
        self.key_to_balance = "num_tokens_approximate"

        # LoadBalancedDistributedSampler checks in dataset.data[key_to_balance] to determine balancing.
        # That means we need to make a dataframe in self.data that has a column with the key_to_balance.
        atom_array_token_lens = []
        for inf_input in self.inference_inputs:
            if "token_id" not in inf_input.atom_array.get_annotation_categories():
                inf_input.atom_array = add_global_token_id_annotation(
                    inf_input.atom_array
                )
                num_tokens = len(np.unique(inf_input.atom_array.token_id))

                # remove the token_id annotation since the pipeline may not be expecting it
                inf_input.atom_array.del_annotation("token_id")
            else:
                num_tokens = len(np.unique(inf_input.atom_array.token_id))
            atom_array_token_lens.append(num_tokens)
        self.data = pd.DataFrame({self.key_to_balance: atom_array_token_lens})

    def __len__(self):
        return len(self.inference_inputs)

    def __getitem__(self, idx):
        return self.inference_inputs[idx]
