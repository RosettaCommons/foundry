import json
import re
from os import PathLike
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from atomworks.io.utils.io_utils import to_cif_file
from biotite.structure import AtomArray, AtomArrayStack, stack

from foundry.utils.alignment import weighted_rigid_align

DICTIONARY_LIKE_EXTENSIONS = {".json", ".yaml", ".yml", ".pkl"}
CIF_LIKE_EXTENSIONS = {".cif", ".pdb", ".bcif", ".cif.gz", ".pdb.gz", ".bcif.gz"}


def dump_structures(
    atom_arrays: AtomArrayStack | list[AtomArray] | AtomArray,
    base_path: PathLike,
    one_model_per_file: bool,
    extra_fields: list[str] | Literal["all"] = [],
    **kwargs,
) -> None:
    """Dump structures to CIF files, given the coordinates and input AtomArray.

    Args:
        atom_arrays (AtomArrayStack | list[AtomArray] | AtomArray): Either an AtomArrayStack, a list of AtomArray objects,
            or a single AtomArray object to be dumped to CIF file(s)
        base_path (PathLike): Base path where the output files will be saved.
        one_model_per_file (bool): Flag to determine if each model should be dumped into a separate file. Has no effect if
            `atom_arrays` is a list of AtomArrays.
        extra_fields (list[str] | Literal["all"]): List of extra fields to include in the CIF file.
    """
    base_path = Path(base_path)
    if one_model_per_file:
        assert (
            isinstance(atom_arrays, AtomArrayStack) or isinstance(atom_arrays, list)
        ), "AtomArrayStack or list of AtomArray required when one_model_per_file is True"
        # One model per file —> loop over the diffusion batch
        for i in range(len(atom_arrays)):
            path = f"{base_path}_model_{i}"
            to_cif_file(
                atom_arrays[i],
                path,
                file_type="cif.gz",
                include_entity_poly=False,
                extra_fields=extra_fields,
                **kwargs,
            )
    else:
        # Include all models in a single CIF file
        to_cif_file(
            atom_arrays,
            base_path,
            file_type="cif.gz",
            include_entity_poly=False,
            extra_fields=extra_fields,
            **kwargs,
        )


def dump_metadata(
    prediction_metadata: dict,
    base_path: PathLike,
    one_model_per_file: bool,
):
    """
    Dump JSONs of prediction metadata to disk.

    Args:
        prediction_metadata (dict): Dictionary containing metadata for the predictions.
            Instantiated after the models' predictions in trainer. Keys are model indices (from 0)
        base_path (PathLike): Base path where the output files will be saved.
        one_model_per_file (bool): If True, save each model's metadata in a separate file.
    """
    if one_model_per_file:
        # One model per file —> loop over the diffusion batch
        for i in prediction_metadata:
            path = f"{base_path}_model_{i}"
            with open(f"{path}.json", "w") as f:
                json.dump(prediction_metadata[i], f, indent=4)
    else:
        # Include all models in a single JSON file
        with open(f"{base_path}.json", "w") as f:
            json.dump(prediction_metadata, f, indent=4)


def dump_trajectories(
    trajectory_list: list[torch.Tensor | np.ndarray],
    atom_array: AtomArray,
    base_path: Path,
    align_structures: bool = True,
    coord_atom_lvl_to_be_noised: torch.Tensor | None = None,
    is_motif_atom_with_fixed_pos: torch.Tensor | None = None,
    max_frames: int = 100,
) -> None:
    """Write denoising trajectories to CIF files.

    Args:
        trajectory_list (List[torch.Tensor]): List of tensors of length n_steps representing the diffusion trajectory at each step.
            Each tensor has shape [D, L, 3], where D is the diffusion batch size and L is the number of atoms.
        atom_array (np.ndarray): Atom array corresponding to the coordinates.
        base_path (Path): Base path where the output files will be saved.
        align_structures (bool): Flag to determine if the structures should be aligned on the final prediction.
            If False, each step may have a different alignment.
    """
    n_steps = len(trajectory_list)

    if align_structures:
        # ... align the trajectories on the last prediction
        w_L = torch.ones(*trajectory_list[0].shape[:2]).to(trajectory_list[0].device)
        X_exists_L = torch.ones(trajectory_list[0].shape[1], dtype=torch.bool).to(
            trajectory_list[0].device
        )
        for step in range(n_steps - 1):
            trajectory_list[step] = weighted_rigid_align(
                X_L=trajectory_list[-1],
                X_gt_L=trajectory_list[step],
                X_exists_L=X_exists_L,
                w_L=w_L,
            )

    # ... invert the list, to make the trajectory compatible with PyMol (which builds the bond graph from the first frame)
    trajectory_list = trajectory_list[::-1]

    # ... Select subset of frames if necessary
    if n_steps > max_frames:
        selected_indices = torch.linspace(0, n_steps - 1, max_frames).long().tolist()
        trajectory_list = [trajectory_list[i] for i in selected_indices]
        n_steps = len(trajectory_list)

    # ... iterate over the range of D (diffusion batch size; e.g., 5 during validation)
    # (We want to convert `aligned_trajectory_list` to a list of length D where each item is a tensor of shape [n_steps, L, 3])
    trajectories_split_by_model = []
    for d in range(trajectory_list[0].shape[0]):
        trajectory_for_single_model = torch.stack(
            [trajectory_list[step][d] for step in range(n_steps)], dim=0
        )
        trajectories_split_by_model.append(trajectory_for_single_model)

    #  ... write the trajectories to CIF files, named by epoch, dataset, example_id, and model index (within the diffusion batch)
    for i, trajectory in enumerate(trajectories_split_by_model):
        if isinstance(trajectory, torch.Tensor):
            trajectory = trajectory.cpu().numpy()
        atom_array_stack = build_stack_from_atom_array_and_batched_coords(
            trajectory, atom_array
        )

        path = f"{base_path}_model_{i}"
        to_cif_file(
            atom_array_stack, path, file_type="cif.gz", include_entity_poly=False
        )


def build_stack_from_atom_array_and_batched_coords(
    coords: np.ndarray | torch.Tensor,
    atom_array: AtomArray,
) -> AtomArrayStack:
    """Builds an AtomArrayStack from an AtomArray and a set of coordinates with a batch dimension.

    Additionally, handles the case where the AtomArray contains multiple transformations and we must adjust the chain_id.

    Args:
        coords (np.array): The coordinates to be assigned to the AtomArrayStack. Must have shape (nbatch, n_atoms, 3).
        atom_array (AtomArray): The AtomArray to be stacked. Must have shape (n_atoms,)
    """
    if isinstance(coords, torch.Tensor):
        coords = coords.cpu().numpy()

    assert (
        coords.shape[-2] == atom_array.array_length()
    ), f"N batched coordinates {coords.shape} != {atom_array.array_length()}"

    # (Diffusion batch size will become the number of models)
    n_batch = coords.shape[0]

    # Build the stack and assign the coordinates
    atom_array_stack = stack([atom_array for _ in range(n_batch)])
    atom_array_stack.coord = coords

    # Adjust chain_id if there are multiple transformations
    # (Otherwise, we will have ambiguous bond annotations, since only `chain_id` is used for the bond annotations)
    if (
        "transformation_id" in atom_array.get_annotation_categories()
        and len(np.unique(atom_array_stack.transformation_id)) > 1
    ):
        new_chain_ids = np.char.add(
            atom_array_stack.chain_id, atom_array_stack.transformation_id
        )
        atom_array_stack.set_annotation("chain_id", new_chain_ids)

    return atom_array_stack


def find_files_with_extension(path: PathLike, supported_file_types: list) -> list[Path]:
    """Recursively find all files with the given extensions in the specified path.

    Args:
        path (PathLike): Path to the directory containing the files.
        supported_file_types (list): List of supported file extensions.

    Returns:
        list[Path]: List of files with the given extensions.
    """
    files_with_supported_types = []
    path = Path(path)

    # Check if the path is a directory
    if path.is_dir():
        # Search for files with each supported extension
        for file_type in supported_file_types:
            files_with_supported_types.extend(path.glob(f"*{file_type}"))
    elif path.is_file() and path.suffix in supported_file_types:
        # If it's a file and has a supported extension, add to the list
        files_with_supported_types.append(path)

    return files_with_supported_types


def create_example_id_extractor(extensions: set | list = CIF_LIKE_EXTENSIONS) -> str:
    """Create a function with closure that extracts example_ids from file paths with specified extensions.

    Example:
        >>> extractor = create_example_id_extractor({".cif", ".cif.gz"})
        >>> extractor("example.path.example_id.cif.gz")
        'example_id'
    """
    pattern = re.compile(
        "(" + "|".join(re.escape(ext) + "$" for ext in extensions) + ")"
    )

    def extract_id(file_path: PathLike) -> str:
        """Extract example_id from file path."""
        # Remove extension and get last part after splitting by dots
        without_ext = pattern.sub("", Path(file_path).name)
        return without_ext.split(".")[-1]

    return extract_id


def extract_example_id_from_path(file_path: PathLike, extensions: set | list) -> str:
    """Extract example_id from file path with specified extensions."""
    extractor = create_example_id_extractor(extensions)
    return extractor(file_path)
