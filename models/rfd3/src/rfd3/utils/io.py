import re
from os import PathLike
from pathlib import Path

import numpy as np
import torch
from biotite.structure import AtomArray, AtomArrayStack, stack

DICTIONARY_LIKE_EXTENSIONS = {".json", ".yaml", ".yml", ".pkl"}
CIF_LIKE_EXTENSIONS = {".cif", ".pdb", ".bcif", ".cif.gz", ".pdb.gz", ".bcif.gz"}


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
