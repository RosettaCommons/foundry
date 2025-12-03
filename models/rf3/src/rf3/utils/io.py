from os import PathLike
from pathlib import Path

import numpy as np
import torch
from atomworks.io.utils.io_utils import to_cif_file
from atomworks.ml.utils.io import apply_sharding_pattern
from atomworks.ml.utils.misc import hash_sequence
from beartype.typing import Literal
from biotite.structure import AtomArray, AtomArrayStack, stack

from foundry.utils.alignment import weighted_rigid_align
from foundry.utils.ddp import RankedLogger

ranked_logger = RankedLogger(__name__, rank_zero_only=True)

DICTIONARY_LIKE_EXTENSIONS = {".json", ".yaml", ".yml", ".pkl"}
CIF_LIKE_EXTENSIONS = {".cif", ".pdb", ".bcif", ".cif.gz", ".pdb.gz", ".bcif.gz"}


def get_sharded_output_path(
    example_id: str,
    base_dir: Path,
    sharding_pattern: str | None = None,
) -> Path:
    """Get output directory path for an example with optional sharding.

    Args:
        example_id: Example identifier (used as final directory name).
        base_dir: Base output directory.
        sharding_pattern: Sharding pattern like ``/0:2/2:4/`` or ``None`` for no sharding.
            Pattern defines how to split the hash of ``example_id`` into nested directories.

    Returns:
        Output directory path. If sharding is enabled, returns ``base_dir/shard1/shard2/.../example_id``.
        Otherwise returns ``base_dir/example_id``.

    Examples:
        Without sharding::

            get_sharded_output_path("entry_1", Path("/out"))
            # Returns: /out/entry_1

        With sharding pattern ``/0:2/2:4/``::

            get_sharded_output_path("entry_1", Path("/out"), "/0:2/2:4/")
            # Computes hash of "entry_1" (e.g., "a1b2c3d4e5f")
            # Returns: /out/a1/b2/entry_1
    """
    if not sharding_pattern:
        return base_dir / example_id

    # Hash the example ID and apply sharding pattern
    example_hash = hash_sequence(example_id)
    sharded_path = apply_sharding_pattern(example_hash, sharding_pattern)

    # Return base_dir / sharded_directories / example_id
    return base_dir / sharded_path.parent / example_id


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


def dump_structures(
    atom_arrays: AtomArrayStack | list[AtomArray] | AtomArray,
    base_path: PathLike,
    one_model_per_file: bool,
    extra_fields: list[str] | Literal["all"] = [],
    file_type: str = "cif.gz",
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
                file_type=file_type,
                include_entity_poly=False,
                extra_fields=extra_fields,
            )
    else:
        # Include all models in a single CIF file
        to_cif_file(
            atom_arrays,
            base_path,
            file_type=file_type,
            include_entity_poly=False,
            extra_fields=extra_fields,
        )


def dump_trajectories(
    trajectory_list: list[torch.Tensor | np.ndarray],
    atom_array: AtomArray,
    base_path: Path,
    align_structures: bool = True,
    file_type: str = "cif.gz",
) -> None:
    """Write denoising trajectories to CIF files.

    Args:
        trajectory_list (List[torch.Tensor]): List of tensors of length n_steps representing the diffusion trajectory at each step.
            Each tensor has shape [D, L, 3], where D is the diffusion batch size and L is the number of atoms.
        atom_array (np.ndarray): Atom array corresponding to the coordinates.
        base_path (Path): Base path where the output files will be saved.
        align_structures (bool): Flag to determine if the structures should be aligned on the final prediction.
            If False, each step may have a different alignment.
        file_type (str): File type for output (e.g., "cif", "cif.gz", "pdb"). Defaults to ``"cif.gz"``.
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
            atom_array_stack, path, file_type=file_type, include_entity_poly=False
        )
