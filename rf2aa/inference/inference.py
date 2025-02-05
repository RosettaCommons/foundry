import argparse
import json
import logging
import pickle
import tempfile
from os import PathLike
from pathlib import Path

import hydra
import numpy as np
import torch
from biotite.structure import AtomArray, AtomArrayStack, stack
from cifutils import parse
from cifutils.tools.inference import (
    build_msa_paths_by_chain_id_from_component_list,
    components_to_atom_array,
)
from cifutils.utils.io_utils import to_cif_file
from datahub.encoding_definitions import AF3SequenceEncoding
from omegaconf import OmegaConf

from rf2aa.metrics.predicted_error import WriteAF3Confidence
from rf2aa.trainer_base import trainer_factory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the sequence encoding; needed to decode the restypes when saving to CIF
encoding = AF3SequenceEncoding()


def build_stack_from_atom_array_and_batched_coords(
    coords: np.ndarray,
    atom_array: AtomArray,
    annotations_to_keep: list[str] = [
        "chain_id",
        "transformation_id",
        "res_id",
        "res_name",
        "element",
        "atom_name",
    ],
) -> AtomArrayStack:
    """Builds an AtomArrayStack from an AtomArray and a set of coordinates with a batch dimension.

    Additionally, handles the case where the AtomArray contains multiple transformations and we must adjust the chain_id.

    Args:
        coords (np.array): The coordinates to be assigned to the AtomArrayStack. Must have shape (nbatch, n_atoms, 3).
        atom_array (AtomArray): The AtomArray to be stacked. Must have shape (n_atoms,)
    """
    # (Diffusion batch size will become the number of models)
    n_batch = coords.shape[0]

    # Remove unwanted annotations
    for annotation in atom_array.get_annotation_categories():
        if annotation not in annotations_to_keep:
            atom_array.del_annotation(annotation)

    # Build the stack and assign the coordinates
    atom_array_stack = stack([atom_array for _ in range(n_batch)])
    atom_array_stack.coord = coords

    # Adjust chain_id if there are multiple transformations
    # (Otherwise, we will have ambiguous bond annotations, since only `chain_id` is used for the bond annotations)
    if (
        "transformation_id" in atom_array.get_annotation_categories()
        and len(np.unique(atom_array_stack.transformation_id)) > 1
    ):
        atom_array_stack.chain_id = (
            atom_array_stack.chain_id + atom_array_stack.transformation_id
        )

    return atom_array_stack


def _spoof_cif_from_dictionary(item: dict, temp_dir: PathLike) -> Path:
    """Unpacks a dictionary to create a CIF file from its components.

    Args:
        item (dict): A dictionary containing 'name' and 'components', optionally 'bonds'.
        temp_dir (Path): Path to the temporary directory for storing CIF files.

    Returns:
        Path: The path to the created CIF file, saved in the temporary directory.

    Raises:
        NotImplementedError: If 'bonds' is present in the dictionary.
        ValueError: If 'name' or 'components' are missing from the dictionary.
    """
    # Validate the dictionary structure ("name" and "components" are required, "bonds" is optional)
    assert "name" in item and "components" in item, (
        "The input dictionary must contain 'name' and 'components' keys."
    )

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


def _build_file_paths_for_prediction(inputs: list, temp_dir: PathLike) -> list[Path]:
    """Prepare files for prediction based on the input paths.

    Input paths may be dictionary-like format (e.g., JSON, YAML, Pickle), CIF/PDB files, or directories containing these files.
    Processes directories to find supported file types and converts dictionary-like formats to CIF files.

    Args:
        inputs (list): List of input paths (JSON, YAML, Pickle, or CIF/PDB).
        temp_dir (Path): Path to the temporary directory for storing CIF files.

    Returns:
        list[Path]: List of file paths for prediction.
    """
    DICTIONARY_LIKE_EXTENSIONS = {".json", ".yaml", ".yml", ".pkl"}
    CIF_LIKE_EXTENSIONS = {".cif", ".pdb", ".bcif", ".cif.gz", ".pdb.gz", ".bcif.gz"}

    # Collect all files from inputs, handling directories and individual files
    paths_to_raw_input_files = []
    for input_path in inputs:
        if Path(input_path).is_dir():
            paths_to_raw_input_files.extend(
                _find_files(
                    input_path, DICTIONARY_LIKE_EXTENSIONS | CIF_LIKE_EXTENSIONS
                )
            )
        else:
            paths_to_raw_input_files.append(Path(input_path))

    paths_to_cif_like_files = []
    for path in paths_to_raw_input_files:
        concatenated_suffix = "".join(path.suffixes)
        if concatenated_suffix in DICTIONARY_LIKE_EXTENSIONS:
            # Spoof CIF files from dictionary-like formats
            with open(path, "rb" if path.suffix == ".pkl" else "r") as file:
                # Load data based on file extension
                if path.suffix == ".json":
                    data = json.load(file)
                elif path.suffix in {".yaml", ".yml"}:
                    raise NotImplementedError("YAML files are not yet supported.")
                elif path.suffix == ".pkl":
                    data = pickle.load(file)

                if isinstance(data, dict):
                    data = [
                        data
                    ]  # Convert single dictionary to list for uniform processing

                for item in data:
                    paths_to_cif_like_files.append(
                        _spoof_cif_from_dictionary(item, temp_dir)
                    )
        elif concatenated_suffix in CIF_LIKE_EXTENSIONS:
            # Directly use CIF-like files
            paths_to_cif_like_files.append(path)
        else:
            raise ValueError(
                f"Unsupported file extension: {path.suffix} (path: {path}; paths: {paths_to_raw_input_files})."
            )

    return paths_to_cif_like_files


def _find_files(path: PathLike, supported_file_types: list) -> list[Path]:
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


class EvaluateAF3:
    """Class for inference with AF3. Evaluates a trained AF3 model on a set of spoofed CIFs."""

    def __init__(
        self,
        checkpoint_path: PathLike,
        cif_out_dir: PathLike,
        n_recycles: int,
        diffusion_batch_size: int,
        residue_renaming_dict: dict | None = None,
        temp_dir: PathLike | None = None,
        num_steps: int = 200,
        solver: str = "af3",
    ):
        """Initialize the evaluator.

        Args:
            checkpoint_path (PathLike): Path to the checkpoint file, e.g., /path/to/checkpoint.pt.
            cif_out_dir (PathLike): Directory to save the output (predicted) CIF files.
            world_size (int): Number of GPUs to use for evaluation.
            n_recycles (int): Number of recycles for AF3. The default is 10.
            diffusion_batch_size (int): Diffusion batch size for AF3. Each predicted structure will be saved as a separate model within the same CIF file.
            residue_renaming_dict (dict): Dictionary of residue names to rename to avoid CCD clashes, e.g., {'ALA': 'L:1'}.
            temp_dir (PathLike): Temporary directory to store intermediate files. The default is None.
            num_steps (int): Number of steps for sampling of the diffusion model. The default is 200; we see reasonable results with 50 steps.
            solver (str): Solver to use for inference. Options are 'af3', 'simple', 'euler', and 'heun'. The default is 'af3'.
        """

        # Load the checkpoint
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))

        # Load the config
        self.config = OmegaConf.create(checkpoint["training_config"])

        # Sampler sets diffusion batch size based on the following, not strictly on batch size in vaildation transform
        self.config.dataset_params["diffusion_batch_size_valid"] = diffusion_batch_size
        self.config.af3_inference["num_steps"] = num_steps
        self.config.af3_inference["solver"] = solver

        # Load the AF-3 trainer
        self.trainer = trainer_factory[self.config.experiment.trainer](
            config=self.config
        )
        self.trainer.checkpoint = checkpoint

        # Set the output directory for the CIF files (e.g., predicted structures)
        self.cif_out_dir = Path(cif_out_dir) if cif_out_dir else Path("./")

        # Model parameters
        self.n_recycles = n_recycles
        self.diffusion_batch_size = diffusion_batch_size
        if "confidence_loss" in self.config.loss:
            self.confidence_writer = WriteAF3Confidence(
                **self.config.loss.confidence_loss
            )
        else:
            self.confidence_writer = None

        # Rename residues
        self.residue_renaming_dict = residue_renaming_dict
        self.temp_dir = Path(temp_dir)

    def construct_pipeline(self):
        """Construct the AF3 inference pipeline."""
        self.config.dataset_params.val.interface.transform.n_recycles = self.n_recycles
        self.config.dataset_params.val.interface.transform.diffusion_batch_size = (
            self.diffusion_batch_size
        )
        self.config.dataset_params.val.interface.transform.return_atom_array = (
            True  # Required for `to_cif`
        )

        assert (
            self.config.dataset_params.val.interface.transform.n_recycles
            == self.n_recycles
        ), "Number of recycles not set correctly."
        assert (
            self.config.dataset_params.val.interface.transform.diffusion_batch_size
            == self.diffusion_batch_size
        ), "Diffusion batch size not set correctly."
        pipeline = hydra.utils.instantiate(
            self.config.dataset_params.val.interface.transform
        )
        return pipeline

    def eval(self, files: list[PathLike]):
        """Evaluate the model on a set of spoofed CIF files.

        Args:
            files (list[PathLike]): List of paths to spoofed CIF files or directories containing spoofed CIF files.
                Coordinates must be present but may contain NaN values. If a directory is provided,
                all files with the extensions .cif, .pdb, .bcif, .cif.gz, .pdb.gz, .bcif.gz will be processed.
        """
        # Construct the model and load the checkpoint
        gpu = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.trainer.construct_model(device=gpu, inference=True)
        self.trainer.load_model()

        # Set the model to evaluation mode
        self.trainer.model.eval()

        logger.info("Building Transform pipeline...")

        # Construct the AF3 inference pipeline
        pipeline = self.construct_pipeline()

        logger.info(f"Found {len(files)} structures to predict: {files}.")

        for structure in files:
            # ... parse into an AtomArray (`parse` handles all valid formats)
            logger.info(f"Parsing from path: {structure}")
            example_id = structure.name.split(".")[0]

            # If we're renaming residues, we do a brute-force replacement in the CIF file
            if self.residue_renaming_dict:
                logger.info(
                    f"Renaming residues in {structure} with brute-force find and replace: {self.residue_renaming_dict}"
                )
                with open(structure, "r") as f:
                    content = f.read()
                    for old_res, new_res in self.residue_renaming_dict.items():
                        content = content.replace(old_res, new_res)
                structure = Path(self.temp_dir / structure.name)
                with open(structure, "w") as f:
                    f.write(content)

            out = parse(structure, remove_hydrogens=True)

            # ... get the atom array and set NaN coordinates to random
            atom_array = (
                out["assemblies"]["1"][0]
                if "assemblies" in out
                else out["asym_unit"][0]
            )

            # HACK: Set NaN coordinates to random values to avoid unexpected behavior in the pipeline
            atom_array.coord[np.isnan(atom_array.coord)] = np.random.rand(
                *atom_array.coord[np.isnan(atom_array.coord)].shape
            )

            # ... assemble the pipeline input in a format compatible with the DataHub pipeline
            pipeline_input = {
                "example_id": example_id,
                "atom_array": atom_array,
                "chain_info": out["chain_info"],
            }

            # ... run dataloading and featurization
            pipeline_output = pipeline(pipeline_input)

            # Model inference
            with torch.no_grad():
                outputs = self.trainer.sampler.sample(
                    [pipeline_output],
                    n_cycle=self.n_recycles,
                    use_amp=self.config.training_params.use_amp,
                )

            # Override the AtomArray with the predited coordinates
            atom_array_stack = build_stack_from_atom_array_and_batched_coords(
                outputs["X_L"].cpu().numpy(), pipeline_output["atom_array"]
            )

            # Write the atom array to a CIF file
            # NOTE: To make the secondary structure appear, run `dss` in PyMol (see: https://biology.stackexchange.com/questions/70143/can-pymol-show-cartoon-secondary-structure-for-a-pdb-of-multiple-frames)
            out_path = to_cif_file(
                atom_array_stack, self.cif_out_dir / example_id, file_type="cif"
            )
            logger.info(f"Prediction for {example_id} written to {out_path}.")

            if "confidence" in outputs:
                loss_input = {
                    "example_id": example_id,
                    "is_real_atom": pipeline_output["confidence_feats"]["is_real_atom"],
                }
                logger.info(f"Writing {example_id}.score to {self.cif_out_dir}")
                df = self.confidence_writer(None, outputs, loss_input)
                df.to_csv(self.cif_out_dir / f"{example_id}.score", index=False)
                logger.info(
                    f"Confidence metrics for {example_id}.cif written to {self.cif_out_dir / example_id}.score."
                )


def main():
    parser = argparse.ArgumentParser(description="Evaluate AF3 using specified paths.")
    parser.add_argument(
        "inputs",
        nargs="+",
        help="List of paths to supported file types or directories of of supported files.",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, required=True, help="Path to the checkpoint file"
    )
    parser.add_argument(
        "--cif_out_dir", type=str, required=True, help="Directory for output CIF files"
    )
    parser.add_argument(
        "--n_recycles", type=int, default=10, help="Number of recycles for AF3"
    )
    parser.add_argument(
        "--diffusion_batch_size",
        type=int,
        default=5,
        help="Diffusion batch size for AF3",
    )
    parser.add_argument(
        "--rename_residues",
        type=str,
        default="",
        help="Dictionary of residue names to rename to avoid CCD clashes, e.g., {'ALA': 'L:1'}",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=200,
        help="Number of steps for sampling of the diffusion model",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="af3",
        help="Solver to use for inference. Options are 'af3', 'simple', 'euler', and 'heun'.",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Prepare inputs based on the file types
        file_paths_for_prediction = _build_file_paths_for_prediction(
            args.inputs, temp_dir
        )

        # Rename residues if necessary (e.g., for MPNN outputs that have ligand names that clash with the CCD)
        residue_renaming_dict = (
            json.loads(args.rename_residues) if args.rename_residues else {}
        )

        # Construct the evaluator
        evaluator = EvaluateAF3(
            checkpoint_path=args.checkpoint_path,
            cif_out_dir=args.cif_out_dir,
            n_recycles=args.n_recycles,
            diffusion_batch_size=args.diffusion_batch_size,
            residue_renaming_dict=residue_renaming_dict,
            temp_dir=temp_dir,
            num_steps=args.num_steps,
            solver=args.solver,
        )

        # Launch the evaluation
        evaluator.eval(files=file_paths_for_prediction)


if __name__ == "__main__":
    main()
