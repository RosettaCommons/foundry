import hydra
from os import PathLike
from pathlib import Path
from cifutils import parse
from omegaconf import OmegaConf
import torch
from rf2aa.trainer_base import trainer_factory

import os
from datahub.utils.io import convert_af3_model_output_to_atom_array_stack
from cifutils.tools.inference import components_to_atom_array, build_msa_paths_by_chain_id_from_component_list
from datahub.encoding_definitions import AF3SequenceEncoding
from cifutils.utils.io_utils import to_cif_file
import logging
import tempfile
import argparse
import json
from rf2aa.metrics.metric_utils import write_confidence_metrics
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the sequence encoding; needed to decode the restypes when saving to CIF
encoding = AF3SequenceEncoding()

class EvaluateAF3:
    """Class for inference with AF3. Evaluates a trained AF3 model on a set of spoofed CIFs."""
    def __init__(self, 
                 checkpoint_path: PathLike, 
                 cif_out_dir: PathLike, 
                 n_recycles: int, 
                 diffusion_batch_size: int
                 ):
        """Initialize the evaluator.

        Args:
            checkpoint_path (PathLike): Path to the checkpoint file, e.g., /path/to/checkpoint.pt.
            cif_out_dir (PathLike): Directory to save the output (predicted) CIF files.
            world_size (int): Number of GPUs to use for evaluation.
            n_recycles (int): Number of recycles for AF3. The default is 10.
            diffusion_batch_size (int): Diffusion batch size for AF3. Each predicted structure will be saved as a separate model within the same CIF file.
        """

        # Load the checkpoint
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))

        # Load the config
        self.config = OmegaConf.create(checkpoint["training_config"])

        # Sampler sets diffusion batch size based on the following, not strictly on batch size in vaildation transform
        self.config.dataset_params["diffusion_batch_size_valid"] = diffusion_batch_size

        # Load the AF-3 trainer
        self.trainer = trainer_factory[self.config.experiment.trainer](config=self.config)
        self.trainer.checkpoint = checkpoint

        # Set the output directory for the CIF files (e.g., predicted structures)
        self.cif_out_dir = Path(cif_out_dir) if cif_out_dir is not None else None

        # Model parameters
        self.n_recycles = n_recycles
        self.diffusion_batch_size = diffusion_batch_size

    def construct_pipeline(self):
        """Construct the AF3 inference pipeline."""
        self.config.dataset_params.val.interface.transform.n_recycles = self.n_recycles
        self.config.dataset_params.val.interface.transform.diffusion_batch_size = self.diffusion_batch_size

        assert self.config.dataset_params.val.interface.transform.n_recycles == self.n_recycles, "Number of recycles not set correctly."
        assert self.config.dataset_params.val.interface.transform.diffusion_batch_size == self.diffusion_batch_size, "Diffusion batch size not set correctly."
        pipeline = hydra.utils.instantiate(self.config.dataset_params.val.interface.transform)
        return pipeline
    
    def find_files(self, spoofed_cif_path):
        """Find all files with the given extensions in the spoofed CIF directory.

        Args:
            spoofed_cif_path (Path): Path to the directory containing spoofed CIF files.

        Returns:
            List[Path]: List of files with the given extensions.
        """
        matched_files = []
        valid_extensions = [".cif", ".pdb", ".bcif", ".cif.gz", ".pdb.gz", ".bcif.gz"]

        if spoofed_cif_path.is_file():
            # Check if the file has one of the expected extensions
            if any(spoofed_cif_path.name.endswith(ext) for ext in valid_extensions):
                matched_files.append(spoofed_cif_path)
        else:
            # If it's a directory, search for files with the given extensions
            logger.info(f"Searching for files with extensions {valid_extensions} in {spoofed_cif_path}...")
            for ext in valid_extensions:
                matched_files.extend(spoofed_cif_path.glob(f"*{ext}"))
        
        return matched_files

    def eval(self, files: list[os.PathLike]):
        """Evaluate the model on a set of spoofed CIF files. 

        Args:
            files (list[os.PathLike]): List of paths to spoofed CIF files or directories containing spoofed CIF files.
                Coordinates must be present but may contain NaN values. If a directory is provided, 
                all files with the extensions .cif, .pdb, .bcif, .cif.gz, .pdb.gz, .bcif.gz will be processed.
        """
        # Construct the model and load the checkpoint
        gpu = f"cuda:0" if torch.cuda.is_available() else "cpu"
        self.trainer.construct_model(device=gpu, inference=True)
        self.trainer.load_model()

        # Set the model to evaluation mode
        self.trainer.model.eval()

        logger.info(f"Building Transform pipeline...")

        # Construct the AF3 inference pipeline
        pipeline = self.construct_pipeline()

        # Accumulate all structures to predict
        structures_to_predict = []
        for file in files:
            assert Path(file).exists(), f"Path {file} does not exist."
            structures_to_predict.extend(self.find_files(file))

        logger.info(f"Found {len(structures_to_predict)} structures to predict: {structures_to_predict}.")

        for structure in structures_to_predict:
            # ... parse into an AtomArray (`parse` handles all valid formats)
            logger.info(f"Parsing from path: {structure}")
            example_id = structure.name.split('.')[0]
            out = parse(structure, remove_hydrogens=True)

            # ... get the atom array and set NaN coordinates to random
            atom_array = out["assemblies"]["1"][0] if "assemblies" in out else out["asym_unit"][0]
            # HACK: Set NaN coordinates to random values to avoid unexpected behavior in the pipeline
            atom_array.coord[np.isnan(atom_array.coord)] = np.random.rand(*atom_array.coord[np.isnan(atom_array.coord)].shape)

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
                outputs = self.trainer.sampler.sample([pipeline_output], n_cycle=self.n_recycles, use_amp=self.config.training_params.use_amp)

            # Collect information needed to write out the CIF file
            atom_to_token_map = pipeline_output["feats"]["atom_to_token_map"].cpu().numpy()
            decoded_restypes = encoding.decode(torch.argmax(pipeline_output["feats"]["restype"], dim=-1).cpu())
            pn_unit_iids = pipeline_output["ground_truth"]["chain_iid_token_lvl"]
            xyz = outputs["X_L"].cpu().numpy()
            elements = torch.argmax(pipeline_output["feats"]["ref_element"], -1).cpu().numpy()

            # Convert the model output to an atom array
            atom_array_stack = convert_af3_model_output_to_atom_array_stack(
                atom_to_token_map=atom_to_token_map,
                pn_unit_iids=pn_unit_iids,
                decoded_restypes=decoded_restypes,
                xyz=xyz,
                elements=elements,
            )

            # Write the atom array to a CIF file
            # NOTE: To make the secondary structure appear, run `dss` in PyMol (see: https://biology.stackexchange.com/questions/70143/can-pymol-show-cartoon-secondary-structure-for-a-pdb-of-multiple-frames)
            logger.info(f"Writing prediction for {example_id}.cif to {self.cif_out_dir / example_id}...")
            out_path = Path(to_cif_file(atom_array_stack, self.cif_out_dir / f"{example_id}.cif", include_entity_poly=False))
            logger.info(f"Prediction for {example_id} written to {out_path}.")

            if "confidence" in outputs:
                logger.info(f"Writing {example_id}.score to {self.cif_out_dir}")
                write_confidence_metrics(outputs, self.cif_out_dir / f"{example_id}.score", device=gpu)
                logger.info(f"Confidence metrics for {example_id}.cif written to {self.cif_out_dir / example_id}.score.")
                
def main():
    parser = argparse.ArgumentParser(description="Evaluate AF3 using specified paths.")
    parser.add_argument("inputs", nargs="+", help="List of paths to files (JSON or CIF/PDB) or directories of CIF/PDB files.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument("--cif_out_dir", type=str, required=True, help="Directory for output CIF files")
    parser.add_argument("--n_recycles", type=int, default=10, help="Number of recycles for AF3")
    parser.add_argument("--diffusion_batch_size", type=int, default=5, help="Diffusion batch size for AF3")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Prepare inputs based on the file types
        file_paths_for_prediction = []
        for path in args.inputs:
            path = Path(path)
            if path.suffix in {".json"}:
                with open(path, 'r') as json_file:
                    # ... load the JSON data
                    inputs = json.load(json_file)

                    # ... build components
                    atom_array, components = components_to_atom_array(inputs, return_components=True)
                    msa_paths_by_chain_id = build_msa_paths_by_chain_id_from_component_list(components)

                    # ... create a temporary CIF file from the JSON data
                    # (By writing the MSA paths to a category in the CIF file, they will ultimately end up in `chain_info`, as desired)
                    path = temp_dir / f"{path.stem}.cif"
                    save_path = to_cif_file(
                        atom_array,
                        path,
                        extra_categories={"msa_paths_by_chain_id": msa_paths_by_chain_id} if msa_paths_by_chain_id else None,
                    )
                    file_paths_for_prediction.append(Path(save_path))
            else:
                file_paths_for_prediction.append(path)
        
        # Construct the evaluator
        evaluator = EvaluateAF3(
            checkpoint_path=args.checkpoint_path,
            cif_out_dir=args.cif_out_dir,
            n_recycles=args.n_recycles,
            diffusion_batch_size=args.diffusion_batch_size,
        )

        # Launch the evaluation
        evaluator.eval(files=file_paths_for_prediction)

if __name__ == "__main__":
    main()
