from os import PathLike
from pathlib import Path

import hydra
import numpy as np
import torch
from cifutils import parse
from omegaconf import OmegaConf

from modelhub.utils.inference import build_file_paths_for_prediction
from modelhub.utils.ddp import RankedLogger, set_accelerator_based_on_availability
from modelhub.utils.logging import print_config_tree
from modelhub.utils.datasets import (
    assemble_distributed_inference_loader_from_list_of_paths,
)
from modelhub.utils.predicted_error import compile_af3_confidence_outputs, annotate_atom_array_b_factor_with_plddt
from modelhub.inference_engines.base import InferenceEngine
from modelhub.utils.io import (
    dump_structures,
    dump_trajectories,
    build_stack_from_atom_array_and_batched_coords,
)
import logging
from biotite.structure import AtomArray
from lightning.fabric import seed_everything

logging.basicConfig(level=logging.INFO)
ranked_logger = RankedLogger(__name__, rank_zero_only=True)


class AF3InferenceEngine(InferenceEngine):
    """Class for inference with AF3. Evaluates a trained AF3 model on a set of spoofed CIFs."""

    def __init__(
        self,
        # Base arguments
        inputs: PathLike | list[PathLike],
        ckpt_path: PathLike,
        out_dir: PathLike | None,
        num_nodes: int,
        devices_per_node: int,
        skip_existing: bool,
        # Model args
        n_recycles: int,
        diffusion_batch_size: int,
        residue_renaming_dict: str | dict,
        num_steps: int,
        solver: str,
        print_config: bool,
        temp_dir: PathLike,
        seed: int,
        # Structure dumping arguments
        dump_predictions: bool,
        dump_trajectories: bool,
        one_model_per_file: bool,
    ):
        """Initialize the Inference Engine for AF3.

        Note that for inference, we initialize the Hydra configuration from the checkpoint; we then override specific parameters based on the input arguments
        and inference-specific considerations.

        Args:
            ckpt_path: Path to the checkpoint file.
            out_dir: Directory for output files. If None, the current directory will be used.
            skip_existing: If True, only predict the structures that are not already in the output directory.
            num_nodes: Number of nodes for distributed inference. The default is 1.
            devices_per_node: Number of devices per node for distributed inference. The default is 1.

            n_recycles (int): Number of recycles for AF3.
            diffusion_batch_size (int): Diffusion batch size for AF3. Each predicted structure will be saved as a separate model within the same CIF file.
            residue_renaming_dict (dict): Dictionary of residue names to rename to avoid CCD clashes, e.g., {'ALA': 'L:1'}.
            num_steps (int): Number of steps for sampling of the diffusion model. AF-3 uses 200; we see no degradation in performance with 50.
            solver (str): Solver to use for inference. Options are 'af3', 'simple', 'euler', and 'heun'.
            print_config (bool): Pretty-print the Hydra configs.
            temp_dir (PathLike): Temporary directory to store intermediate files.
            seed (int): Random seed for reproducibility / augmentation. If None, the default seed from the config will be used.

            dump_predictions (bool): Whether to dump structures (CIF files).
            dump_trajectories (bool): Whether to dump denoising trajectories.
            one_model_per_file (bool): If True, write each structure within a diffusion batch to its own CIF files.
                If False, include each structure within a diffusion batch as a separate model within one CIF file.
        """
        if solver != "af3":
            # TODO: Port over additional solvers (Frank already coded; need to modify for new framework)
            raise NotImplementedError(
                f"Solver {solver} not implemented. Only 'af3' is supported for inference."
            )

        # Load the training config from the checkpoint
        # TODO: Load checkpoint only once (instead of twice)
        ranked_logger.info(f"Loading checkpoint from {Path(ckpt_path).resolve()}...")
        checkpoint = torch.load(
            ckpt_path, "cpu"
        )  # We only extract the `train_cfg` from the checkpoint initially
        self.cfg = OmegaConf.create(checkpoint["train_cfg"])

        self.paths = build_file_paths_for_prediction(
            input=inputs, 
            temp_dir=temp_dir, 
            existing_outputs_dir=out_dir if skip_existing else None
        )

        # Override specific parameters within the Hydra config:
        #  (a) based on the input arguments
        self.cfg.model.net.inference_sampler.num_timesteps = num_steps
        self.cfg.model.net.inference_sampler.solver = solver
        self.cfg.trainer.num_nodes = num_nodes
        self.cfg.trainer.devices_per_node = devices_per_node

        set_accelerator_based_on_availability(self.cfg)

        # (b) based on the dataset (we will apply when constructing the pipeline)
        self.dataset_overrides = {
            "diffusion_batch_size": diffusion_batch_size,
            "n_recycles": n_recycles,
            "undesired_res_names": [],
        }

        self.print_config = print_config

        # ... set the random seed for reproducibility (and for augmentation, e.g., for antibodies)
        seed = seed or self.cfg.seed
        ranked_logger.info(f"Seeding everything with seed={seed}...")
        seed_everything(seed, workers=True, verbose=True)

        ranked_logger.info("Instantiating trainer...")
        if self.print_config:
            print_config_tree(
                self.cfg.trainer, resolve=True, title="INFERENCE TRAINER CONFIGURATION"
            )

        # ... instantiate the trainer with the (modified) configuration
        self.trainer = hydra.utils.instantiate(
            self.cfg.trainer,
            _convert_="partial",
            _recursive_=False,
        )

        self.ckpt_path = ckpt_path

        # Set the output directory for the CIF files (e.g., predicted structures)
        self.cif_out_dir = Path(out_dir) if out_dir else Path("./")

        # Rename residues
        self.residue_renaming_dict = residue_renaming_dict
        self.temp_dir = Path(temp_dir)

        # Structure dumping
        self.dump_predictions = dump_predictions
        self.dump_trajectories = dump_trajectories
        self.one_model_per_file = one_model_per_file

    def construct_pipeline(self):
        """Construct the AF3 inference pipeline.

        By convention we use the "interface" dataset stored in the checkpoint to construct the pipeline.
        """
        # ... find the first validation dataset stored under "val"
        first_val_dataset_key, first_val_dataset = next(
            iter(self.cfg.datasets.val.items())
        )
        ranked_logger.info(
            f"Using the settings from the first validation dataset: {first_val_dataset_key}."
        )

        assert (
            first_val_dataset.dataset.transform.is_inference
        ), "Inference must be enabled for the validation dataset."
        for key, value in self.dataset_overrides.items():
            first_val_dataset.dataset.transform[key] = value

        if self.print_config:
            print_config_tree(
                first_val_dataset.dataset.transform,
                resolve=True,
                title="INFERENCE TRANSFORM PIPELINE",
            )

        pipeline = hydra.utils.instantiate(
            first_val_dataset.dataset.transform,
        )

        return pipeline
    
    def parse_from_path(self, path_to_structure: Path) -> dict:
        """Parse a structure from a CIF file.

        Perform additional processing if necessary, such as renaming residues.
        """
        # If we're renaming residues, we do a brute-force replacement in the CIF file
        if self.residue_renaming_dict:
            ranked_logger.info(
                f"Renaming residues in {path_to_structure} with brute-force find and replace: {self.residue_renaming_dict}"
            )
            with open(path_to_structure, "r") as f:
                content = f.read()
                for old_res, new_res in self.residue_renaming_dict.items():
                    content = content.replace(old_res, str(new_res))
            path_to_structure = Path(self.temp_dir / path_to_structure.name)
            with open(path_to_structure, "w") as f:
                f.write(content)

        return parse(path_to_structure, remove_hydrogens=True)
    
    def prepare_atom_array(self, atom_array: AtomArray) -> AtomArray:
        """Prepare the AtomArray for inference.

        By default, we set NaN coordinates to random values to avoid unexpected behavior in the pipeline.
        """
        # HACK: Set NaN coordinates to random values to avoid unexpected behavior in the pipeline
        # TODO: Hunt down why NaN coordinates lead to this behavior
        atom_array.coord[np.isnan(atom_array.coord)] = np.random.rand(
            *atom_array.coord[np.isnan(atom_array.coord)].shape
        )

        return atom_array

    def eval(self):
        """Evaluate the model on a set of spoofed CIF files."""
        if self.print_config:
            print_config_tree(
                self.cfg.model, resolve=True, title="INFERENCE MODEL CONFIGURATION"
            )

        # ... spawn processes for distributed training, if using multiple GPUs
        ranked_logger.info(
            f"Spawning {self.trainer.fabric.world_size} processes from {self.trainer.fabric.global_rank}..."
        )

        # ==============================================================================
        # Construct the model and load the checkpoint
        # ==============================================================================

        self.trainer.initialize_or_update_trainer_state({"train_cfg": self.cfg})
        self.trainer.construct_model()
        self.trainer.load_checkpoint(ckpt_path=self.ckpt_path)

        self.trainer.state["model"].eval()

        # ==============================================================================
        # Prepare pipeline and inference loader
        # ==============================================================================

        ranked_logger.info("Building Transform pipeline...")

        # Construct the AF3 inference pipeline
        pipeline = self.construct_pipeline()

        ranked_logger.info(f"Found {len(self.paths)} structures to predict!")

        loader = assemble_distributed_inference_loader_from_list_of_paths(
            paths=self.paths,
            world_size=self.trainer.fabric.world_size,
            rank=self.trainer.fabric.global_rank,
        )

        # ==============================================================================
        # Evaluate, using `validation_step``
        # ==============================================================================

        for batch_idx, path_to_structure in enumerate(loader):
            # (We only have one path per batch)
            path_to_structure = path_to_structure[0]

            ranked_logger.info(
                f"Predicting structure {batch_idx + 1}/{len(loader)}: {path_to_structure.name}"
            )

            # ... parse into an AtomArray (`parse` handles all valid formats)
            ranked_logger.info(f"Parsing from path: {path_to_structure}")
            example_id = path_to_structure.name.split(".")[0]

            out = self.parse_from_path(path_to_structure)

            # ... get the atom array and set NaN coordinates to random
            atom_array = (
                out["assemblies"]["1"][0]
                if "assemblies" in out
                else out["asym_unit"][0]
            )

            atom_array = self.prepare_atom_array(atom_array)

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
                pipeline_output = self.trainer.fabric.to_device(pipeline_output)
                network_output = self.trainer.validation_step(
                    batch=pipeline_output,
                    batch_idx=0,
                    compute_metrics=False,
                )["network_output"]

                # TODO: Log `metrics_output` to a file (or store directly within the CIF file)

            # ... build the predicted AtomArrayStack
            atom_array_stack = build_stack_from_atom_array_and_batched_coords(
                network_output["X_L"], pipeline_output["atom_array"]
            )

            if "plddt" in network_output:
                confidence_outs = compile_af3_confidence_outputs(
                    plddt_logits=network_output["plddt"],
                    pae_logits=network_output["pae"],
                    pde_logits=network_output["pde"],
                    chain_iid_token_lvl=pipeline_output["ground_truth"][
                        "chain_iid_token_lvl"
                    ],
                    is_real_atom=pipeline_output["confidence_feats"]["is_real_atom"],
                    example_id=example_id,
                    confidence_loss_cfg=self.cfg.trainer.loss.confidence_loss,
                )
                atom_array_list = annotate_atom_array_b_factor_with_plddt(
                    atom_array_stack,
                    confidence_outs["plddt"],
                    pipeline_output["confidence_feats"]["is_real_atom"],
                )
                logging.info(f"Annotated PLDDT scores into B-factors for {example_id}. Forcing one model per file to accommodate separate b_factors in each model.")
                self.one_model_per_file = True
                confidence_outs["confidence_df"].to_csv(
                    self.cif_out_dir / f"{example_id}.score", index=False
                )
                ranked_logger.info(
                    f"Confidence metrics for {example_id} written to {self.cif_out_dir / example_id}.score."
                )

            if self.dump_predictions:
                dump_structures(
                    atom_arrays=atom_array_stack if not "plddt" in network_output else atom_array_list,
                    base_path=self.cif_out_dir / example_id,
                    one_model_per_file=self.one_model_per_file,
                )

            if self.dump_trajectories:
                dump_trajectories(
                    trajectory_list=network_output["X_denoised_L_traj"],
                    atom_array=pipeline_output["atom_array"],
                    base_path=self.cif_out_dir / f"{example_id}_denoised",
                    post_process_function=self.post_process_atom_array,
                )
                dump_trajectories(
                    trajectory_list=network_output["X_noisy_L_traj"],
                    atom_array=pipeline_output["atom_array"],
                    base_path=self.cif_out_dir / f"{example_id}_noisy",
                    post_process_function=self.post_process_atom_array,
                )

            ranked_logger.info(
                f"Outputs for {example_id} written to {self.cif_out_dir / example_id}."
            )
    