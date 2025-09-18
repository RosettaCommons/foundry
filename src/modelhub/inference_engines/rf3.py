import logging
from os import PathLike
from pathlib import Path

import hydra
import pandas as pd
import torch
from atomworks.io import parse
from atomworks.io.transforms.categories import category_to_dict
from lightning.fabric import seed_everything
from omegaconf import OmegaConf

from modelhub.inference_engines.base import InferenceEngine
from modelhub.model.RF3 import ShouldEarlyStopFn
from modelhub.utils.datasets import (
    assemble_distributed_inference_loader_from_list_of_paths,
)
from modelhub.utils.ddp import RankedLogger, set_accelerator_based_on_availability
from modelhub.utils.inference import (
    apply_conformer_and_template_selections,
    build_file_paths_for_prediction,
)
from modelhub.utils.io import (
    build_stack_from_atom_array_and_batched_coords,
    dump_structures,
    dump_trajectories,
)
from modelhub.utils.logging import print_config_tree
from modelhub.utils.predicted_error import (
    annotate_atom_array_b_factor_with_plddt,
    compile_af3_confidence_outputs,
    get_mean_atomwise_plddt,
)

logging.basicConfig(level=logging.INFO)
ranked_logger = RankedLogger(__name__, rank_zero_only=True)


def should_early_stop_by_mean_plddt(
    threshold: float, is_real_atom: torch.Tensor, max_value_of_plddt: float
) -> ShouldEarlyStopFn:
    """Returns a closure that triggers early stopping when mean pLDDT falls below the specified threshold."""

    def fn(confidence_outputs: dict, **kwargs):
        mean_plddt = get_mean_atomwise_plddt(
            plddt_logits=confidence_outputs["plddt_logits"].unsqueeze(0),
            is_real_atom=is_real_atom,
            max_value=max_value_of_plddt,
        )
        return (mean_plddt < threshold).item(), {
            "mean_plddt": mean_plddt.item(),
            "threshold": threshold,
        }

    return fn


class RF3InferenceEngine(InferenceEngine):
    """Class for inference with RF3. Evaluates a trained RF3 model on a set of spoofed CIFs."""

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
        template_selection: list[str] | str | None,
        ground_truth_conformer_selection: list[str] | str | None,
        num_steps: int,
        solver: str,
        print_config: bool,
        temp_dir: PathLike,
        seed: int,
        metrics_cfg: dict | OmegaConf,
        # Structure dumping arguments
        dump_predictions: bool,
        dump_trajectories: bool,
        one_model_per_file: bool,
        annotate_b_factor_with_plddt: bool,
        early_stopping_plddt_threshold: float | None,
        # Debugging
        raise_if_missing_msa_for_protein_of_length_n: int | None,
    ):
        """Initialize the Inference Engine for RF3.

        Note that for inference, we initialize the Hydra configuration from the checkpoint; we then override specific parameters based on the input arguments
        and inference-specific considerations.

        Args:
            ckpt_path: Path to the checkpoint file.
            out_dir: Directory for output files. If None, the current directory will be used.
            skip_existing: If True, only predict the structures that are not already in the output directory.
            num_nodes: Number of nodes for distributed inference.
            devices_per_node: Number of devices per node for distributed inference.

            n_recycles (int): Number of recycles for RF3.
            diffusion_batch_size (int): Diffusion batch size for RF3. Each predicted structure will be saved as a separate model within the same CIF file.
            residue_renaming_dict (dict): Dictionary of residue names to rename to avoid CCD clashes, e.g., {'ALA': 'L:1'}.
            template_selection (str): Selection syntax for token-level ground-truth template selection. If None, no residues will be selected.
            ground_truth_conformer_selection (str): Selection syntax for residues that should use ground truth conformers instead of generated ones.
                Uses AtomSelection format (e.g., "*/HEM" for all heme groups, "A1-10" for residues 1-10 in chain A).
                Must include all atoms within a given residue; we do not support partial residue selection for ground truth conformers.
                If None, no residues will use ground truth conformers.
            num_steps (int): Number of steps for sampling of the diffusion model. AF-3 uses 200; we see no degradation in performance with 50.
            solver (str): Solver to use for inference. We only support 'af3' for now.
            print_config (bool): Pretty-print the Hydra configs.
            temp_dir (PathLike): Temporary directory to store intermediate files.
            seed (int): Random seed for reproducibility / augmentation. If None, the default seed from the config will be used.

            dump_predictions (bool): Whether to dump structures (CIF files).
            dump_trajectories (bool): Whether to dump denoising trajectories.
            one_model_per_file (bool): If True, write each structure within a diffusion batch to its own CIF files.
                If False, include each structure within a diffusion batch as a separate model within one CIF file.
            annotate_b_factor_with_plddt (bool): If True, annotate the B-factor of the predicted structures with the atom-wise pLDDT scores.
            early_stopping_plddt_threshold (float | None): The value for average all-atom pLDDT after a single recycle that will trigger early-exit for that prediction.
                If the average pLDDT is below this value, the model will stop recycling and note in the output score file that early stopping was triggered.
                If None, early stopping will not be used (which may marginally speed up prediction times).
            raise_if_missing_msa_for_protein_of_length_n (int | None): If an MSA is missing for a protein of a given length, raise an error.
                If None, an error will not be raised. Useful for debugging and to ensure that a provided MSA is used.
        """
        if solver != "af3":
            # TODO: Port over additional solvers
            raise NotImplementedError(
                f"Solver {solver} not implemented. Only 'af3' is supported for inference."
            )

        # Load the training config from the checkpoint
        # TODO: Load checkpoint only once (instead of twice)
        ranked_logger.info(f"Loading checkpoint from {Path(ckpt_path).resolve()}...")
        checkpoint = torch.load(
            ckpt_path, "cpu", weights_only=False
        )  # We only extract the `train_cfg` from the checkpoint initially
        self.cfg = OmegaConf.create(checkpoint["train_cfg"])

        self.paths = build_file_paths_for_prediction(
            input=inputs,
            temp_dir=temp_dir,
            existing_outputs_dir=out_dir if skip_existing else None,
        )

        # Override specific parameters within the Hydra config:
        #  (a) based on the input arguments
        self.cfg.model.net.inference_sampler.num_timesteps = num_steps
        self.cfg.model.net.inference_sampler.solver = solver
        self.cfg.trainer.num_nodes = num_nodes
        self.cfg.trainer.devices_per_node = devices_per_node
        self.cfg.trainer.precision = "bf16-mixed"  # HACK: Temporary hack until our checkpoint configs are updated
        self.cfg.trainer._target_ = "modelhub.trainers.rf3.RF3TrainerWithConfidence"  # HACK: Enables inference with 9/21 checkpoint for benchmarking
        self.cfg.model.net._target_ = "modelhub.model.RF3.RF3WithConfidence"  # HACK: Enables inference with 9/21 checkpoint for benchmarking

        # We don't want to compute all of the training metrics, since they may error during inference
        self.cfg.trainer["metrics"] = {}

        set_accelerator_based_on_availability(self.cfg)

        # (b) based on the dataset (we will apply when constructing the pipeline)
        self.dataset_overrides = {
            "diffusion_batch_size": diffusion_batch_size,
            "n_recycles": n_recycles,
            "raise_if_missing_msa_for_protein_of_length_n": raise_if_missing_msa_for_protein_of_length_n,  # Don't raise if MSA is missing
            "undesired_res_names": [],
            "template_noise_scales": {
                "atomized": 1e-5,  # No noise (TODO: Make configurable)
                "not_atomized": 1e-5,  # No noise (TODO: Make configurable)
            },
            "allowed_chain_types_for_conditioning": None,  # Avoid random conditioning
            "protein_msa_dirs": [],  # To be consistent with installing on non-IPD clusters
            "rna_msa_dirs": [],  # To be consistent with installing on non-IPD clusters
            "p_give_polymer_ref_conf": 0.0,  # Never randomly give ground truth conformers
            "p_give_non_polymer_ref_conf": 0.0,  # Never randomly give ground truth conformers
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

        if metrics_cfg is not None:
            self.cfg.trainer["metrics"].update(metrics_cfg)

        # ... instantiate the trainer with the (modified) configuration
        self.trainer = hydra.utils.instantiate(
            self.cfg.trainer,
            _convert_="partial",
            _recursive_=False,
        )

        # Early stopping
        self.early_stopping_plddt_threshold = early_stopping_plddt_threshold

        # Paths
        self.cif_out_dir = Path(out_dir) if out_dir else Path("./")
        self.ckpt_path = ckpt_path

        # Rename residues
        self.residue_renaming_dict = residue_renaming_dict
        self.temp_dir = Path(temp_dir)

        self.template_selection = template_selection
        self.ground_truth_conformer_selection = ground_truth_conformer_selection

        # Structure dumping
        self.dump_predictions = dump_predictions
        self.dump_trajectories = dump_trajectories
        self.one_model_per_file = one_model_per_file
        self.annotate_b_factor_with_plddt = annotate_b_factor_with_plddt

    def construct_pipeline(self):
        """Construct the RF3 inference pipeline.

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

        return parse(path_to_structure, hydrogen_policy="remove", keep_cif_block=True)

    # Removed class-local selection helpers; use utility functions instead

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

        # Ensure optimizer isn't loaded
        self.trainer.state["optimizer"] = None
        self.trainer.state["train_cfg"].model.optimizer = None

        self.trainer.setup_model_optimizers_and_schedulers()
        self.trainer.state["model"].eval()

        # ==============================================================================
        # Prepare pipeline and inference loader
        # ==============================================================================

        ranked_logger.info("Building Transform pipeline...")

        # Construct the RF3 inference pipeline
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
            example_id = path_to_structure.name[: path_to_structure.name.rfind(".")]

            out = self.parse_from_path(path_to_structure)

            # ... get the atom array and set NaN coordinates to random
            atom_array = (
                out["assemblies"]["1"][0]
                if "assemblies" in out
                else out["asym_unit"][0]
            )

            # ... extract template information from the CIF file, if present
            template_selection_from_CIF = (
                category_to_dict(out["cif_block"], "template_selection")
                if "cif_block" in out
                else {}
            )
            ground_truth_conformer_selection_from_CIF = (
                category_to_dict(out["cif_block"], "ground_truth_conformer_selection")
                if "cif_block" in out
                else {}
            )

            # First, apply the template selection from the CIF file
            atom_array = apply_conformer_and_template_selections(
                atom_array,
                template_selection=list(
                    template_selection_from_CIF.get("template_selection", [])
                ),
                ground_truth_conformer_selection=list(
                    ground_truth_conformer_selection_from_CIF.get(
                        "ground_truth_conformer_selection", []
                    )
                ),
            )

            # Then, apply the template selection from the command line, if provided
            atom_array = apply_conformer_and_template_selections(
                atom_array,
                template_selection=self.template_selection,
                ground_truth_conformer_selection=self.ground_truth_conformer_selection,
            )

            # ... assemble the pipeline input in a format compatible with the DataHub pipeline
            pipeline_input = {
                "example_id": example_id,
                "atom_array": atom_array,
                "chain_info": out["chain_info"],
            }

            # ... run dataloading and featurization
            pipeline_output = pipeline(pipeline_input)

            should_early_stop_fn = None
            if (
                "confidence_feats" in pipeline_output
                and self.early_stopping_plddt_threshold
                and self.early_stopping_plddt_threshold > 0
                and "confidence_feats" in pipeline_output
            ):
                should_early_stop_fn = should_early_stop_by_mean_plddt(
                    self.early_stopping_plddt_threshold,
                    pipeline_output["confidence_feats"]["is_real_atom"],
                    self.cfg.trainer.loss.confidence_loss.plddt.max_value,
                )

            # Model inference
            with torch.no_grad():
                pipeline_output = self.trainer.fabric.to_device(pipeline_output)
                if should_early_stop_fn:
                    valid_step_outs = self.trainer.validation_step(
                        batch=pipeline_output,
                        batch_idx=0,
                        compute_metrics=True,
                        should_early_stop_fn=should_early_stop_fn,
                    )
                else:
                    valid_step_outs = self.trainer.validation_step(
                        batch=pipeline_output,
                        batch_idx=0,
                        compute_metrics=True,
                    )
                network_output = valid_step_outs["network_output"]
                metrics_output = valid_step_outs["metrics_output"]
                # TODO: Log `metrics_output` to a file (or store directly within the CIF file)
                df_to_save = pd.DataFrame([metrics_output])
                df_to_save.to_csv(
                    self.cif_out_dir / f"{example_id}_metrics.csv",
                    index=False,
                )
            if network_output.get("early_stopped", False):
                # TODO: Rework how we save outputs so it's easy for users
                ranked_logger.warning(
                    f"Early stopping triggered for {example_id} with mean pLDDT {network_output['mean_plddt']:.2f} < {self.early_stopping_plddt_threshold:.2f}!"
                )
                # Prune keys with null values...
                dict_to_save = {
                    k: v for k, v in network_output.items() if v is not None
                }
                # ... then convert to a DataFrame and save to CSV
                df_to_save = pd.DataFrame([dict_to_save])
                df_to_save.to_csv(
                    self.cif_out_dir / f"{example_id}.score",
                    index=False,
                )

                # (Skip to the next example that we will predict)
                continue

            # ... build the predicted AtomArrayStack
            atom_array_stack = build_stack_from_atom_array_and_batched_coords(
                network_output["X_L"], pipeline_output["atom_array"]
            )

            # (Sometimes we will instead need a list of AtomArrays, e.g., for B-factor annotation)
            atom_array_list = None
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

                if self.annotate_b_factor_with_plddt:
                    # Annotate the B-factors of the predicted structures with the pLDDT scores
                    # (Forces one model per file, if `one_model_per_file` is False)
                    atom_array_list = annotate_atom_array_b_factor_with_plddt(
                        atom_array_stack,
                        confidence_outs["plddt"],
                        pipeline_output["confidence_feats"]["is_real_atom"],
                    )
                    logging.info(
                        f"Annotated PLDDT scores into B-factors for {example_id}. Forcing one model per file to accommodate separate b_factors in each model."
                    )
                    self.one_model_per_file = True

                confidence_outs["confidence_df"].to_csv(
                    self.cif_out_dir / f"{example_id}.score", index=False
                )

                ranked_logger.info(
                    f"Confidence metrics for {example_id} written to {self.cif_out_dir / example_id}.score."
                )

            if self.dump_predictions:
                dump_structures(
                    atom_arrays=atom_array_list or atom_array_stack,
                    base_path=self.cif_out_dir / example_id,
                    one_model_per_file=self.one_model_per_file,
                )

            if self.dump_trajectories:
                dump_trajectories(
                    trajectory_list=network_output["X_denoised_L_traj"],
                    atom_array=pipeline_output["atom_array"],
                    base_path=self.cif_out_dir / f"{example_id}_denoised",
                )
                dump_trajectories(
                    trajectory_list=network_output["X_noisy_L_traj"],
                    atom_array=pipeline_output["atom_array"],
                    base_path=self.cif_out_dir / f"{example_id}_noisy",
                )

            ranked_logger.info(
                f"Outputs for {example_id} written to {self.cif_out_dir / example_id}!"
            )
