import logging
from os import PathLike
from pathlib import Path

import hydra
import pandas as pd
import torch
import torch.distributed as dist
from atomworks.ml.preprocessing.msa.finding import (
    get_msa_depth_and_ext_from_folder,
    get_msa_dirs_from_env,
)
from atomworks.ml.samplers import LoadBalancedDistributedSampler
from lightning.fabric import seed_everything
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from modelhub.utils.ddp import RankedLogger, set_accelerator_based_on_availability
from modelhub.utils.logging import print_config_tree
from rf3.model.RF3 import ShouldEarlyStopFn
from rf3.utils.inference import (
    InferenceInput,
    InferenceInputDataset,
    prepare_inference_inputs_from_paths,
)
from rf3.utils.io import (
    build_stack_from_atom_array_and_batched_coords,
    dump_structures,
    get_sharded_output_path,
)
from rf3.utils.predicted_error import (
    annotate_atom_array_b_factor_with_plddt,
    compile_af3_confidence_outputs,
    get_mean_atomwise_plddt,
)
from modelhub.metrics.metric import MetricManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
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


class RF3InferenceEngine:
    """RF3 inference engine.

    Separates model setup (expensive, once) from inference (can run multiple times).

    Usage:
      # Setup once
      engine = RF3InferenceEngine(
          ckpt_path="rf3_latest.pt",
          n_recycles=10,
          diffusion_batch_size=5,
      )

      # Run inference multiple times with different inputs
      results1 = engine.run(inputs="path/to/cifs", out_dir="./predictions")
      results2 = engine.run(inputs=InferenceInput.from_atom_array(array), out_dir=None)
      results3 = engine.run(inputs=[input1, input2], out_dir="./more_predictions")
    """

    def __init__(
        self,
        ckpt_path: PathLike,
        # Model parameters
        n_recycles: int = 10,
        diffusion_batch_size: int = 5,
        num_steps: int = 50,
        seed: int | None = None,
        template_noise_scale: float = 1e-5,
        early_stopping_plddt_threshold: float | None = None,
        metrics_cfg: dict | OmegaConf | MetricManager | None = None,
        num_nodes: int = 1,
        devices_per_node: int = 1,
        cyclic_chains: list[str] = [],
        # Debug
        print_config: bool = False,
        raise_if_missing_msa_for_protein_of_length_n: int | None = None,
    ):
        """Initialize inference engine and load model.

        Model config is loaded from checkpoint and overridden with parameters provided here.

        Args:
          ckpt_path: Path to model checkpoint.
          n_recycles: Number of recycles. Defaults to ``10``.
          diffusion_batch_size: Number of structures to generate per input. Defaults to ``5``.
          num_steps: Number of diffusion steps. Defaults to ``50``.
          seed: Random seed. If None, uses external RNG state. Defaults to ``None``.
          template_noise_scale: Noise scale for template coordinates. Defaults to ``1e-5``.
          early_stopping_plddt_threshold: Stop early if pLDDT below threshold. Defaults to ``None``.
          metrics_cfg: Metrics configuration. Can be:
              - dict/OmegaConf with Hydra configs
              - Pre-instantiated MetricManager
              - None (no metrics).
              Defaults to ``None``.
          num_nodes: Number of nodes for distributed inference. Defaults to ``1``.
          devices_per_node: Number of devices per node. Defaults to ``1``.
          print_config: Whether to print config trees. Defaults to ``False``.
          raise_if_missing_msa_for_protein_of_length_n: Debug flag for MSA checking. Defaults to ``None``.
        """
        # Load checkpoint and config
        ranked_logger.info(f"Loading checkpoint from {Path(ckpt_path).resolve()}...")
        checkpoint = torch.load(ckpt_path, "cpu", weights_only=False)
        self.cfg = OmegaConf.create(checkpoint["train_cfg"])

        # Override config with inference parameters
        self.cfg.model.net.inference_sampler.num_timesteps = num_steps
        self.cfg.trainer.num_nodes = num_nodes
        self.cfg.trainer.devices_per_node = devices_per_node

        # Set metrics - can be dict/OmegaConf or MetricManager
        # Store MetricManager separately since OmegaConf can't serialize it
        self._custom_metric_manager = None
        if isinstance(metrics_cfg, MetricManager):
            # Already instantiated - store separately and pass to trainer later
            self._custom_metric_manager = metrics_cfg
            self.cfg.trainer["metrics"] = {}  # Empty dict in config
        elif metrics_cfg is not None:
            # Hydra config dict
            self.cfg.trainer["metrics"] = metrics_cfg
        else:
            self.cfg.trainer["metrics"] = {}

        set_accelerator_based_on_availability(self.cfg)

        # set MSA directories
        if env_var_msa_dirs := get_msa_dirs_from_env(raise_if_not_set=False):
            override_msa_dirs = [str(msa_dir) for msa_dir in env_var_msa_dirs]
            ranked_logger.info(
                f"Using MSA directories from environment variable: {override_msa_dirs}"
            )
        else:
            override_msa_dirs = [
                "/projects/msa/hhblits",
                "/projects/msa/mmseqs_gpu",
                "/projects/msa/lab",
            ]
            ranked_logger.info(f"Using default MSA directories: {override_msa_dirs}")

        # Dataset overrides
        self.dataset_overrides = {
            "diffusion_batch_size": diffusion_batch_size,
            "n_recycles": n_recycles,
            "raise_if_missing_msa_for_protein_of_length_n": raise_if_missing_msa_for_protein_of_length_n,
            "undesired_res_names": [],
            "template_noise_scales": {
                "atomized": template_noise_scale,
                "not_atomized": template_noise_scale,
            },
            "allowed_chain_types_for_conditioning": None,
            "protein_msa_dirs": [
                {
                    "dir": msa_dir,
                    "extension": extension.value,
                    "directory_depth": depth,
                }
                for msa_dir, depth, extension in [
                    (msa_dir, *get_msa_depth_and_ext_from_folder(Path(msa_dir)))
                    for msa_dir in override_msa_dirs
                ]
            ],
            "rna_msa_dirs": [],
            # (Paranoia - in validation, these should be set correctly anyhow)
            "p_give_polymer_ref_conf": 0.0,
            "p_give_non_polymer_ref_conf": 0.0,
            "p_dropout_ref_conf": 0.0,
        }

        self.cyclic_chains = cyclic_chains

        self.print_config = print_config

        # Set random seed (only if seed is not None)
        if seed is not None or self.cfg.seed is not None:
            seed = seed or self.cfg.seed
            ranked_logger.info(f"Seeding everything with seed={seed}...")
            seed_everything(seed, workers=True, verbose=True)
        else:
            ranked_logger.info("Seed is None - using external RNG state")

        # Instantiate trainer
        ranked_logger.info("Instantiating trainer...")
        if self.print_config:
            print_config_tree(
                self.cfg.trainer, resolve=True, title="INFERENCE TRAINER CONFIGURATION"
            )

        self.trainer = hydra.utils.instantiate(
            self.cfg.trainer,
            _convert_="partial",
            _recursive_=False,
        )

        # If we have a custom MetricManager, override the trainer's metrics
        if self._custom_metric_manager is not None:
            self.trainer.metrics = self._custom_metric_manager

        self.ckpt_path = ckpt_path
        self.early_stopping_plddt_threshold = early_stopping_plddt_threshold

        # Setup model
        ranked_logger.info("Setting up model...")
        self.trainer.fabric.launch()
        self.trainer.initialize_or_update_trainer_state({"train_cfg": self.cfg})
        self.trainer.construct_model()

        ranked_logger.info("Loading model weights from checkpoint...")
        self.trainer.load_checkpoint(checkpoint=checkpoint)

        # Ensure optimizer isn't loaded
        self.trainer.state["optimizer"] = None
        self.trainer.state["train_cfg"].model.optimizer = None

        self.trainer.setup_model_optimizers_and_schedulers()
        self.trainer.state["model"].eval()

        # Construct pipeline
        ranked_logger.info("Building Transform pipeline...")
        first_val_dataset_key, first_val_dataset = next(
            iter(self.cfg.datasets.val.items())
        )
        ranked_logger.info(
            f"Using settings from validation dataset: {first_val_dataset_key}."
        )

        assert (
            first_val_dataset.dataset.transform.is_inference
        ), "Inference must be enabled for the validation dataset."

        # Provide manual overrides to dataset config
        for key, value in self.dataset_overrides.items():
            first_val_dataset.dataset.transform[key] = value

        if self.print_config:
            print_config_tree(
                first_val_dataset.dataset.transform,
                resolve=True,
                title="INFERENCE TRANSFORM PIPELINE",
            )

        self.pipeline = hydra.utils.instantiate(
            first_val_dataset.dataset.transform,
        )

        ranked_logger.info("Model loaded and ready for inference.")

    def run(
        self,
        inputs: InferenceInput | list[InferenceInput] | PathLike | list[PathLike],
        # Output control
        out_dir: PathLike | None = None,
        dump_predictions: bool = True,
        dump_trajectories: bool = False,
        one_model_per_file: bool = False,
        annotate_b_factor_with_plddt: bool = False,
        sharding_pattern: str | None = None,
        skip_existing: bool = False,
        # Selection overrides (applied to all input types)
        template_selection: list[str] | str | None = None,
        ground_truth_conformer_selection: list[str] | str | None = None,
    ) -> dict[str, dict] | None:
        """Run inference on inputs.

        Requires a pre-initialized inference engine.

        Args:
          inputs: Single/list of InferenceInput objects, or file paths, or directory.
          out_dir: Output directory. If None, returns results as an AtomArray and dictionaries of metrics. Defaults to ``None``.
          dump_predictions: Whether to save predicted structures. Defaults to ``True``.
          dump_trajectories: Whether to save diffusion trajectories. Defaults to ``False``.
          one_model_per_file: Save each model in separate file. Defaults to ``False``.
          annotate_b_factor_with_plddt: Write pLDDT to B-factor column. Defaults to ``False``.
          sharding_pattern: Sharding pattern for output organization. Defaults to ``None``.
          skip_existing: Skip inputs with existing outputs. Defaults to ``False``.
          template_selection: Template selection override. Defaults to ``None``.
          ground_truth_conformer_selection: Conformer selection override. Defaults to ``None``.

        Returns:
          If ``out_dir`` is None: Dict mapping example_id to results dict.
          If ``out_dir`` is set: None (results saved to disk).
        """
        # Setup output directory if provided
        out_dir = Path(out_dir) if out_dir else None
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            ranked_logger.info(f"Outputs will be written to {out_dir.resolve()}.")

        # Convert inputs to InferenceInput objects
        if isinstance(inputs, InferenceInput):
            inference_inputs = [inputs]
        elif isinstance(inputs, list) and all(
            isinstance(i, InferenceInput) for i in inputs
        ):
            inference_inputs = inputs
        elif isinstance(inputs, (str, Path)) or (
            isinstance(inputs, list) and isinstance(inputs[0], (str, Path))
        ):
            inference_inputs = prepare_inference_inputs_from_paths(
                inputs=inputs,
                existing_outputs_dir=out_dir if skip_existing else None,
                sharding_pattern=sharding_pattern,
                template_selection=template_selection,
                ground_truth_conformer_selection=ground_truth_conformer_selection,
            )
        else:
            raise ValueError(f"Unsupported inputs type: {type(inputs)}")

        # Flag chains for cyclization if specified
        if self.cyclic_chains:
            for input_spec in inference_inputs:
                input_spec.cyclic_chains = self.cyclic_chains

        # make InferenceInputDataset
        inference_dataset = InferenceInputDataset(inference_inputs)
        ranked_logger.info(f"Found {len(inference_dataset)} structures to predict!")

        # make LoadBalancedDistributedSampler
        sampler = LoadBalancedDistributedSampler(
            dataset=inference_dataset,
            key_to_balance=inference_dataset.key_to_balance,
            num_replicas=self.trainer.fabric.world_size,
            rank=self.trainer.fabric.global_rank,
            drop_last=False,
        )

        loader = DataLoader(
            dataset=inference_dataset,
            sampler=sampler,
            batch_size=1,
            num_workers=0,  # multiprocessing is disabled since it shouldn't be hard to read InferenceInput objects
            collate_fn=lambda x: x,  # no collation since we're not batching
            pin_memory=True,
            drop_last=False,
        )

        # Prepare results dict (if returning in-memory)
        results = {} if out_dir is None else None

        # Main inference loop
        for batch_idx, input_spec in enumerate(loader):
            input_spec = input_spec[
                0
            ]  # since we're not batching, the loader returns a list of length 1
            ranked_logger.info(
                f"Predicting structure {batch_idx + 1}/{len(loader)}: {input_spec.example_id}"
            )

            # Create output directory for this example if saving to disk
            if out_dir:
                example_out_dir = get_sharded_output_path(
                    input_spec.example_id, out_dir, sharding_pattern
                )
                example_out_dir.mkdir(parents=True, exist_ok=True)

            # Run through Transform pipeline
            pipeline_output = self.pipeline(input_spec.to_pipeline_input())

            # Setup early stopping function if configured
            should_early_stop_fn = None
            if (
                "confidence_feats" in pipeline_output
                and self.early_stopping_plddt_threshold
                and self.early_stopping_plddt_threshold > 0
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

            # Handle early stopping
            if network_output.get("early_stopped", False):
                ranked_logger.warning(
                    f"Early stopping triggered for {input_spec.example_id} "
                    f"with mean pLDDT {network_output['mean_plddt']:.2f} < "
                    f"{self.early_stopping_plddt_threshold:.2f}!"
                )

                if out_dir:
                    # Save early stop info to disk
                    dict_to_save = {
                        k: v for k, v in network_output.items() if v is not None
                    }
                    df_to_save = pd.DataFrame([dict_to_save])
                    df_to_save.to_csv(example_out_dir / "score.csv", index=False)

                    df_to_save = pd.DataFrame([metrics_output])
                    df_to_save.to_csv(
                        example_out_dir / f"{input_spec.example_id}_metrics.csv",
                        index=False,
                    )
                else:
                    # Store in results dict
                    results[input_spec.example_id] = {
                        "early_stopped": True,
                        "mean_plddt": network_output["mean_plddt"],
                        "metrics": metrics_output,
                    }

                continue

            # Build predicted structures
            atom_array_stack = build_stack_from_atom_array_and_batched_coords(
                network_output["X_L"], pipeline_output["atom_array"]
            )

            # Compile confidence outputs if available
            atom_array_list = None
            confidence_df = None
            if "plddt" in network_output:
                confidence_outs = compile_af3_confidence_outputs(
                    plddt_logits=network_output["plddt"],
                    pae_logits=network_output["pae"],
                    pde_logits=network_output["pde"],
                    chain_iid_token_lvl=pipeline_output["ground_truth"][
                        "chain_iid_token_lvl"
                    ],
                    is_real_atom=pipeline_output["confidence_feats"]["is_real_atom"],
                    example_id=input_spec.example_id,
                    confidence_loss_cfg=self.cfg.trainer.loss.confidence_loss,
                )
                confidence_df = confidence_outs["confidence_df"]

                if annotate_b_factor_with_plddt:
                    atom_array_list = annotate_atom_array_b_factor_with_plddt(
                        atom_array_stack,
                        confidence_outs["plddt"],
                        pipeline_output["confidence_feats"]["is_real_atom"],
                    )
                    logging.info(
                        f"Annotated pLDDT scores into B-factors for {input_spec.example_id}. "
                        "Forcing one model per file."
                    )
                    one_model_per_file = True

            # Save or return results
            if out_dir:
                # Save to disk
                df_to_save = pd.DataFrame([metrics_output])
                df_to_save.to_csv(
                    example_out_dir / f"{input_spec.example_id}_metrics.csv",
                    index=False,
                )

                if confidence_df is not None:
                    confidence_df.to_csv(
                        example_out_dir / f"{input_spec.example_id}_score.csv",
                        index=False,
                    )

                if dump_predictions:
                    dump_structures(
                        atom_arrays=atom_array_list or atom_array_stack,
                        base_path=example_out_dir / input_spec.example_id,
                        one_model_per_file=one_model_per_file,
                    )

                if dump_trajectories:
                    dump_trajectories(
                        trajectory_list=network_output["X_denoised_L_traj"],
                        atom_array=pipeline_output["atom_array"],
                        base_path=example_out_dir / "denoised",
                    )
                    dump_trajectories(
                        trajectory_list=network_output["X_noisy_L_traj"],
                        atom_array=pipeline_output["atom_array"],
                        base_path=example_out_dir / "noisy",
                    )

                ranked_logger.info(
                    f"Outputs for {input_spec.example_id} written to {example_out_dir}!"
                )
            else:
                # Store in memory
                results[input_spec.example_id] = {
                    "predicted_structures": atom_array_list or atom_array_stack,
                    "metrics": metrics_output,
                    "confidence_scores": confidence_df,
                }

        # merge results across ranks
        self.trainer.fabric.barrier()
        if results is not None and dist.is_initialized():
            gathered_results = [None] * self.trainer.fabric.world_size
            dist.all_gather_object(
                gathered_results, results
            )  # returns a list of dicts, need to combine them
            gathered_results = {
                k: v for result in gathered_results for k, v in result.items()
            }  # combine the dicts into a single dict
            results = gathered_results

        return results
