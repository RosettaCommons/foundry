import json
import logging
import re
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import TextIO

import pandas as pd
import torch
import torch.distributed as dist
from atomworks.io.utils.io_utils import to_cif_file
from atomworks.ml.preprocessing.msa.finding import (
    get_msa_depth_and_ext_from_folder,
    get_msa_dirs_from_env,
)
from atomworks.ml.samplers import LoadBalancedDistributedSampler
from biotite.structure import AtomArray, AtomArrayStack
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from foundry.inference_engines.base import BaseInferenceEngine
from foundry.metrics.metric import MetricManager
from foundry.utils.ddp import RankedLogger
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
    compile_af3_style_confidence_outputs,
    get_mean_atomwise_plddt,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
ranked_logger = RankedLogger(__name__, rank_zero_only=True)

# Default metrics configuration for RF3 inference (ptm, iptm, clashing chains)
DEFAULT_RF3_METRICS_CFG = {
    "ptm": {"_target_": "rf3.metrics.predicted_error.ComputePTM"},
    "iptm": {"_target_": "rf3.metrics.predicted_error.ComputeIPTM"},
    "count_clashing_chains": {
        "_target_": "rf3.metrics.clashing_chains.CountClashingChains"
    },
}


def dump_json_compact_arrays(obj: dict, f: TextIO) -> None:
    """Dump JSON with indented structure but compact arrays (AF3 style).

    Arrays are written on single lines instead of one element per line.
    """
    # First dump with indent to get structure
    json_str = json.dumps(obj, indent=2)
    # Collapse arrays onto single lines using regex
    # Match arrays that span multiple lines and collapse them
    pattern = re.compile(r"\[\s*\n\s*([^\[\]]*?)\s*\n\s*\]", re.DOTALL)
    while pattern.search(json_str):
        json_str = pattern.sub(
            lambda m: "["
            + ",".join(item.strip() for item in m.group(1).split(","))
            + "]",
            json_str,
        )
    f.write(json_str)


def compute_ranking_score(
    iptm: float | None,
    ptm: float | None,
    has_clash: bool,
) -> float:
    """Compute ranking score.

    Formula: 0.8 * ipTM + 0.2 * pTM - 100 * has_clash

    For single-chain predictions where ipTM is None, uses pTM only.
    """
    if iptm is None:
        # Single chain - use pTM only
        iptm = ptm if ptm is not None else 0.0
    if ptm is None:
        ptm = 0.0
    return 0.8 * iptm + 0.2 * ptm - 100 * int(has_clash)


@dataclass
class RF3Output:
    """Output container for RF3 predictions, analogous to RFD3Output.

    Stores predicted structures and confidence metrics in AlphaFold3-compatible format.
    """

    example_id: str
    atom_array: AtomArray
    summary_confidences: dict = field(default_factory=dict)
    confidences: dict | None = None
    sample_idx: int = 0
    seed: int = 0

    def dump(
        self,
        out_dir: Path,
        file_type: str = "cif",
        dump_full_confidences: bool = True,
    ) -> None:
        """Save output to disk in AlphaFold3-compatible format.

        Args:
            out_dir: Directory to save outputs to.
            file_type: File type for structure output ("cif" or "cif.gz").
            dump_full_confidences: Whether to save full per-atom confidences.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        sample_name = f"{self.example_id}_seed-{self.seed}_sample-{self.sample_idx}"
        base_path = out_dir / sample_name

        # Save structure
        to_cif_file(
            self.atom_array,
            f"{base_path}_model",
            file_type=file_type,
            include_entity_poly=False,
        )

        # Save summary_confidences.json
        with open(f"{base_path}_summary_confidences.json", "w") as f:
            dump_json_compact_arrays(self.summary_confidences, f)

        # Save confidences.json (optional, for full per-atom data)
        if dump_full_confidences and self.confidences:
            with open(f"{base_path}_confidences.json", "w") as f:
                dump_json_compact_arrays(self.confidences, f)


def dump_ranking_scores(
    outputs: list[RF3Output],
    out_dir: Path,
    example_id: str,
) -> None:
    """Write {example_id}_ranking_scores.csv with ranking scores for all samples."""
    rows = [
        {
            "seed": o.seed,
            "sample": o.sample_idx,
            "ranking_score": o.summary_confidences.get("ranking_score"),
        }
        for o in outputs
    ]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{example_id}_ranking_scores.csv", index=False)


def dump_top_ranked_outputs(
    outputs: list[RF3Output],
    out_dir: Path,
    example_id: str,
    file_type: str = "cif",
) -> RF3Output:
    """Copy the top-ranked model and summary to the top-level directory.

    Returns the top-ranked RF3Output.
    """
    # Find the output with the highest ranking score
    best_output = max(
        outputs,
        key=lambda o: o.summary_confidences.get("ranking_score", float("-inf")),
    )

    # Save top-ranked model at top level
    to_cif_file(
        best_output.atom_array,
        out_dir / f"{example_id}_model",
        file_type=file_type,
        include_entity_poly=False,
    )

    # Save top-ranked summary_confidences at top level
    with open(out_dir / f"{example_id}_summary_confidences.json", "w") as f:
        dump_json_compact_arrays(best_output.summary_confidences, f)

    # Save top-ranked full confidences at top level (if present)
    if best_output.confidences:
        with open(out_dir / f"{example_id}_confidences.json", "w") as f:
            dump_json_compact_arrays(best_output.confidences, f)

    return best_output


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


class RF3InferenceEngine(BaseInferenceEngine):
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
        # Model parameters
        n_recycles: int = 10,
        diffusion_batch_size: int = 5,
        num_steps: int = 50,
        # Templating, MSAs, etc.
        template_noise_scale: float = 1e-5,
        raise_if_missing_msa_for_protein_of_length_n: int | None = None,
        # Output control
        compress_outputs: bool = False,
        early_stopping_plddt_threshold: float | None = None,
        # Metrics
        metrics_cfg: dict | OmegaConf | MetricManager | str | None = "default",
        **kwargs,
    ):
        """Initialize inference engine and load model.

        Model config is loaded from checkpoint and overridden with parameters provided here.

        Args:
          n_recycles: Number of recycles. Defaults to ``10``.
          diffusion_batch_size: Number of structures to generate per input. Defaults to ``5``.
          num_steps: Number of diffusion steps. Defaults to ``50``.
          template_noise_scale: Noise scale for template coordinates. Defaults to ``1e-5``.
          raise_if_missing_msa_for_protein_of_length_n: Debug flag for MSA checking. Defaults to ``None``.
          compress_outputs: Whether to gzip output files. Defaults to ``False``.
          early_stopping_plddt_threshold: Stop early if pLDDT below threshold. Defaults to ``None``.
          metrics_cfg: Metrics configuration. Can be:
              - "default" to use standard RF3 metrics (ptm, iptm, clashing chains)
              - dict/OmegaConf with Hydra configs
              - Pre-instantiated MetricManager
              - None (no metrics).
              Defaults to ``"default"``.
          **kwargs: Additional arguments passed to BaseInferenceEngine:
              - ckpt_path (PathLike, required): Path to model checkpoint.
              - seed (int | None): Random seed. If None, uses external RNG state. Defaults to ``None``.
              - num_nodes (int): Number of nodes for distributed inference. Defaults to ``1``.
              - devices_per_node (int): Number of devices per node. Defaults to ``1``.
              - verbose (bool): If True, show detailed logging and config trees. Defaults to ``False``.
        """
        # set MSA directories from environment variable only
        if env_var_msa_dirs := get_msa_dirs_from_env(raise_if_not_set=False):
            override_msa_dirs = [str(msa_dir) for msa_dir in env_var_msa_dirs]
            ranked_logger.debug(
                f"Using MSA directories from environment variable: {override_msa_dirs}"
            )
        else:
            override_msa_dirs = []
            ranked_logger.debug(
                "No MSA directories set (LOCAL_MSA_DIRS env var not found)"
            )

        super().__init__(
            transform_overrides={
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
                "use_element_for_atom_names_of_atomized_tokens": True,
            },
            inference_sampler_overrides={
                "num_timesteps": num_steps,
            },
            **kwargs,
        )

        # remove loss override if present (i.e. keep from checkpoint)
        self.overrides["trainer"].pop("loss", None)

        # Store metrics config for later - will be set directly on trainer in initialize()
        self._metrics_cfg = metrics_cfg

        # Dataset overrides
        self.early_stopping_plddt_threshold = early_stopping_plddt_threshold
        self.compress_outputs = compress_outputs

    def initialize(self):
        # Log checkpoint path on first init (base class logger may be suppressed in quiet mode)
        if not self.initialized_:
            ranked_logger.info(
                f"Loading checkpoint from {Path(self.ckpt_path).resolve()}..."
            )

        cfg = super().initialize()

        if cfg is not None:
            self.cfg = cfg  # store for later use

            # Set trainer metrics directly based on what was requested
            # This bypasses the OmegaConf merge issue with empty dicts
            if isinstance(self._metrics_cfg, MetricManager):
                # Already instantiated - use directly
                self.trainer.metrics = self._metrics_cfg
            elif self._metrics_cfg == "default":
                # Use default RF3 metrics (ptm, iptm, clashing chains)
                self.trainer.metrics = MetricManager.instantiate_from_hydra(
                    metrics_cfg=DEFAULT_RF3_METRICS_CFG
                )
            elif self._metrics_cfg is not None:
                # Hydra config dict - instantiate MetricManager
                self.trainer.metrics = MetricManager.instantiate_from_hydra(
                    metrics_cfg=self._metrics_cfg
                )
            else:
                # No metrics requested - disable them
                self.trainer.metrics = None

        return cfg

    def run(
        self,
        inputs: (
            InferenceInput
            | list[InferenceInput]
            | AtomArray
            | list[AtomArray]
            | PathLike
            | list[PathLike]
        ),
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
        cyclic_chains: list[str] = [],
    ) -> dict[str, dict] | None:
        """Run inference on inputs.

        Requires a pre-initialized inference engine.

        Args:
          inputs: Single/list of InferenceInput objects, AtomArray objects, file paths, or directory.
          out_dir: Output directory. If None, returns results as an AtomArray and dictionaries of metrics. Defaults to ``None``.
          dump_predictions: Whether to save predicted structures. Defaults to ``True``.
          dump_trajectories: Whether to save diffusion trajectories. Defaults to ``False``.
          one_model_per_file: Save each model in separate file. Defaults to ``False``.
          annotate_b_factor_with_plddt: Write pLDDT to B-factor column. Defaults to ``False``.
          sharding_pattern: Sharding pattern for output organization. Defaults to ``None``.
          skip_existing: Skip inputs with existing outputs. Requires ``out_dir`` to be set. If ``True`` when ``out_dir=None``, a warning is logged and skipping is disabled. Defaults to ``False``.
          template_selection: Template selection override. Defaults to ``None``.
          ground_truth_conformer_selection: Conformer selection override. Defaults to ``None``.
          cyclic_chains: List of chain IDs to cyclize. Defaults to ``[]``.

        Returns:
          If ``out_dir`` is None: Dict mapping example_id to list of RF3Output objects.
          If ``out_dir`` is set: None (results saved to disk).
        """
        self.initialize()

        # Setup output directory if provided
        out_dir = Path(out_dir) if out_dir else None
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            ranked_logger.info(f"Outputs will be written to {out_dir.resolve()}.")
        if not out_dir:
            ranked_logger.warning(
                "out_dir is None - results will be returned in memory! If you want to save to disk, please provide an out_dir."
            )

        # Validate skip_existing configuration
        if skip_existing and out_dir is None:
            ranked_logger.warning(
                "skip_existing=True requires out_dir to be set. "
                "Disabling skip_existing for in-memory inference mode."
            )
            skip_existing = False

        # Determine file type based on compression setting
        file_type = "cif.gz" if self.compress_outputs else "cif"

        # Convert inputs to InferenceInput objects
        if isinstance(inputs, InferenceInput):
            inference_inputs = [inputs]
        elif isinstance(inputs, list) and all(
            isinstance(i, InferenceInput) for i in inputs
        ):
            inference_inputs = inputs
        elif isinstance(inputs, AtomArray):
            # Single AtomArray - convert to InferenceInput
            inference_inputs = [
                InferenceInput.from_atom_array(
                    inputs,
                    template_selection=template_selection,
                    ground_truth_conformer_selection=ground_truth_conformer_selection,
                )
            ]
        elif isinstance(inputs, list) and all(isinstance(i, AtomArray) for i in inputs):
            # List of AtomArrays - convert each to InferenceInput
            inference_inputs = [
                InferenceInput.from_atom_array(
                    arr,
                    example_id=f"inference_{i}",
                    template_selection=template_selection,
                    ground_truth_conformer_selection=ground_truth_conformer_selection,
                )
                for i, arr in enumerate(inputs)
            ]
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
        if cyclic_chains:
            for input_spec in inference_inputs:
                input_spec.cyclic_chains = cyclic_chains

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
            num_samples = (
                len(atom_array_stack)
                if isinstance(atom_array_stack, AtomArrayStack)
                else 1
            )

            # Build RF3Output objects for each sample
            rf3_outputs: list[RF3Output] = []
            for sample_idx in range(num_samples):
                # Get atom array for this sample
                if isinstance(atom_array_stack, AtomArrayStack):
                    sample_atom_array = atom_array_stack[sample_idx]
                else:
                    sample_atom_array = atom_array_stack

                # Compile confidence outputs in AF3 format (if available)
                summary_confidences = {}
                confidences = None
                if "plddt" in network_output:
                    conf_outs = compile_af3_style_confidence_outputs(
                        plddt_logits=network_output["plddt"],
                        pae_logits=network_output["pae"],
                        pde_logits=network_output["pde"],
                        chain_iid_token_lvl=pipeline_output["ground_truth"][
                            "chain_iid_token_lvl"
                        ],
                        is_real_atom=pipeline_output["confidence_feats"][
                            "is_real_atom"
                        ],
                        atom_array=pipeline_output["atom_array"],
                        confidence_loss_cfg=self.cfg.trainer.loss.confidence_loss,
                        batch_idx=sample_idx,
                    )
                    summary_confidences = conf_outs["summary_confidences"]
                    confidences = conf_outs["confidences"]

                    # Annotate b-factor with pLDDT if requested
                    if annotate_b_factor_with_plddt:
                        atom_array_list = annotate_atom_array_b_factor_with_plddt(
                            atom_array_stack,
                            conf_outs["plddt"],
                            pipeline_output["confidence_feats"]["is_real_atom"],
                        )
                        sample_atom_array = atom_array_list[sample_idx]

                # Add metrics (ptm, iptm, has_clash) to summary_confidences
                if metrics_output:
                    ptm_key = f"ptm.ptm_{sample_idx}"
                    iptm_key = f"iptm.iptm_{sample_idx}"
                    clash_key = f"count_clashing_chains.has_clash_{sample_idx}"

                    ptm_val = metrics_output.get(ptm_key)
                    iptm_val = metrics_output.get(iptm_key)
                    has_clash = bool(metrics_output.get(clash_key, 0))

                    # Convert to native Python floats for JSON serialization
                    ptm = float(ptm_val) if ptm_val is not None else None
                    iptm = float(iptm_val) if iptm_val is not None else None

                    summary_confidences["ptm"] = ptm
                    summary_confidences["iptm"] = iptm
                    summary_confidences["has_clash"] = has_clash

                    ranking_score = compute_ranking_score(
                        iptm=iptm,
                        ptm=ptm,
                        has_clash=has_clash,
                    )
                    summary_confidences["ranking_score"] = round(ranking_score, 4)

                rf3_outputs.append(
                    RF3Output(
                        example_id=input_spec.example_id,
                        atom_array=sample_atom_array,
                        summary_confidences=summary_confidences,
                        confidences=confidences,
                        sample_idx=sample_idx,
                        seed=self.seed if self.seed is not None else 0,
                    )
                )

            # Save or return results
            if out_dir:
                # Save to disk in AlphaFold3-style directory structure
                # Top-level: ranking_scores.csv, best model, best summary
                dump_ranking_scores(rf3_outputs, example_out_dir, input_spec.example_id)
                dump_top_ranked_outputs(
                    rf3_outputs,
                    example_out_dir,
                    input_spec.example_id,
                    file_type=file_type,
                )

                # Per-sample subdirectories
                if dump_predictions:
                    for rf3_out in rf3_outputs:
                        sample_subdir = (
                            example_out_dir
                            / f"seed-{rf3_out.seed}_sample-{rf3_out.sample_idx}"
                        )
                        rf3_out.dump(
                            out_dir=sample_subdir,
                            file_type=file_type,
                            dump_full_confidences=True,
                        )

                if dump_trajectories:
                    dump_structures(
                        atom_arrays=network_output["X_denoised_L_traj"],
                        base_path=example_out_dir / "denoised",
                        one_model_per_file=True,
                        file_type=file_type,
                    )
                    dump_structures(
                        atom_arrays=network_output["X_noisy_L_traj"],
                        base_path=example_out_dir / "noisy",
                        one_model_per_file=True,
                        file_type=file_type,
                    )

                ranked_logger.info(
                    f"Outputs for {input_spec.example_id} written to {example_out_dir}!"
                )
            else:
                # Store in memory - return list of RF3Output objects
                results[input_spec.example_id] = rf3_outputs

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
