"""
AF3 Inference Pipeline

This pipeline handles ALL model interactions:
- Inference (structures, embeddings, confidence)
- Loss computation (all types of losses)
- Optimization objectives
- Gradient computation
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import hydra
import torch
from biotite.structure import AtomArray
from omegaconf import OmegaConf
from rf3.chemical import NHEAVY
from rf3.metrics.metric_utils import (
    compute_mean_over_subsampled_pairs,
    create_interface_masks_2d,
)
from rf3.utils.io import build_stack_from_atom_array_and_batched_coords
from rf3.utils.predicted_error import compile_af3_confidence_outputs

from foundry.utils.ddp import RankedLogger, set_accelerator_based_on_availability
from rfo.backprop.inference_pipeline.optimization_forward import optimization_forward
from rfo.backprop.tools.af3 import cif2input
from rfo.backprop.tools.loss import (
    ContactLoss,
    DistanceConstraint,
    DistogramCCELoss,
    DistogramInterfaceEntropyLoss,
    WeightedInterfaceEntropyLoss,
    unbin_logits,
)

logging.basicConfig(level=logging.INFO)
ranked_logger = RankedLogger(__name__, rank_zero_only=True)


class AF3InferencePipeline:
    """
    AF3 inference pipeline that handles ALL model interactions.

    This class is the interface for:
    - Running inference (structures, embeddings, confidence)
    - Computing losses and objectives
    - Handling gradients for optimization

    Two types of forward passes:
    - forward: Direct model access with gradient support (for optimization)
    - run_inference: Through trainer's validation_step (for structures/confidence)
    """

    def __init__(
        self,
        ckpt_path: str,
        n_recycles: int = 10,
        diffusion_batch_size: int = 5,
        num_steps: int = 50,
        device: str = "cuda",
        load_ema_weights: bool = True,
    ):
        """Initialize the AF3 inference pipeline."""
        self.ckpt_path = Path(ckpt_path)
        self.n_recycles = n_recycles
        self.diffusion_batch_size = diffusion_batch_size
        self.num_steps = num_steps
        self.device = torch.device(device)
        self.load_ema_weights = load_ema_weights
        self._initialize_model()

    def _initialize_model(self):
        ranked_logger.info(f"Loading checkpoint from {self.ckpt_path.resolve()}...")

        checkpoint = torch.load(self.ckpt_path, "cpu", weights_only=False)
        self.cfg = OmegaConf.create(checkpoint["train_cfg"])

        # Save confidence config before nulling the loss (needed for PAE/pLDDT unbinning)
        self.confidence_config = None
        if (
            self.cfg.get("trainer")
            and self.cfg.trainer.get("loss")
            and self.cfg.trainer.loss.get("confidence_loss")
        ):
            self.confidence_config = OmegaConf.to_container(
                self.cfg.trainer.loss.confidence_loss, resolve=True
            )
            self.confidence_config = OmegaConf.create(self.confidence_config)

        # Apply inference overrides (aligned with BaseInferenceEngine)
        self.cfg.model.net.inference_sampler.num_timesteps = self.num_steps
        self.cfg.model.net.inference_sampler.solver = "af3"
        self.cfg.trainer.num_nodes = 1
        self.cfg.trainer.devices_per_node = 1
        self.cfg.trainer.loss = (
            None  # skip loss instantiation (not needed for inference)
        )
        self.cfg.trainer.metrics = {}  # skip metrics instantiation

        set_accelerator_based_on_availability(self.cfg)

        # Construct trainer (follows BaseInferenceEngine._construct_trainer pattern)
        ranked_logger.info("Instantiating trainer...")
        self.trainer = hydra.utils.instantiate(
            self.cfg.trainer,
            _convert_="partial",
            _recursive_=False,
        )

        self.trainer.fabric.launch()
        self.trainer.initialize_or_update_trainer_state({"train_cfg": self.cfg})
        self.trainer.construct_model()

        ranked_logger.info("Loading model weights from checkpoint...")
        self.trainer.load_checkpoint(checkpoint=checkpoint)

        self.trainer.state["optimizer"] = None
        self.trainer.state["train_cfg"].model.optimizer = None
        self.trainer.setup_model_optimizers_and_schedulers()
        self.trainer.state["model"].eval()

        # Get model references
        self.model = self.trainer.state["model"]
        self.trunk_model = self._unwrap_model(self.model)
        self.has_confidence = hasattr(self.trunk_model, "confidence_head")

        # Get config references
        self.distogram_config = self.cfg.model.net.distogram_head

        self._initialize_pipeline()

    def _initialize_pipeline(self):
        """Build inference features without inheriting training-only assumptions."""
        first_val_dataset_key, first_val_dataset = next(
            iter(self.cfg.datasets.val.items())
        )
        ranked_logger.info(
            f"Using settings from validation dataset: {first_val_dataset_key}"
        )

        transform_overrides = {
            "diffusion_batch_size": self.diffusion_batch_size,
            "n_recycles": self.n_recycles,
            "is_inference": True,
            "run_confidence_head": True,
            # Sequence-built atoms have random placeholder coordinates. They
            # cannot establish physical terminal bonds in a linear protein.
            "add_cyclic_bonds": False,
            # Public checkpoints may embed a training site's private cache.
            # Override it explicitly, including values serialized in train_cfg.
            "residue_cache_dir": None,
        }
        transform_cfg = OmegaConf.merge(
            first_val_dataset.dataset.transform,
            OmegaConf.create(transform_overrides),
        )
        self.pipeline = hydra.utils.instantiate(transform_cfg)

    @staticmethod
    def _unwrap_model(module):
        """Unwrap to get core model."""
        while hasattr(module, "module") or hasattr(module, "model"):
            if hasattr(module, "module"):
                module = module.module
            elif hasattr(module, "model"):
                module = module.model
        return module

    def forward(
        self,
        trunk_input: Dict,
        confidence_input: Dict,
        pipeline_output: Dict,
        n_cycle: int = 1,
        compute_gradient: bool = False,
        skip_diffusion: bool = False,
    ) -> Dict:
        """
        Direct forward pass through the model.

        This is for optimization - gives access to raw model outputs with gradient support.
        """
        # set model mode
        # if compute_gradient:
        #     self.model.train()
        # else:
        #     self.model.eval()

        # coord_atom_lvl_to_be_noised = pipeline_output["coord_atom_lvl_to_be_noised"].to(self.device)

        coord_atom_lvl_to_be_noised = pipeline_output["coord_atom_lvl_to_be_noised"]

        input_with_confidence = {
            **trunk_input,
            "seq": confidence_input["seq"],
            "frame_atom_idxs": confidence_input["frame_atom_idxs"],
            "rep_atom_idxs": confidence_input["rep_atom_idxs"],
        }
        input_with_confidence = self.trainer.fabric.to_device(input_with_confidence)
        coord_atom_lvl_to_be_noised = self.trainer.fabric.to_device(
            coord_atom_lvl_to_be_noised
        )
        # context manager for gradient computation
        context = torch.enable_grad() if compute_gradient else torch.no_grad()

        # with context:
        with context, self.trainer.fabric.autocast():
            out = optimization_forward(
                self.trunk_model,
                input_with_confidence,
                n_cycle=n_cycle,
                coordinates=coord_atom_lvl_to_be_noised,
                skip_diffusion=skip_diffusion,
            )

            # Add unbinned distogram for distance constraints
            if "distogram" in out:
                out["distogram_unbinned"] = self._unbin_distogram(
                    out["distogram"], min_distance=2.0, max_distance=22.0
                )

        return out

    def compute_loss(
        self,
        loss_type: Union[str, Dict],
        trunk_input: Dict,
        trunk_output: Dict,
        confidence_input: Dict,
        pipeline_output: Dict,
    ) -> torch.Tensor:
        # Distance constraints (expect loss_type to be dict with constraint info)
        if (
            isinstance(loss_type, dict)
            and loss_type.get("type") == "distance_constraint"
        ):
            constraints = loss_type.get("constraints", [])
            weight = loss_type.get("weight", 1.0)

            return DistanceConstraint(
                _name="distance_constraint", constraints=constraints, weight=weight
            ).forward(
                {}, trunk_input, trunk_output, confidence_input, {}, pipeline_output
            )[0]

        # Contact losses
        elif loss_type == "inter_contact":
            return ContactLoss(
                _name="inter_contact", contact_type="inter", cutoff=22.0, k=1
            ).forward(
                {}, trunk_input, trunk_output, confidence_input, {}, pipeline_output
            )[0]

        elif loss_type == "intra_contact":
            return ContactLoss(
                _name="intra_contact",
                contact_type="intra",
                cutoff=14.0,
                k=2,
                min_seq_sep=9,
            ).forward(
                {}, trunk_input, trunk_output, confidence_input, {}, pipeline_output
            )[0]

        elif loss_type == "sum_contact_loss":
            inter_loss = self.compute_loss(
                "inter_contact",
                trunk_input,
                trunk_output,
                confidence_input,
                pipeline_output,
            )
            intra_loss = self.compute_loss(
                "intra_contact",
                trunk_input,
                trunk_output,
                confidence_input,
                pipeline_output,
            )
            return inter_loss + intra_loss

        # Interface entropy
        elif loss_type == "interface_entropy":
            return DistogramInterfaceEntropyLoss(
                _name="interface_entropy", weight=1.0
            ).forward(
                {}, trunk_input, trunk_output, confidence_input, {}, pipeline_output
            )[0]

        # Weighted interface entropy
        elif loss_type == "weighted_interface_entropy":
            return WeightedInterfaceEntropyLoss(
                _name="weighted_interface_entropy", weight=1.0
            ).forward(
                {}, trunk_input, trunk_output, confidence_input, {}, pipeline_output
            )[0]

        # Distogram CCE
        elif loss_type == "distogram_cce":
            X_rep_atoms_I = pipeline_output["ground_truth"]["coord_token_lvl"].to(
                self.device
            )
            crd_mask_rep_atoms_I = pipeline_output["ground_truth"]["mask_token_lvl"].to(
                self.device
            )
            pred_num_bins = self.distogram_config.bins - 1

            return DistogramCCELoss.distogram_cce_loss(
                pred_distogram=trunk_output["distogram"],
                X_rep_atoms_I=X_rep_atoms_I,
                crd_mask_rep_atoms_I=crd_mask_rep_atoms_I,
                bins=pred_num_bins,
            )

        # Confidence-based losses (confidence as losses)
        elif loss_type in [
            "pae_interface_mean",
            "pae_interface_min",
            "pae_mean",
            "pde_mean",
            "plddt_mean",
            "iptm",
        ]:
            # Only compute the specific metric requested (not all of them!)
            metrics = self._process_confidence_loss(
                trunk_output, pipeline_output, metric_to_compute=loss_type
            )

            if loss_type == "pae_interface_mean":
                return metrics.get(
                    "pae_interface_mean", torch.tensor(float("inf"), device=self.device)
                )
            elif loss_type == "pae_interface_min":
                return metrics.get(
                    "pae_interface_min", torch.tensor(float("inf"), device=self.device)
                )
            elif loss_type == "pae_mean":
                return metrics.get(
                    "pae_mean", torch.tensor(float("inf"), device=self.device)
                )
            elif loss_type == "pde_mean":
                return metrics.get(
                    "pde_mean", torch.tensor(float("inf"), device=self.device)
                )
            elif loss_type == "plddt_mean":
                # Negative because we want to maximize pLDDT
                return -metrics.get("plddt_mean", torch.tensor(0.0, device=self.device))
            elif loss_type == "iptm":
                # Negative because we want to maximize iPTM (higher iPTM is better)
                return -metrics.get("iptm", torch.tensor(0.0, device=self.device))

        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def _process_confidence_loss(
        self,
        trunk_output: Dict,
        pipeline_output: Dict,
        metric_to_compute: str = None,  # Only compute specific metric if provided
    ) -> Dict:
        """Process confidence outputs for use as losses. Only computes requested metric for efficiency."""
        if not self.has_confidence or self.confidence_config is None:
            return {}

        # Get logits
        pae_logits = trunk_output.get("pae")
        pde_logits = trunk_output.get("pde")
        plddt_logits = trunk_output.get("plddt")

        if plddt_logits is None:
            return {}

        metrics = {}

        # Get metadata
        ch_label = pipeline_output["ground_truth"]["chain_iid_token_lvl"]
        is_real_atom = pipeline_output["confidence_feats"]["is_real_atom"]

        # Only compute pLDDT if needed
        if metric_to_compute is None or metric_to_compute in ["plddt_mean"]:
            # Reshape pLDDT if needed
            if len(plddt_logits.shape) == 3:
                plddt_logits = plddt_logits.reshape(
                    plddt_logits.shape[0],
                    plddt_logits.shape[1],
                    NHEAVY,
                    self.confidence_config.plddt.n_bins,
                )

            plddt = unbin_logits(
                plddt_logits.permute(0, 3, 1, 2).to(torch.float),
                self.confidence_config.plddt.max_value,
                self.confidence_config.plddt.n_bins,
            )

            # Get per-residue pLDDT
            plddt_masked = plddt * is_real_atom[None, ..., :NHEAVY]
            plddt_per_residue = plddt_masked.sum(dim=-1) / is_real_atom.sum(
                dim=-1
            ).clamp(min=1)

            metrics["plddt_per_residue"] = plddt_per_residue
            metrics["plddt_mean"] = plddt_per_residue.mean()

        # Only compute PAE if needed
        if metric_to_compute is None or metric_to_compute in [
            "pae_interface_mean",
            "pae_interface_min",
            "pae_mean",
            "iptm",
        ]:
            if pae_logits is not None:
                pae = unbin_logits(
                    pae_logits.permute(0, 3, 1, 2).to(torch.float),
                    self.confidence_config.pae.max_value,
                    self.confidence_config.pae.n_bins,
                )

                # Only compute global PAE mean if specifically requested
                if metric_to_compute is None or metric_to_compute == "pae_mean":
                    metrics["pae_mean"] = pae.mean()

                # Only compute interface metrics if requested
                if metric_to_compute is None or metric_to_compute in [
                    "pae_interface_mean",
                    "pae_interface_min",
                    "iptm",
                ]:
                    pairs_to_score = create_interface_masks_2d(
                        ch_label, device=pae.device
                    )
                    if len(pairs_to_score) > 0:
                        interface_mask = next(iter(pairs_to_score.values()))

                        # PAE interface mean
                        if (
                            metric_to_compute is None
                            or metric_to_compute == "pae_interface_mean"
                        ):
                            pae_interface = compute_mean_over_subsampled_pairs(
                                pae, interface_mask
                            )
                            metrics["pae_interface_mean"] = pae_interface.mean()

                        # PAE interface min
                        if (
                            metric_to_compute is None
                            or metric_to_compute == "pae_interface_min"
                        ):
                            masked_pae = torch.where(
                                interface_mask.unsqueeze(0),
                                pae,
                                torch.tensor(float("inf"), device=pae.device),
                            )
                            valid_pae = masked_pae[masked_pae != float("inf")]
                            if len(valid_pae) > 0:
                                metrics["pae_interface_min"] = valid_pae.min()
                            else:
                                metrics["pae_interface_min"] = torch.tensor(
                                    float("inf"), device=pae.device, requires_grad=False
                                )
                    else:
                        # No interface found
                        if (
                            metric_to_compute is None
                            or metric_to_compute == "pae_interface_mean"
                        ):
                            metrics["pae_interface_mean"] = torch.tensor(
                                float("inf"), device=pae.device, requires_grad=False
                            )
                        if (
                            metric_to_compute is None
                            or metric_to_compute == "pae_interface_min"
                        ):
                            metrics["pae_interface_min"] = torch.tensor(
                                float("inf"), device=pae.device, requires_grad=False
                            )

                # Only compute iPTM if requested (note: not differentiable due to .item())
                if metric_to_compute is None or metric_to_compute == "iptm":
                    iptm_score = self._calculate_iptm(pae, ch_label)
                    metrics["iptm"] = torch.tensor(
                        iptm_score, device=pae.device, requires_grad=False
                    )

        # Only compute PDE if needed
        if metric_to_compute is None or metric_to_compute == "pde_mean":
            if pde_logits is not None:
                pde = unbin_logits(
                    pde_logits.permute(0, 3, 1, 2).to(torch.float),
                    self.confidence_config.pde.max_value,
                    self.confidence_config.pde.n_bins,
                )
                metrics["pde_mean"] = pde.mean()

        return metrics

    def _calculate_iptm(self, pae: torch.Tensor, ch_label) -> float:
        """Calculate iPTM (interface Predicted TM-score) from PAE."""
        if pae.dim() == 3 and pae.size(0) == 1:
            pae = pae[0]
        elif pae.dim() != 2:
            raise ValueError("`pae` must be [L, L] or [1, L, L].")

        L = pae.size(0)
        device = pae.device
        dtype = pae.dtype

        # Convert ch_label to tensor if it's numpy array
        if not isinstance(ch_label, torch.Tensor):
            # Handle string chain labels
            import numpy as np

            ch_label_np = np.asarray(ch_label)
            if ch_label_np.dtype.kind in {"U", "S", "O"}:  # String types
                # Convert strings to numeric indices
                unique_chains = np.unique(ch_label_np)
                chain_to_idx = {chain: i for i, chain in enumerate(unique_chains)}
                ch_label_numeric = np.array([chain_to_idx[c] for c in ch_label_np])
                ch_label = torch.tensor(
                    ch_label_numeric, device=device, dtype=torch.long
                )
            else:
                ch_label = torch.tensor(ch_label_np, device=device)
        else:
            ch_label = ch_label.to(device)

        Leff = max(L, 19)
        d0 = 1.24 * ((Leff - 15.0) ** (1.0 / 3.0)) - 1.8
        d0 = torch.tensor(d0, dtype=dtype, device=device)

        tm_like = 1.0 / (1.0 + (pae / d0) ** 2)  # [L, L]

        pair_mask = ch_label[:, None] != ch_label[None, :]  # [L, L] bool
        if pair_mask.sum() == 0:
            return float("nan")  # single-chain case

        pair_mask_f = pair_mask.float()

        norm = pair_mask_f.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        per_alignment = (tm_like * pair_mask_f / norm).sum(dim=-1)  # [L]
        iptm = per_alignment.max()  # scalar tensor

        return iptm.item()

    def optimization_objective(
        self,
        trunk_input: Dict,
        confidence_input: Dict,
        pipeline_output: Dict,
        objective_type: str = "sum_contact_loss",
        n_cycle: int = 1,
        skip_diffusion: bool = True,
    ) -> torch.Tensor:
        """
        Compute optimization objective with gradient enabled.

        This is the main method that optimizers should call.
        """
        # Forward pass with gradients
        output = self.forward(
            trunk_input,
            confidence_input,
            pipeline_output,
            n_cycle=n_cycle,
            compute_gradient=True,
            skip_diffusion=skip_diffusion,
        )

        # Compute the requested loss
        return self.compute_loss(
            objective_type, trunk_input, output, confidence_input, pipeline_output
        )

    """Inference  pipeline as in inference engine (for validation purpose)"""

    def process_input(
        self, input_path: Union[str, Path, AtomArray], example_id: Optional[str] = None
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Process input through the pipeline (following old pipeline pattern).

        Returns:
            Tuple of (network_input, confidence_input, pipeline_output)
        """
        if isinstance(input_path, AtomArray):
            # AtomArray input
            atom_array = input_path
            if example_id is None:
                example_id = "direct_input"

            pipeline_input = {
                "example_id": example_id,
                "atom_array": atom_array,
                "chain_info": {},
            }
            pipeline_output = self.pipeline(pipeline_input)
        else:
            # CIF file input
            network_input, confidence_input, pipeline_output = cif2input(
                input_path, self.pipeline, device=self.device
            )
            return network_input, confidence_input, pipeline_output

        # Assemble network input
        network_input = self._assemble_network_inputs(pipeline_output)

        # Assemble confidence input
        confidence_input = {}
        if "confidence_feats" in pipeline_output:
            if "rf2aa_seq" in pipeline_output["confidence_feats"]:
                confidence_input["seq"] = pipeline_output["confidence_feats"][
                    "rf2aa_seq"
                ]
            if (
                "pae_frame_idx_token_lvl_from_atom_lvl"
                in pipeline_output["confidence_feats"]
            ):
                confidence_input["frame_atom_idxs"] = pipeline_output[
                    "confidence_feats"
                ]["pae_frame_idx_token_lvl_from_atom_lvl"]

        if "ground_truth" in pipeline_output:
            if "rep_atom_idxs" in pipeline_output["ground_truth"]:
                confidence_input["rep_atom_idxs"] = pipeline_output["ground_truth"][
                    "rep_atom_idxs"
                ]

        # Move to device
        network_input = self._to_device(network_input)
        confidence_input = self._to_device(confidence_input)
        pipeline_output = self._to_device(pipeline_output)

        return network_input, confidence_input, pipeline_output

    def _assemble_network_inputs(self, example: dict) -> dict:
        """Assemble network inputs (from old pipeline)."""
        coord_noised = example["coord_atom_lvl_to_be_noised"] + example["noise"]
        coord_noised = torch.nan_to_num(coord_noised)

        network_input = {
            "X_noisy_L": coord_noised,
            "t": example["t"],
            "f": example["feats"],
        }

        # Force-cast some features to bfloat16
        # for x in ["msa_stack", "profile", "template_distogram", "template_restype", "template_unit_vector"]:
        #     if x in network_input["f"] and hasattr(network_input["f"][x], 'to'):
        #         network_input["f"][x] = network_input["f"][x].to(torch.bfloat16)

        return network_input

    def _unbin_distogram(self, distogram_logits, min_distance=2.0, max_distance=22.0):
        """
        Unbin distogram logits to get continuous distance matrix.

        Args:
            distogram_logits: [I, I, bins] binned distance logits
            min_distance: Minimum distance for bins
            max_distance: Maximum distance for bins

        Returns:
            [I, I] unbinned distance matrix
        """
        num_bins = distogram_logits.shape[-1]
        bin_step = (max_distance - min_distance) / (num_bins - 1)
        bin_centers = torch.arange(
            min_distance,
            max_distance + bin_step,
            bin_step,
            device=distogram_logits.device,
        )[:num_bins]

        probabilities = torch.nn.Softmax(dim=-1)(distogram_logits)
        unbinned = (probabilities * bin_centers[None, None, :]).sum(dim=-1)

        return unbinned

    def _to_device(self, data: Dict) -> Dict:
        """Move data to device."""
        import tree

        return tree.map_structure(
            lambda x: x.to(self.device) if hasattr(x, "to") else x, data
        )

    def run_inference(
        self,
        network_input: Dict,
        confidence_input: Dict,
        pipeline_output: Dict,
        n_recycles: Optional[int] = None,
    ) -> Dict:
        """
        Run model inference to get structure and confidence predictions.

        Uses trainer's validation_step.

        Args:
            network_input: Network input dictionary from process_input
            confidence_input: Confidence input dictionary from process_input
            pipeline_output: Full pipeline output (needed for trainer)
            n_recycles: Number of recycles (uses default if None)

        Returns:
            Dictionary containing network outputs including structures and confidence scores
        """
        if n_recycles is None:
            n_recycles = network_input["f"]["msa_stack"].shape[0]

        batch = pipeline_output
        batch["feats"] = network_input["f"]
        batch["t"] = network_input["t"]
        batch["noise"] = (
            network_input["X_noisy_L"] - batch["coord_atom_lvl_to_be_noised"]
        )

        # add confidence inputs to batch
        if "confidence_feats" not in batch:
            batch["confidence_feats"] = {}
        batch["confidence_feats"].update(
            {
                k: v
                for k, v in confidence_input.items()
                if k in ["seq", "frame_atom_idxs"]
            }
        )
        if "rep_atom_idxs" in confidence_input:
            if "ground_truth" not in batch:
                batch["ground_truth"] = {}
            batch["ground_truth"]["rep_atom_idxs"] = confidence_input["rep_atom_idxs"]

        batch = self.trainer.fabric.to_device(batch)

        with torch.no_grad():
            outputs = self.trainer.validation_step(
                batch=batch,
                batch_idx=0,
                compute_metrics=False,
            )

        return outputs["network_output"]

    def get_confidence_metrics(
        self, network_output: Dict, pipeline_output: Dict, example_id: str = "inference"
    ) -> Dict:
        """
        Compile confidence metrics from network outputs.

        Args:
            network_output: Output from run_inference
            pipeline_output: Pipeline output from process_input
            example_id: Example identifier

        Returns:
            Dictionary containing compiled confidence metrics
        """
        if "plddt" not in network_output:
            ranked_logger.warning("No confidence outputs in network output")
            return {}

        if self.confidence_config is None:
            ranked_logger.warning(
                "No confidence config available, cannot compile confidence outputs"
            )
            return {}

        confidence_outputs = compile_af3_confidence_outputs(
            plddt_logits=network_output["plddt"],
            pae_logits=network_output.get("pae"),
            pde_logits=network_output.get("pde"),
            chain_iid_token_lvl=pipeline_output["ground_truth"]["chain_iid_token_lvl"],
            is_real_atom=pipeline_output["confidence_feats"]["is_real_atom"],
            example_id=example_id,
            confidence_loss_cfg=self.confidence_config,
        )

        return confidence_outputs

    def get_structures(
        self, network_output: Dict, pipeline_output: Dict
    ) -> torch.Tensor:
        """
        Extract predicted structures from network output.

        Args:
            network_output: Output from run_inference
            pipeline_output: Pipeline output from process_input

        Returns:
            Predicted atomic coordinates tensor [B, L, 3]
        """
        if "X_pred_rollout_L" in network_output:
            return network_output["X_pred_rollout_L"]
        elif "X_L" in network_output:
            return network_output["X_L"]
        else:
            raise KeyError("No structure predictions found in network output")

    def infer_from_cif(
        self,
        cif_path: Union[str, Path],
    ) -> Dict:
        """
        Complete inference pipeline from CIF file input.

        Args:
            cif_path: Path to input CIF file

        Returns:
            Dictionary containing:
                - 'coordinates': Predicted atomic coordinates [B, L, 3]
                - 'confidence': Confidence metrics dictionary
                - 'plddt': Per-residue pLDDT scores
                - 'pae': Predicted aligned error matrix
                - 'pde': Predicted distance error matrix
                - 'confidence_df': DataFrame with aggregated metrics
                - 'atom_array_stack': AtomArrayStack with predicted structures
                - 'S_I': Token-level single representation [I, C_s]
                - 'Z_II': Token-level pair representation [I, I, C_z]
        """
        # input processing
        network_input, confidence_input, pipeline_output = self.process_input(cif_path)

        # run inference
        network_output = self.run_inference(
            network_input, confidence_input, pipeline_output
        )

        # get structures
        coordinates = self.get_structures(network_output, pipeline_output)

        # get confidence metrics
        confidence_metrics = self.get_confidence_metrics(
            network_output,
            pipeline_output,
            example_id=Path(cif_path).stem
            if isinstance(cif_path, (str, Path))
            else "inference",
        )

        # build atom array stack
        atom_array_stack = build_stack_from_atom_array_and_batched_coords(
            coordinates, pipeline_output["atom_array"]
        )

        output = {
            "coordinates": coordinates,
            "confidence": confidence_metrics,
            "plddt": confidence_metrics.get("plddt"),
            "pae": confidence_metrics.get("pae"),
            "pde": confidence_metrics.get("pde"),
            "confidence_df": confidence_metrics.get("confidence_df"),
            "atom_array_stack": atom_array_stack,
            "S_I": network_output.get("S_I"),
            "Z_II": network_output.get("Z_II"),
        }

        return output
