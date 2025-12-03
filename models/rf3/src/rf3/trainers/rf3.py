import hydra
import torch
from beartype.typing import Any
from einops import repeat
from jaxtyping import Float, Int
from lightning_utilities import apply_to_collection
from omegaconf import DictConfig
from rf3.loss.af3_losses import (
    ResidueSymmetryResolution,
    SubunitSymmetryResolution,
)
from rf3.model.RF3 import ShouldEarlyStopFn
from rf3.utils.io import build_stack_from_atom_array_and_batched_coords
from rf3.utils.recycling import get_recycle_schedule

from foundry.common import exists
from foundry.metrics.losses import Loss
from foundry.metrics.metric import MetricManager
from foundry.trainers.fabric import FabricTrainer
from foundry.training.EMA import EMA
from foundry.utils.ddp import RankedLogger
from foundry.utils.torch import assert_no_nans, assert_same_shape

ranked_logger = RankedLogger(__name__, rank_zero_only=True)


def _remap_outputs(
    xyz: Float[torch.Tensor, "D L 3"], mapping: Int[torch.Tensor, "D L"]
) -> Float[torch.Tensor, "D L 3"]:
    """Helper function to remap outputs using a mapping tensor."""
    for i in range(xyz.shape[0]):
        xyz[i, mapping[i]] = xyz[i].clone()
    return xyz


class RF3Trainer(FabricTrainer):
    """Standard Trainer for AF3-style models"""

    def __init__(
        self,
        *,
        n_recycles_train: int | None = None,
        loss: DictConfig | dict | None = None,
        metrics: DictConfig | dict | MetricManager | None = None,
        seed=None,  # dumped
        **kwargs,
    ):
        """See `FabricTrainer` for the additional initialization arguments.

        Args:
            n_recycles_train: Maximum number of recycles (per-batch), for models that support recycling. During training, the model will be recycled a
                random number of times between 1 and `n_recycles_train`. During inference, we determine the number of recycles from the MSA stack shape. However,
                for training, we must sample the number of recycles upfront, so all GPUs within a distributed batch can sample the same number of recycles.
            loss: Configuration for the loss function. If None, the loss function will not be instantiated.
            metrics: Metrics configuration. Can be:
                - DictConfig/dict with Hydra configs (instantiated internally)
                - Pre-instantiated MetricManager
                - None (no metrics)
        """
        super().__init__(**kwargs)

        # (Initialize recycle schedule upfront so all GPU's can sample the same number of recycles within a batch)
        self.n_recycles_train = n_recycles_train
        self.recycle_schedule = get_recycle_schedule(
            max_cycle=n_recycles_train,
            n_epochs=self.max_epochs,  # Set by FabricTrainer
            n_train=self.n_examples_per_epoch,  # Set by FabricTrainer
            world_size=self.fabric.world_size,
        )  # [n_epochs, n_examples_per_epoch // world_size]

        # Metrics
        if isinstance(metrics, MetricManager):
            # Already instantiated
            self.metrics = metrics
        elif metrics is not None:
            # Hydra config - instantiate
            self.metrics = MetricManager.instantiate_from_hydra(metrics_cfg=metrics)
        else:
            # No metrics
            self.metrics = None

        # Loss
        self.loss = Loss(**loss) if loss else None

        # (Symmetry resolution)
        self.subunit_symm_resolve = SubunitSymmetryResolution()
        self.residue_symm_resolve = ResidueSymmetryResolution()

    def construct_model(self):
        """Construct the model and optionally wrap with EMA."""
        # ... instantiate model with Hydra and Fabric
        with self.fabric.init_module():
            ranked_logger.info("Instantiating model...")

            model = hydra.utils.instantiate(
                self.state["train_cfg"].model.net,
                _recursive_=False,
            )

            # Optionally, wrap the model with EMA
            if self.state["train_cfg"].model.ema is not None:
                ranked_logger.info("Wrapping model with EMA...")
                model = EMA(model, **self.state["train_cfg"].model.ema)

        self.initialize_or_update_trainer_state({"model": model})

    def _assemble_network_inputs(self, example: dict) -> dict:
        """Assemble and validate the network inputs."""
        assert_same_shape(example["coord_atom_lvl_to_be_noised"], example["noise"])
        network_input = {
            "X_noisy_L": example["coord_atom_lvl_to_be_noised"] + example["noise"],
            "t": example["t"],
            "f": example["feats"],
        }

        try:
            assert_no_nans(
                network_input["X_noisy_L"],
                msg=f"network_input (X_noisy_L) for example_id: {example['example_id']}",
            )
        except AssertionError as e:
            if self.state["model"].training:
                # In some cases, we may indeed have NaNs in the the noisy coordinates; we can safely replace them with zeros,
                # and begin noising of those coordinates (which will not have their loss computed) from the origin.
                # Such a situation could occur if there was a chain in the crop with no resolved residues (but that contained resolved
                # residues outside the crop); we then would not be able to resolve the missing coordinates to their "closest resolved neighbor"
                # within the same chain.
                network_input["X_noisy_L"] = torch.nan_to_num(
                    network_input["X_noisy_L"]
                )
                ranked_logger.warning(str(e))
            else:
                # During validation, since we do not crop, there should be no NaN's in the coordinates to noise
                # (They were either removed, as is done with fully unresolved chains, or resolved accoring to our pipeline's rules)
                raise e

        assert_no_nans(
            network_input["f"],
            msg=f"NaN detected in `feats` for example_id: {example['example_id']}",
        )

        return network_input

    def _assemble_loss_extra_info(self, example: dict) -> dict:
        """Assembles metadata arguments to the loss function (incremental to the network inputs and outputs)."""
        # ... reshape
        diffusion_batch_size = example["coord_atom_lvl_to_be_noised"].shape[0]
        X_gt_L = repeat(
            example["ground_truth"]["coord_atom_lvl"],
            "l c -> d l c",
            d=diffusion_batch_size,
        )  # [L, 3] -> [D, L, 3] with broadcasting
        crd_mask_L = repeat(
            example["ground_truth"]["mask_atom_lvl"],
            "l -> d l",
            d=diffusion_batch_size,
        )  # [L] -> [D, L] with broadcasting

        loss_extra_info = {
            "X_gt_L": X_gt_L,  # [D, L, 3]
            "crd_mask_L": crd_mask_L,  # [D, L]
        }

        # ... merge with ground_truth key
        loss_extra_info.update(example["ground_truth"])

        return loss_extra_info

    def _assemble_metrics_extra_info(self, example: dict, network_output: dict) -> dict:
        """Prepares the extra info for the metrics"""
        # We need the same information as for the loss...
        metrics_extra_info = self._assemble_loss_extra_info(example)

        # ... and possibly some additional metadata from the example dictionary
        # TODO: Generalize, so we always use the `extra_info` key, rather than unpacking the ground truth as well
        metrics_extra_info.update(
            {
                # TODO: Remove, instead using `extra_info` for all keys
                **{
                    k: example["ground_truth"][k]
                    for k in [
                        "interfaces_to_score",
                        "pn_units_to_score",
                        "chain_iid_token_lvl",
                    ]
                    if k in example["ground_truth"]
                },
                "example_id": example[
                    "example_id"
                ],  # We require the example ID for logging
                # (From the parser)
                **example.get("extra_info", {}),
            }
        )

        # Record metrics_tags for this example
        metrics_extra_info["metrics_tags"] = example.get("metrics_tags", set())

        # (Create a shallow copy to avoid modifying the original dictionary)
        return {**metrics_extra_info}

    def training_step(
        self,
        batch: Any,
        batch_idx: int,
        is_accumulating: bool,
    ) -> None:
        """Training step, running forward and backward passes.

        Args:
            batch: The current batch; can be of any form.
            batch_idx: The index of the current batch.
            is_accumulating: Whether we are accumulating gradients (i.e., not yet calling optimizer.step()).
                If this is the case, we should skip the synchronization during the backward pass.

        Returns:
            None; we call `loss.backward()` directly, and store the outputs in `self._current_train_return`.
        """
        model = self.state["model"]
        assert model.training, "Model must be training!"

        # Recycling
        # (Number of recycles for the current batch; shared across all GPUs within a distributed batch)
        n_cycle = self.recycle_schedule[self.state["current_epoch"], batch_idx].item()

        with self.fabric.no_backward_sync(model, enabled=is_accumulating):
            # (We assume batch size of 1 for structure predictions)
            example = batch[0] if not isinstance(batch, dict) else batch

            network_input = self._assemble_network_inputs(example)

            # Forward pass (without rollout)
            network_output = model.forward(input=network_input, n_cycle=n_cycle)
            assert_no_nans(
                network_output,
                msg=f"network_output for example_id: {example['example_id']}",
            )

            loss_extra_info = self._assemble_loss_extra_info(example)

            total_loss, loss_dict_batched = self.loss(
                network_input=network_input,
                network_output=network_output,
                # TODO: Rename `loss_input` to `extra_info` to pattern-match metrics
                loss_input=loss_extra_info,
            )

            # Backward pass
            self.fabric.backward(total_loss)

            # ... store the outputs without gradients for use in logging, callbacks, learning rate schedulers, etc.
            self._current_train_return = apply_to_collection(
                {"total_loss": total_loss, "loss_dict": loss_dict_batched},
                dtype=torch.Tensor,
                function=lambda x: x.detach(),
            )

    def validation_step(
        self,
        batch: Any,
        batch_idx: int,
        compute_metrics: bool = True,
    ) -> dict:
        """Validation step, running forward pass and computing validation metrics.

        Args:
            batch: The current batch; can be of any form.
            batch_idx: The index of the current batch.
            compute_metrics: Whether to compute metrics. If False, we will not compute metrics, and the output will be None.
                Set to False during the inference pipeline, where we need the network output but cannot compute metrics (since we
                do not have the ground truth).

        Returns:
            dict: Output dictionary containing the validation metrics and network output.
        """
        model = self.state["model"]
        assert not model.training, "Model must be in evaluation mode during validation!"

        example = batch[0] if not isinstance(batch, dict) else batch

        network_input = self._assemble_network_inputs(example)

        assert_no_nans(
            network_input,
            msg=f"network_input for example_id: {example['example_id']}",
        )

        # ... forward pass (with rollout)
        # (Note that forward() passes to the EMA/shadow model if the model is not training)
        network_output = model.forward(
            input=network_input,
            n_cycle=example["feats"]["msa_stack"].shape[
                0
            ],  # Determine the number of recycles from the MSA stack shape
            coord_atom_lvl_to_be_noised=example["coord_atom_lvl_to_be_noised"],
        )

        assert_no_nans(
            network_output,
            msg=f"network_output for example_id: {example['example_id']}",
        )

        metrics_output = {}
        if compute_metrics and exists(self.metrics):
            metrics_extra_info = self._assemble_metrics_extra_info(
                example, network_output
            )

            # Symmetry resolution
            # TODO: Refactor such that symmetry returns the ideal coordinate permutation, we apply permutation, and pass adjusted prediction to metrics
            # (without needing to use `extra_info` as we are now)
            # TODO: Update symmetry resolution to be functional (vs. using class variable), take explicit inputs (vs. all from netowork_ouput), and use extra_info for the keys it needs
            metrics_extra_info = self.subunit_symm_resolve(
                network_output,
                metrics_extra_info,
                example["symmetry_resolution"],
            )

            metrics_extra_info = self.residue_symm_resolve(
                network_output,
                metrics_extra_info,
                example["automorphisms"],
            )

            metrics_output = self.metrics(
                network_input=network_input,
                network_output=network_output,
                extra_info=metrics_extra_info,
                # (Uses the permuted ground truth after symmetry resolution)
                ground_truth_atom_array_stack=build_stack_from_atom_array_and_batched_coords(
                    metrics_extra_info["X_gt_L"], example.get("atom_array", None)
                ),
                predicted_atom_array_stack=build_stack_from_atom_array_and_batched_coords(
                    network_output["X_L"], example.get("atom_array", None)
                ),
            )

            # Avoid gradients in stored values to prevent memory leaks
            if metrics_output is not None:
                metrics_output = apply_to_collection(
                    metrics_output, torch.Tensor, lambda x: x.detach()
                )

        network_output = apply_to_collection(
            network_output, torch.Tensor, lambda x: x.detach()
        )

        return {"metrics_output": metrics_output, "network_output": network_output}


class RF3TrainerWithConfidence(RF3Trainer):
    """AF-3 trainer with rollout and confidence prediction"""

    def construct_model(self):
        super().construct_model()

        # Freeze gradients for all modules except the confidence head
        for name, param in self.state["model"].named_parameters():
            if "model.confidence_head" not in name:
                param.requires_grad = False

    def _assemble_network_inputs(self, example):
        # assemble the base network inputs...
        network_input = super()._assemble_network_inputs(example)
        #  ... and then add the confidence-specific inputs
        network_input.update(
            {
                "seq": example["confidence_feats"]["rf2aa_seq"],
                "rep_atom_idxs": example["ground_truth"]["rep_atom_idxs"],
                "frame_atom_idxs": example["confidence_feats"][
                    "pae_frame_idx_token_lvl_from_atom_lvl"
                ],
            }
        )

        return network_input

    def _assemble_loss_extra_info(self, example):
        # assemble the base loss extra info...
        loss_extra_info = super()._assemble_loss_extra_info(example)
        # ... and then add the confidence-specific inputs
        loss_extra_info.update(
            {
                # TODO: We are duplicating network_input here; we should be able to significantly trim this dictionary
                "seq": example["confidence_feats"]["rf2aa_seq"],
                "atom_frames": example["confidence_feats"]["atom_frames"],
                "tok_idx": example["feats"]["atom_to_token_map"],
                "is_real_atom": example["confidence_feats"]["is_real_atom"],
                "rep_atom_idxs": example["ground_truth"]["rep_atom_idxs"],
                "frame_atom_idxs": example["confidence_feats"][
                    "pae_frame_idx_token_lvl_from_atom_lvl"
                ],
            }
        )

        return loss_extra_info

    def _assemble_metrics_extra_info(self, example, network_output):
        # assemble the base metrics extra info...
        metrics_extra_info = super()._assemble_metrics_extra_info(
            example, network_output
        )
        # ... and then add the confidence-specific inputs
        # TODO: Refactor; we should not need pass confidence log config through metrics extra info, it should be a property of the Metric (e.g., passed at `_init_` using Hydra interpolation from the relevant loss config)
        metrics_extra_info.update(
            {
                "is_real_atom": example["confidence_feats"]["is_real_atom"],
                "is_ligand": example["feats"]["is_ligand"],
                # TODO: Refactor so that we pass the relevant values from the config direclty to the Metric upon instantiation (reference in Hydra through interpolation)
                "confidence_loss": self.state["train_cfg"].trainer.loss.confidence_loss,
            }
        )

        return metrics_extra_info

    def training_step(
        self,
        batch: Any,
        batch_idx: int,
        is_accumulating: bool,
    ) -> None:
        """Perform mini-rollout and assess gradient of the confidence head parameters with respect to the confidence loss."""
        model = self.state["model"]
        assert model.training, "Model must be training!"

        # Recycling
        # (Number of recycles for the current batch; shared across all GPUs within a distributed batch)
        n_cycle = self.recycle_schedule[self.state["current_epoch"], batch_idx].item()

        with self.fabric.no_backward_sync(model, enabled=is_accumulating):
            # (We assume batch size of 1 for structure predictions)
            example = batch[0] if not isinstance(batch, dict) else batch

            network_input = self._assemble_network_inputs(example)

            # Forward pass (with mini-rollout)
            # NOTE: We use the non-EMA weights for structure prediction; this approach is theoretically sub-optimal, since
            # we should be using the EMA weights for structure prediction (given those parameters are frozen) and the non-EMA weights
            # for the confidence head, to better match the inference-time task
            network_output = model.forward(
                input=network_input,
                n_cycle=n_cycle,
                coord_atom_lvl_to_be_noised=example["coord_atom_lvl_to_be_noised"],
            )
            assert_no_nans(
                network_output,
                msg=f"network_output for example_id: {example['example_id']}",
            )

            loss_extra_info = self._assemble_loss_extra_info(example)

            # Remap X_L to the rollout X_L so ground truth matches rollout batch dimension during the symmetry resolution
            # NOTE: Since `X_L` derives from the mini-rollout, we cannot compute standard training loss and perform gradient updates
            network_output["X_L"] = network_output["X_pred_rollout_L"]

            # (Symmetry resolution)
            loss_extra_info = self.subunit_symm_resolve(
                network_output, loss_extra_info, example["symmetry_resolution"]
            )
            loss_extra_info = self.residue_symm_resolve(
                network_output, loss_extra_info, example["automorphisms"]
            )

            # We only assess the confidence loss
            total_loss, loss_dict_batched = self.loss(
                network_input=network_input,
                network_output=network_output,
                # TODO: Rename `loss_input` to `extra_info` to pattern-match metrics
                loss_input=loss_extra_info,
            )

            # Backward pass
            self.fabric.backward(total_loss)

            # ... store the outputs without gradients for use in logging, callbacks, learning rate schedulers, etc.
            self._current_train_return = apply_to_collection(
                {"total_loss": total_loss, "loss_dict": loss_dict_batched},
                dtype=torch.Tensor,
                function=lambda x: x.detach(),
            )

    def validation_step(
        self,
        batch: Any,
        batch_idx: int,
        compute_metrics: bool = True,
        should_early_stop_fn: ShouldEarlyStopFn | None = None,
    ) -> dict:
        """Validation step, running forward pass (with full rollout) and computing validation metrics, including confidence."""
        model = self.state["model"]
        assert not model.training, "Model must be in evaluation mode during validation!"

        example = batch[0] if not isinstance(batch, dict) else batch

        network_input = self._assemble_network_inputs(example)

        assert_no_nans(
            network_input,
            msg=f"network_input for example_id: {example['example_id']}",
        )

        # ... forward pass (with FULL rollout)
        # (Note that forward() passes to the EMA/shadow model if the model is not training)
        network_output = model.forward(
            input=network_input,
            n_cycle=example["feats"]["msa_stack"].shape[
                0
            ],  # Determine the number of recycles from the MSA stack shape
            coord_atom_lvl_to_be_noised=example["coord_atom_lvl_to_be_noised"],
            should_early_stop_fn=should_early_stop_fn,
        )

        assert_no_nans(
            network_output,
            msg=f"network_output for example_id: {example['example_id']}",
        )

        # Remap X_L to the rollout X_L
        network_output["X_L"] = network_output.get("X_pred_rollout_L")

        metrics_output = {}
        if (
            compute_metrics
            and exists(self.metrics)
            and not network_output.get("early_stopped", False)
        ):
            # Assemble the base metrics extra info and add confidence-specific inputs
            metrics_extra_info = self._assemble_metrics_extra_info(
                example, network_output
            )

            # Symmetry resolution
            metrics_extra_info = self.subunit_symm_resolve(
                network_output,
                metrics_extra_info,
                example["symmetry_resolution"],
            )

            metrics_extra_info = self.residue_symm_resolve(
                network_output,
                metrics_extra_info,
                example["automorphisms"],
            )

            metrics_output = self.metrics(
                network_input=network_input,
                network_output=network_output,
                extra_info=metrics_extra_info,
                # (Uses the permuted ground truth after symmetry resolution)
                ground_truth_atom_array_stack=build_stack_from_atom_array_and_batched_coords(
                    metrics_extra_info["X_gt_L"], example.get("atom_array", None)
                ),
                predicted_atom_array_stack=build_stack_from_atom_array_and_batched_coords(
                    network_output["X_L"], example.get("atom_array", None)
                ),
            )

            # Avoid gradients in stored values to prevent memory leaks
            if metrics_output is not None:
                metrics_output = apply_to_collection(
                    metrics_output, torch.Tensor, lambda x: x.detach()
                )

        network_output = (
            apply_to_collection(network_output, torch.Tensor, lambda x: x.detach())
            if network_output is not None
            else None
        )

        return {"metrics_output": metrics_output, "network_output": network_output}
