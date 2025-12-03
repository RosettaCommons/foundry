import numpy as np
import torch
from beartype.typing import Any, List, Union
from biotite.structure import AtomArray, AtomArrayStack
from biotite.structure.residues import get_residue_starts
from einops import repeat
from lightning_utilities import apply_to_collection
from omegaconf import DictConfig
from rfd3.metrics.design_metrics import get_all_backbone_metrics
from rfd3.metrics.hbonds_hbplus_metrics import get_hbond_metrics
from rfd3.trainer.recycling import get_recycle_schedule
from rfd3.trainer.trainer_utils import (
    _build_atom_array_stack,
    _cleanup_virtual_atoms_and_assign_atom_name_elements,
    _reassign_unindexed_token_chains,
    _reorder_dict,
    process_unindexed_outputs,
)
from rfd3.utils.io import (
    build_stack_from_atom_array_and_batched_coords,
)

from foundry.metrics.losses import Loss
from foundry.metrics.metric import MetricManager
from foundry.trainers.fabric import FabricTrainer
from foundry.utils.ddp import RankedLogger
from foundry.utils.torch import assert_no_nans, assert_same_shape

global_logger = RankedLogger(__name__, rank_zero_only=False)


class AADesignTrainer(FabricTrainer):
    """Mostly for unique things like saving outputs and parsing inputs

    Args:
        allow_sequence_outputs (bool): Whether to allow sequence outputs in the model.
        convert_non_protein_designed_res_to_ala (bool): Convert non-protein designed residues to ALA. Useful if the
            sequence head spuriously predicts NA residues (when it's performing very poorly).
        cleanup_inference_outputs (bool): Not implemented yet.
        load_sequence_head_weights_if_present (bool): Whether to load the sequence head weights from the checkpoint.
        association_scheme (str): Association scheme to use for the sequence head. Defaults to "atom14".
        seed (int | None): The random seed used for this design, which will be dumped in the output JSON.
            If None, no value will be dumped.
    """

    def __init__(
        self,
        allow_sequence_outputs,
        cleanup_guideposts,
        cleanup_virtual_atoms,
        read_sequence_from_sequence_head,
        output_full_json,
        association_scheme,
        compute_non_clash_metrics_for_diffused_region_only=False,
        seed=None,  # Deprecated
        n_recycles_train: int | None = None,
        loss: DictConfig | dict | None = None,
        metrics: DictConfig | dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.allow_sequence_outputs = allow_sequence_outputs
        self.cleanup_guideposts = cleanup_guideposts
        self.cleanup_virtual_atoms = cleanup_virtual_atoms
        self.read_sequence_from_sequence_head = read_sequence_from_sequence_head
        self.output_full_json = output_full_json
        self.compute_non_clash_metrics_for_diffused_region_only = (
            compute_non_clash_metrics_for_diffused_region_only
        )
        self.association_scheme = association_scheme
        self.seed = None

        # (Initialize recycle schedule upfront so all GPU's can sample the same number of recycles within a batch)
        self.n_recycles_train = n_recycles_train
        self.recycle_schedule = get_recycle_schedule(
            max_cycle=n_recycles_train,
            n_epochs=self.max_epochs,  # Set by FabricTrainer
            n_train=self.n_examples_per_epoch,  # Set by FabricTrainer
            world_size=self.fabric.world_size,
        )  # [n_epochs, n_examples_per_epoch // world_size]

        # Metrics
        # (We could have instantiated loss and metrics recursively, but we prioritize being explicit)
        self.metrics = (
            MetricManager.instantiate_from_hydra(metrics_cfg=metrics)
            if metrics
            else None
        )
        # Loss (full precision)
        with torch.autocast(device_type=self.fabric.device.type, enabled=False):
            self.loss = Loss(**loss) if loss else None

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
                global_logger.warning(str(e))
            else:
                # During validation, since we do not crop, there should be no NaN's in the coordinates to noise
                # (They were either removed, as is done with fully unresolved chains, or resolved accoring to our pipeline's rules)
                raise e

        assert_no_nans(
            network_input["f"],
            msg=f"NaN detected in `feats` for example_id: {example['example_id']}",
        )

        return network_input

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
            coord_atom_lvl_to_be_noised=example["coord_atom_lvl_to_be_noised"],
        )

        assert_no_nans(
            network_output,
            msg=f"network_output for example_id: {example['example_id']}",
        )

        # ... Convert output to a stack of atom arrays
        predicted_atom_array_stack, prediction_metadata = (
            self._build_predicted_atom_array_stack(network_output, example)
        )

        metrics_output = {}
        if compute_metrics:
            assert self.metrics is not None, "Metrics are not defined!"

            metrics_extra_info = self._assemble_metrics_extra_info(
                example, network_output
            )

            metrics_output = self.metrics(
                network_input=network_input,
                network_output=network_output,
                extra_info=metrics_extra_info,
                # (Uses the permuted ground truth after symmetry resolution)
                ground_truth_atom_array_stack=build_stack_from_atom_array_and_batched_coords(
                    metrics_extra_info["X_gt_L"], example.get("atom_array", None)
                ),
                predicted_atom_array_stack=predicted_atom_array_stack,
                prediction_metadata=prediction_metadata,
            )

            if "X_gt_index_to_X" in metrics_extra_info:
                # Remap outputs to minimize error with ground truth
                # TODO: Remap before computing metrics, so that we can avoid pass `extra_info` to metrics (we instead just pass the remapped prediction)
                mapping = metrics_extra_info["X_gt_index_to_X"]  # [D, L]
                network_output["X_L"] = _remap_outputs(network_output["X_L"], mapping)

            # Avoid gradients in stored values to prevent memory leaks
            if metrics_output is not None:
                metrics_output = apply_to_collection(
                    metrics_output, torch.Tensor, lambda x: x.detach()
                )

        if network_output is not None:
            network_output = apply_to_collection(
                network_output, torch.Tensor, lambda x: x.detach()
            )

        return {
            "metrics_output": metrics_output,
            "network_output": network_output,
            "predicted_atom_array_stack": predicted_atom_array_stack,
            "prediction_metadata": prediction_metadata,
        }

    def _assemble_loss_extra_info(self, example: dict) -> dict:
        """Assembles metadata arguments to the loss function (incremental to the network inputs and outputs)."""

        # ... reshape
        diffusion_batch_size = example["coord_atom_lvl_to_be_noised"].shape[0]
        X_gt_L = repeat(
            example["ground_truth"]["coord_atom_lvl"],
            "l c -> d l c",
            d=diffusion_batch_size,
        )  # [L, 3] -> [D, L, 3] with broadcasting

        return {
            "X_gt_L": X_gt_L,  # [D, L, 3]
            "X_gt_L_in_input_frame": example[
                "coord_atom_lvl_to_be_noised"
            ],  # [D, L, 3] for no-align loss
            "crd_mask_L": example["ground_truth"]["mask_atom_lvl"],  # [D, L]
            "is_original_unindexed_token": example["ground_truth"][
                "is_original_unindexed_token"
            ],  # [I,]
            # Sequence information:
            "seq_token_lvl": example["ground_truth"]["sequence_gt_I"],  # [I, 32]
            "sequence_valid_mask": example["ground_truth"][
                "sequence_valid_mask"
            ],  # [I,]
        }

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

        # (Create a shallow copy to avoid modifying the original dictionary)
        return {**metrics_extra_info}

    def _build_predicted_atom_array_stack(
        self, network_output: dict, example: dict
    ) -> Union[AtomArrayStack, List[AtomArray]]:
        atom_array = example["atom_array"]
        f = example["feats"]

        # ... Cleanup atom array:
        atom_array.bonds = None
        atom_array.res_name[~atom_array.is_motif_atom_with_fixed_seq] = (
            "UNK"  # Ensure non-motif residues set to UNK
        )
        atom_array = _reassign_unindexed_token_chains(atom_array)

        # ... Build output atom array stack
        atom_array_stack = _build_atom_array_stack(
            network_output["X_L"],
            atom_array,
            sequence_logits=network_output.get("sequence_logits_I"),
            sequence_indices=network_output.get("sequence_indices_I"),
            allow_sequence_outputs=self.allow_sequence_outputs,
            read_sequence_from_sequence_head=self.read_sequence_from_sequence_head,
            association_scheme=self.association_scheme,
        )  # NB: Will be either list (when sequences are saved) or stack

        arrays = atom_array_stack
        metadata_dict = {i: {"metrics": {}} for i in range(len(arrays))}

        # Add the seed to the metadata dictionary if provided
        if self.seed is not None:
            for i in range(len(arrays)):
                metadata_dict[i]["seed"] = self.seed

        atom_array_stack = []
        for i, atom_array in enumerate(arrays):
            # ... Create essential outputs for metadata dictionary
            if "example" in example["specification"]:
                metadata_dict[i] |= {"task": example["specification"]["example"]}

            # ... Add original specification to metadata
            if self.output_full_json:
                metadata_dict[i] |= {
                    "specification": example["specification"],
                }
                if (
                    hasattr(self, "inference_sampler_overrides")
                    and self.inference_sampler_overrides
                ):
                    metadata_dict[i] |= {
                        "inference_sampler": self.inference_sampler_overrides
                    }

            if np.any(atom_array.is_motif_atom_unindexed):
                # ... insert unindexed motif to output
                atom_array_processed, metadata = process_unindexed_outputs(
                    atom_array,
                    insert_guideposts=self.cleanup_guideposts,
                )
                global_logger.info(
                    f"Inserted unindexed motif atoms for example {i} with RMSD {metadata['insertion_rmsd']:.3f} A"
                )
                if self.cleanup_guideposts:
                    atom_array = atom_array_processed

                diffused_index_map = metadata.pop("diffused_index_map", None)
                metadata_dict[i]["metrics"] |= metadata
                if diffused_index_map is not None:
                    metadata_dict[i]["diffused_index_map"] = diffused_index_map
            else:
                metadata_dict[i]["diffused_index_map"] = {}

            # Also record where indexed motifs ended up
            residue_start_atoms = atom_array[get_residue_starts(atom_array)]
            indexed_residue_starts_non_ligand = residue_start_atoms[
                ~residue_start_atoms.is_motif_atom_unindexed
                & ~residue_start_atoms.is_ligand
            ]

            # If the src_component starts with an alphabetic character, it's from an external source
            external_src_mask = np.array(
                [
                    (s[0].isalpha() if len(s) > 0 else False)
                    for s in indexed_residue_starts_non_ligand.src_component
                ]
            )
            indexed_residue_starts_from_external_src = (
                indexed_residue_starts_non_ligand[external_src_mask]
            )

            for token in indexed_residue_starts_from_external_src:
                metadata_dict[i]["diffused_index_map"][token.src_component] = (
                    f"{token.chain_id}{token.res_id}"
                )

            # ... Delete virtual atoms and assign atom names and elements
            if self.cleanup_virtual_atoms:
                atom_array = _cleanup_virtual_atoms_and_assign_atom_name_elements(
                    atom_array, association_scheme=self.association_scheme
                )

                # ... When cleaning up virtual atoms, we can also calculate native_array_metricsl
                metadata_dict[i]["metrics"] |= get_all_backbone_metrics(
                    atom_array,
                    compute_non_clash_metrics_for_diffused_region_only=self.compute_non_clash_metrics_for_diffused_region_only,
                )

            if (
                "active_donor" in atom_array.get_annotation_categories()
                or "active_acceptor" in atom_array.get_annotation_categories()
            ):
                metadata_dict[i]["metrics"] |= get_hbond_metrics(atom_array)

            if "partial_t" in f:
                # Try calcualte a CA RMSD to input:
                aa_in = example["atom_array"]
                xyz_ca_input = aa_in.coord[np.isin(aa_in.atom_name, "CA")]
                xyz_ca_output = atom_array.coord[np.isin(atom_array.atom_name, "CA")]

                # Align ca and calculate RMSD:
                if xyz_ca_input.shape == xyz_ca_output.shape:
                    try:
                        from rfd3.utils.alignment import weighted_rigid_align

                        xyz_ca_output_aligned = (
                            weighted_rigid_align(
                                torch.from_numpy(xyz_ca_input)[None],
                                torch.from_numpy(xyz_ca_output)[None],
                            )
                            .squeeze(0)
                            .numpy()
                        )
                        metadata_dict[i]["metrics"] |= {
                            "ca_rmsd_to_input": float(
                                np.sqrt(
                                    np.mean(
                                        np.square(
                                            xyz_ca_input - xyz_ca_output_aligned
                                        ).sum(-1)
                                    )
                                )
                            )
                        }
                    except Exception as e:
                        global_logger.warning(
                            f"Failed to calculate CA RMSD for partial diffusion output: {e}"
                        )

            atom_array_stack.append(atom_array)

        # Reorder metadata dictionaries to ensure 'metrics' and 'specification' are last
        metadata_dict = {k: _reorder_dict(d) for k, d in metadata_dict.items()}
        return atom_array_stack, metadata_dict
