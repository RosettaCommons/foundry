from collections import OrderedDict

import hydra
import numpy as np
import torch
from atomworks.ml.encoding_definitions import AF3SequenceEncoding
from atomworks.ml.utils.token import (
    get_token_starts,
    spread_token_wise,
)
from beartype.typing import Any, List, Union
from biotite.structure import AtomArray, AtomArrayStack, concatenate, infer_elements
from biotite.structure.residues import get_residue_starts
from einops import repeat
from jaxtyping import Float, Int
from lightning_utilities import apply_to_collection
from omegaconf import DictConfig
from rfd3.constants import (
    ATOM14_ATOM_NAMES,
    VIRTUAL_ATOM_ELEMENT_NAME,
    association_schemes,
    association_schemes_stripped,
)
from rfd3.metrics.design_metrics import get_all_backbone_metrics
from rfd3.metrics.hbonds_hbplus_metrics import get_hbond_metrics
from rfd3.trainer.fabric_trainer import FabricTrainer
from rfd3.trainer.recycling import get_recycle_schedule
from rfd3.transforms.conditioning_utils import (
    process_unindexed_outputs,
)
from rfd3.util.io import (
    build_stack_from_atom_array_and_batched_coords,
)

from modelhub.common import exists
from modelhub.metrics.metric import MetricManager
from modelhub.training.EMA import EMA
from modelhub.utils.ddp import RankedLogger
from modelhub.utils.torch import assert_no_nans, assert_same_shape

global_logger = RankedLogger(__name__, rank_zero_only=False)


def _remap_outputs(
    xyz: Float[torch.Tensor, "D L 3"], mapping: Int[torch.Tensor, "D L"]
) -> Float[torch.Tensor, "D L 3"]:
    """Helper function to remap outputs using a mapping tensor."""
    for i in range(xyz.shape[0]):
        xyz[i, mapping[i]] = xyz[i].clone()
    return xyz

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
        self.inference_sampler_overrides = None

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
        # (We could have instantiated loss and metrics recursively, but we prioritize being explicit)
        self.metrics = (
            MetricManager.instantiate_from_hydra(metrics_cfg=metrics)
            if metrics
            else None
        )

        # Loss (full precision)
        with torch.autocast(device_type=self.fabric.device.type, enabled=False):
            self.loss = AF3Loss(**loss) if loss else None

    # FROM RF3
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

    # ~~~FROM RF3
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

    # FROM RF3
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
                        from rfd3.util.alignment import weighted_rigid_align

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


def _reorder_dict(d: dict) -> OrderedDict:
    """
    Reorders keys in the dictionary to ensure 'metrics' and 'specification' are last (in that order if both present).
    """
    ordered = OrderedDict()
    first_keys = ["task", "diffused_index_map"]
    last_keys = ["metrics", "specification", "inference_sampler"]
    # First
    for k in first_keys:
        if k in d:
            ordered[k] = d[k]
    # Middle
    for k in d:
        if k not in last_keys and k not in first_keys:
            ordered[k] = d[k]
    # Last
    for k in last_keys:
        if k in d:
            ordered[k] = d[k]
    return ordered


def _build_atom_array_stack(
    coords,
    src_atom_array,
    sequence_indices,
    sequence_logits,
    allow_sequence_outputs=True,
    read_sequence_from_sequence_head=True,
    association_scheme: str = "atom14",
):
    """
    Wraps around build_atom_array_and_batched_coords to also include additional modifications to atom array
    """
    atom_array_stack = build_stack_from_atom_array_and_batched_coords(
        coords, src_atom_array.copy()
    )

    # ... Spoof empty sequences to alanines
    atom_array_stack.res_name[
        atom_array_stack.is_protein & (atom_array_stack.res_name == "UNK")
    ] = "ALA"

    # ... Add sequence if available
    if allow_sequence_outputs:
        array_list = []
        if read_sequence_from_sequence_head and exists(sequence_logits):
            sequence_encoding = AF3SequenceEncoding()
            for i, (atom_array, seq_indices, seq_logits) in enumerate(
                zip(atom_array_stack, sequence_indices, sequence_logits)
            ):
                # Set residue names
                diffused_mask = ~atom_array.is_motif_atom_with_fixed_seq
                three_letter_sequence = sequence_encoding.decode(
                    seq_indices.cpu().numpy().astype(int)
                )  # [I]

                atom_array.res_name[diffused_mask] = three_letter_sequence[
                    atom_array.token_id
                ][diffused_mask]  # [L]

                # Set bfactor column as entropy of sequence logits
                p = torch.softmax(seq_logits, dim=-1).cpu().numpy()  # shape (L, 32)
                res_entropy = -np.sum(p * np.log(p + 1e-10), axis=-1)  # shape (L,)
                atom_array.b_factor = spread_token_wise(atom_array, res_entropy)
                array_list.append(atom_array.copy())
        else:
            # This automatically deletes virtual atoms and assigns resname, atom name, and elements
            for atom_array in atom_array_stack:
                atom_array = _readout_seq_from_struc(
                    atom_array, association_scheme=association_scheme
                )
                array_list.append(atom_array)

    # Return as list
    atom_array_stack = array_list

    return atom_array_stack


def _reassign_unindexed_token_chains(atom_array):
    if np.any((mask := atom_array.is_motif_atom_unindexed)):
        # HACK: Since res_ids are the same, we should save them with a different chain index.
        atom_array.chain_id[mask] = "X"
        atom_array.res_id[mask] = atom_array.orig_res_id[mask]

        # Parse to separate chains
        starts = get_token_starts(atom_array)
        unindexed_starts = starts[mask[starts]]
        token_breaks = atom_array[
            unindexed_starts
        ].is_motif_atom_unindexed_motif_breakpoint
        token_group_id = np.cumsum(token_breaks, dtype=int)  # Group by motif breaks
        token_chain_id = np.array([f"X{i}" for i in token_group_id])

        chains = atom_array.chain_id[starts]
        chains[mask[starts]] = token_chain_id
        atom_array.chain_id = spread_token_wise(atom_array, chains)
    return atom_array


def _cleanup_virtual_atoms_and_assign_atom_name_elements(
    atom_array, association_scheme: str = "atom14"
):
    ## remove virtual atoms based on predicted residue and assign correct atom name and elements
    ret_mask = []
    atom_names = []
    # This is used to indicate which residue is unidentified, probably due to an invalid structure.
    # This is different from the ref_mask, which is used to delete virtual atoms, but this one is used to assign UNK resname for invalid residues.
    invalid_mask = []

    # ... Iterate through each residue.
    # Here we iterate through res_id instead of token_id to avoid some atomization cases or something else.
    res_ids = atom_array.res_id
    res_start_indices = np.concatenate(
        [[0], np.where(res_ids[1:] != res_ids[:-1])[0] + 1]
    )
    res_end_indices = np.concatenate([res_start_indices[1:], [len(res_ids)]])
    warning_issued = False
    for start, end in zip(res_start_indices, res_end_indices):
        res_array = atom_array[start:end]

        is_seq_known = all(
            np.array(res_array.is_motif_atom_with_fixed_seq, dtype=bool)
        ) or all(np.array(res_array.is_motif_atom_unindexed, dtype=bool))

        # ... If sequence is known for the original atom array, just skip
        if is_seq_known:
            ret_mask += [True] * len(res_array)
            invalid_mask += [False] * len(res_array)
            res_name = res_array[0].res_name
            atom_names += res_array.gt_atom_name.tolist()
            continue

        # ... If sequence is unknown for the original atom array, use the predicted / inferred sequence
        res_name = res_array[0].res_name
        if res_name not in association_schemes[association_scheme]:
            global_logger.warning(
                "Model predicted non-protein sequence for diffused residue. Cannot clean up outputs. Assigning unknown residue token."
            )
            warning_issued = True
            ret_mask += [True] * len(res_array)
            invalid_mask += [True] * len(res_array)
            atom_names += res_array.atom_name.tolist()
            continue

        scheme = association_schemes[association_scheme][res_name]
        ret_mask += [True if item is not None else False for item in scheme]
        atom_names += [item.strip() if item is not None else "VX" for item in scheme]
        invalid_mask += [False] * len(scheme)

    if len(atom_names) != atom_array.array_length():
        global_logger.warning(
            f"{atom_names=}\n{atom_array.atom_name=}\nAtom names length {len(atom_names)} does not match original array length {atom_array.array_length()}."
            "\nCould not cleanup atom array!!!"
        )
        if not warning_issued:
            raise ValueError("Atom names length does not match original array length. ")
        return atom_array
    atom_array.atom_name = atom_names
    atom_array.element = np.where(
        atom_array.element == VIRTUAL_ATOM_ELEMENT_NAME,
        infer_elements(atom_names),
        atom_array.element,
    )
    atom_array.res_name[invalid_mask] = np.array(["UNK"] * sum(invalid_mask))
    return atom_array[ret_mask]


def _readout_seq_from_struc(
    atom_array, central_atom="CB", threshold=0.5, association_scheme: str = "atom14"
):
    cur_atom_array_list = []

    # Iterate through each residue
    res_ids = atom_array.res_id
    res_start_indices = np.concatenate(
        [[0], np.where(res_ids[1:] != res_ids[:-1])[0] + 1]
    )
    res_end_indices = np.concatenate([res_start_indices[1:], [len(res_ids)]])

    for start, end in zip(res_start_indices, res_end_indices):
        # ... Check if the current residue is after padding (seq unknown):
        cur_res_atom_array = atom_array[start:end]
        is_seq_known = all(
            np.array(cur_res_atom_array.is_motif_atom_with_fixed_seq, dtype=bool)
        )

        # Here it assumes that every non-protein part has its sequence shown (not padded)
        if not is_seq_known:
            # For Glycine: it doesn't have CB, so set the virtual atom as CA.
            # The current way to handle this is to check if predicted CA and CB are too close, because in the case of glycine and we pad virtual atoms based on CB, CB's coords are set as CA.
            # There might be a better way to do this.
            CA_coord = cur_res_atom_array.coord[cur_res_atom_array.atom_name == "CA"]
            CB_coord = cur_res_atom_array.coord[cur_res_atom_array.atom_name == "CB"]
            if np.linalg.norm(CA_coord - CB_coord) < threshold:
                cur_central_atom = "CA"
            else:
                cur_central_atom = central_atom

            central_mask = cur_res_atom_array.atom_name == cur_central_atom

            # ... Calculate the distance to the central atom
            central_coord = cur_res_atom_array.coord[central_mask][
                0
            ]  # Should only have one central atom anyway
            dists = np.linalg.norm(cur_res_atom_array.coord - central_coord, axis=-1)

            # ... Select virtual atom by the distance. Shouldn't count the central atom itself.
            is_virtual = (dists < threshold) & ~central_mask

            # ... Throw away virtual atoms
            cur_res_atom_array_wo_virtual = cur_res_atom_array[~is_virtual]
            cur_pred_res_atom_names = (
                cur_res_atom_array_wo_virtual.atom_name
            )  # e.g. [N, CA, C, O, CB, V6, V2]

            # ... Iterate over the possible restypes and find the matched one if there is any
            has_restype_assigned = False
            for restype, atom_names in association_schemes_stripped[
                association_scheme
            ].items():
                atom_names = np.array(atom_names)

                # Shouldn't match these two
                if restype in ["UNK", "MSK"]:
                    continue

                # ... Find the index of virtual atom names in the standard atom14 names
                atom_name_idx_in_atom14_scheme = np.array(
                    [
                        np.where(ATOM14_ATOM_NAMES == atom_name)[0][0]
                        for atom_name in cur_pred_res_atom_names
                    ]
                )  # five backbone atoms + some virtual atoms, returning e.g. [0, 1, 2, 3, 4, 11, 7]
                atom14_scheme_mask = np.zeros_like(ATOM14_ATOM_NAMES, dtype=bool)
                atom14_scheme_mask[atom_name_idx_in_atom14_scheme] = True

                # ... Find the matched restype by checking if all the non-None posititons and None positions match
                # This is designed to keep virtual atoms and doesn't assign the atom names for now, which will be handled later.
                if all(x is not None for x in atom_names[atom14_scheme_mask]) and all(
                    x is None for x in atom_names[~atom14_scheme_mask]
                ):
                    cur_res_atom_array.res_name = np.array(
                        [restype] * len(cur_res_atom_array)
                    )
                    cur_atom_array_list.append(cur_res_atom_array)
                    has_restype_assigned = True
                    break
        else:
            cur_atom_array_list.append(cur_res_atom_array)
            has_restype_assigned = True

        # ... Give UNK as the residue name if the mapping fails (unrealistic sidechain)
        if not has_restype_assigned:
            cur_res_atom_array.res_name = np.array(["UNK"] * len(cur_res_atom_array))
            cur_atom_array_list.append(cur_res_atom_array)

    cur_atom_array = concatenate(cur_atom_array_list)

    return cur_atom_array
