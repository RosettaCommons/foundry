from atomworks.ml.transforms.base import ConvertToTorch
from mpnn.collate.feature_collator import FeatureCollator
from mpnn.transforms.feature_aggregation.polymer_ligand_interface import (
    FeaturizePolymerLigandInterfaceMask,
)
from mpnn.transforms.polymer_ligand_interface import ComputePolymerLigandInterface

from foundry.metrics.metric import Metric


class SequenceRecovery(Metric):
    """
    Computes sequence recovery accuracy for Protein/Ligand MPNN.

    This metric compares both the sampled predicted sequence and the argmax
    sequence to the ground truth sequence and computes the percentage of
    correctly predicted residues for both versions.
    """

    def __init__(
        self,
        return_per_example_metrics=False,
        return_per_residue_metrics=False,
        **kwargs,
    ):
        """
        Initialize the SequenceRecovery metric.

        Args:
            return_per_example_metrics (bool): If True, returns per-example
                metrics in addition to the aggregate metrics.
            return_per_residue_metrics (bool): If True, returns per-residue
                metrics in addition to the aggregate metrics.
            **kwargs: Additional keyword arguments passed to the base Metric
                class.
        """
        super().__init__(**kwargs)
        self.return_per_example_metrics = return_per_example_metrics
        self.return_per_residue_metrics = return_per_residue_metrics

    @property
    def kwargs_to_compute_args(self):
        """Map input keys to the compute method arguments.

        Returns:
            dict: Mapping from compute method argument names to nested
                dictionary keys in the input kwargs.
        """
        return {
            "S": ("network_input", "input_features", "S"),
            "S_sampled": ("network_output", "decoder_features", "S_sampled"),
            "S_argmax": ("network_output", "decoder_features", "S_argmax"),
            "mask_for_loss": ("network_output", "input_features", "mask_for_loss"),
        }

    def get_per_residue_mask(self, mask_for_loss, **kwargs):
        """
        Get the per-residue mask for computing sequence recovery.

        This method can be overridden by subclasses to apply additional
        masking criteria (e.g., interface residues only).

        Args:
            mask_for_loss (torch.Tensor): [B, L] - mask for loss
            **kwargs: Additional arguments that may be needed by subclasses

        Returns:
            per_residue_mask (torch.Tensor): [B, L] - per-residue mask for
                sequence recovery computation.
        """
        per_residue_mask = mask_for_loss
        return per_residue_mask

    def compute_sequence_recovery_metrics(self, S, S_pred, per_residue_mask):
        """
        Compute sequence recovery metrics using the ground truth sequence,
        the predicted sequence, and the per-residue mask.

        Args:
            S (torch.Tensor): [B, L] - the ground truth sequence.
            S_pred (torch.Tensor): [B, L] - the predicted sequence.
            per_residue_mask (torch.Tensor): [B, L] - per-residue mask for
                computation of sequence recovery.
        Returns:
            sequence_recovery_dict (dict): Dictionary containing the sequence
                recovery metrics.
                - mean_sequence_recovery (torch.Tensor): [1] - mean sequence
                    recovery across (valid) examples (a valid example is one
                    that has at least one valid residue according to the
                    per_residue_mask).
                - sequence_recovery_per_example (torch.Tensor): [B] - sequence
                    recovery per example, undefined for examples
                    with no valid residues.
                - correct_per_example (torch.Tensor): [B] - total number of
                    correct predictions per example.
                - correct_predictions_per_residue (torch.Tensor): [B, L] -
                    boolean tensor indicating if the predicted sequence matches
                    the ground truth sequence (1 for correct, 0 for incorrect,
                    masked by per_residue_mask).
                - total_valid_per_example [B]: number of valid residues per
                    example.
                - valid_examples_mask [B]: boolean mask indicating examples
                    with valid residues.
                - per_residue_mask [B, L]: per-residue mask for NLL computation.
        """
        per_residue_mask = per_residue_mask.float()

        # total_valid_per_example [B] - sum of valid residues per example.
        total_valid_per_example = per_residue_mask.sum(dim=-1)

        # valid_examples_mask [B] - boolean mask indicating examples with
        # valid residues.
        valid_examples_mask = total_valid_per_example > 0

        # Compute sequence recovery accuracy for sampled residues.
        # correct_predictions [B, L] - boolean tensor indicating if the
        # subject sequence matches the ground truth sequence. Masked by the
        # per_residue_mask.
        correct_predictions_per_residue = (S_pred == S).float() * per_residue_mask

        # correct_per_example [B] - sum of correct predictions per example.
        correct_per_example = correct_predictions_per_residue.sum(dim=-1)

        # sequence_recovery_per_example [B] - compute the sequence recovery
        # (accuracy) per example. Undefined if there are no valid residues.
        sequence_recovery_per_example = correct_per_example / total_valid_per_example

        # mean_sequence_recovery [1] - mean sequence recovery across
        # examples with valid residues.
        mean_sequence_recovery = sequence_recovery_per_example[
            valid_examples_mask
        ].mean()

        # Create the sequence recovery dictionary.
        sequence_recovery_dict = {
            "mean_sequence_recovery": mean_sequence_recovery,
            "sequence_recovery_per_example": sequence_recovery_per_example,
            "correct_per_example": correct_per_example,
            "correct_predictions_per_residue": correct_predictions_per_residue,
            "total_valid_per_example": total_valid_per_example,
            "valid_examples_mask": valid_examples_mask,
            "per_residue_mask": per_residue_mask,
        }

        return sequence_recovery_dict

    def compute(self, S, S_sampled, S_argmax, mask_for_loss, **kwargs):
        """
        Compute sequence recovery accuracy for both sampled and argmax
        sequences.

        This method compares both the sampled predicted sequence and the argmax
        sequence to the ground truth sequence and computes the fraction of
        correctly predicted residues for both versions (i.e. the accuracy).

        A NOTE on shapes:
            B: batch size
            L: sequence length
            vocab_size: vocabulary size

        Args:
            S (torch.Tensor): [B, L] - the ground truth sequence.
            S_sampled (torch.Tensor): [B, L] - the sampled sequence,
                sampled from the probabilities (unknown residues are not
                sampled).
            S_argmax (torch.Tensor): [B, L] - the predicted sequence,
                obtained by taking the argmax of the probabilities
                (unknown residues are not selected).
            mask_for_loss (torch.Tensor): [B, L] - mask for loss,
                where True is a residue that is included in the loss
                calculation, and False is a residue that is not included
                in the loss calculation.
            **kwargs: Additional arguments that may be needed by subclasses.

        Returns:
            metric_dict (dict): Dictionary containing the sequence recovery
                metrics.
                - mean_sequence_recovery_sampled (torch.Tensor): [1] -
                    mean sequence recovery for the sampled sequence.
                - mean_sequence_recovery_argmax (torch.Tensor): [1] -
                    mean sequence recovery for the argmax sequence.
            if self.return_per_example_metrics is True:
                - sequence_recovery_per_example_sampled (torch.Tensor): [B] -
                    sequence recovery per example for the sampled sequence,
                    undefined for examples with no valid residues.
                - sequence_recovery_per_example_argmax (torch.Tensor): [B] -
                    sequence recovery per example for the argmax sequence,
                    undefined for examples with no valid residues.
                - correct_per_example_sampled (torch.Tensor): [B] - total
                    number of correct predictions per example for the sampled
                    sequence.
                - correct_per_example_argmax (torch.Tensor): [B] - total
                    number of correct predictions per example for the argmax
                    sequence.
                - total_valid_per_example (torch.Tensor): [B] - number of valid
                    residues per example.
                - valid_examples_mask (torch.Tensor): [B] - boolean mask for
                    valid examples.
            if self.return_per_residue_metrics is True:
                - correct_predictions_per_residue_sampled (torch.Tensor):
                    [B, L] - boolean tensor indicating if the sampled
                    sequence matches the ground truth sequence (1 for correct,
                    0 for incorrect, masked by per_residue_mask).
                - correct_predictions_per_residue_argmax (torch.Tensor):
                    [B, L] - boolean tensor indicating if the argmax sequence
                    matches the ground truth sequence (1 for correct, 0 for
                    incorrect, masked by per_residue_mask).
                - per_residue_mask (torch.Tensor): [B, L] - per-residue
                    mask for sequence recovery computation.
        """
        # per_residue_mask [B, L] - mask for sequence recovery.
        per_residue_mask = self.get_per_residue_mask(mask_for_loss, **kwargs)

        # Compute sequence recovery metrics for sampled sequence.
        sequence_recovery_metrics_sampled = self.compute_sequence_recovery_metrics(
            S, S_sampled, per_residue_mask
        )

        # Compute sequence recovery metrics for argmax sequence.
        sequence_recovery_metrics_argmax = self.compute_sequence_recovery_metrics(
            S, S_argmax, per_residue_mask
        )

        # Prepare the metric dictionary.
        metric_dict = {
            "mean_sequence_recovery_sampled": sequence_recovery_metrics_sampled[
                "mean_sequence_recovery"
            ]
            .detach()
            .item(),
            "mean_sequence_recovery_argmax": sequence_recovery_metrics_argmax[
                "mean_sequence_recovery"
            ]
            .detach()
            .item(),
        }
        if self.return_per_example_metrics:
            metric_dict.update(
                {
                    "sequence_recovery_per_example_sampled": sequence_recovery_metrics_sampled[
                        "sequence_recovery_per_example"
                    ],
                    "sequence_recovery_per_example_argmax": sequence_recovery_metrics_argmax[
                        "sequence_recovery_per_example"
                    ],
                    "correct_per_example_sampled": sequence_recovery_metrics_sampled[
                        "correct_per_example"
                    ],
                    "correct_per_example_argmax": sequence_recovery_metrics_argmax[
                        "correct_per_example"
                    ],
                    "total_valid_per_example": sequence_recovery_metrics_sampled[
                        "total_valid_per_example"
                    ],
                    "valid_examples_mask": sequence_recovery_metrics_sampled[
                        "valid_examples_mask"
                    ],
                }
            )
        if self.return_per_residue_metrics:
            metric_dict.update(
                {
                    "correct_predictions_per_residue_sampled": sequence_recovery_metrics_sampled[
                        "correct_predictions_per_residue"
                    ],
                    "correct_predictions_per_residue_argmax": sequence_recovery_metrics_argmax[
                        "correct_predictions_per_residue"
                    ],
                    "per_residue_mask": sequence_recovery_metrics_sampled[
                        "per_residue_mask"
                    ],
                }
            )

        return metric_dict


class InterfaceSequenceRecovery(SequenceRecovery):
    """
    Computes sequence recovery accuracy for Protein/Ligand MPNN specifically
    for residues at the polymer-ligand interface.

    This metric inherits from SequenceRecovery but only computes metrics for
    residues that are within a specified distance threshold of ligand atoms.
    All returned metric names are prefixed with "interface_".
    """

    def __init__(
        self,
        interface_distance_threshold: float = 5.0,
        return_per_example_metrics: bool = False,
        return_per_residue_metrics: bool = False,
        **kwargs,
    ):
        """
        Initialize the InterfaceSequenceRecovery metric.

        Args:
            interface_distance_threshold (float): Distance threshold in
                Angstroms for considering residues to be at the interface.
                Defaults to 5.0.
            return_per_example_metrics (bool): If True, returns per-example
                metrics in addition to the aggregate metrics.
            return_per_residue_metrics (bool): If True, returns per-residue
                metrics in addition to the aggregate metrics.
            **kwargs: Additional keyword arguments passed to the base Metric
                class.
        """
        super().__init__(
            return_per_example_metrics=return_per_example_metrics,
            return_per_residue_metrics=return_per_residue_metrics,
            **kwargs,
        )
        self.interface_distance_threshold = interface_distance_threshold

    @property
    def kwargs_to_compute_args(self):
        """Map input keys to the compute method arguments.

        Returns:
            dict: Mapping from compute method argument names to nested
                dictionary keys in the input kwargs.
        """
        args_mapping = super().kwargs_to_compute_args
        # Add atom_array to the mapping for interface computation
        args_mapping["atom_array"] = ("network_input", "atom_array")
        return args_mapping

    def get_per_residue_mask(self, mask_for_loss, **kwargs):
        """
        Get the per-residue mask for computing interface sequence recovery.

        This method computes the interface mask by applying transforms to
        detect polymer-ligand interfaces and combines it with the original
        mask_for_loss using logical AND.

        Args:
            mask_for_loss (torch.Tensor): [B, L] - mask for loss
            **kwargs: Additional arguments including atom_array

        Returns:
            per_residue_mask (torch.Tensor): [B, L] - combined mask for
                interface sequence recovery computation.
        """
        # Extract atom arrays from kwargs
        atom_arrays = kwargs.get("atom_array")
        if atom_arrays is None:
            raise ValueError(
                "atom_array is required for interface "
                + "computation but was not found"
            )

        # Initialize transforms
        interface_transform = ComputePolymerLigandInterface(
            distance_threshold=self.interface_distance_threshold
        )
        mask_transform = FeaturizePolymerLigandInterfaceMask()
        convert_to_torch_transform = ConvertToTorch(keys=["input_features"])

        # Process each atom array in the batch
        batch_interface_masks = []
        for atom_array in atom_arrays:
            # Apply interface detection transform
            data = {"atom_array": atom_array}
            data = interface_transform(data)

            # Apply interface mask featurization
            data = mask_transform(data)

            # Convert to torch tensor
            data = convert_to_torch_transform(data)

            # Extract the interface mask
            interface_mask = data["input_features"]["polymer_ligand_interface_mask"]

            # Convert to torch tensor
            batch_interface_masks.append(interface_mask)

        # Collate interface masks with proper padding
        collator = FeatureCollator(
            default_padding={"polymer_ligand_interface_mask": False}
        )

        # Create mock pipeline outputs for collation
        mock_outputs = []
        for interface_mask in batch_interface_masks:
            mock_outputs.append(
                {
                    "input_features": {"polymer_ligand_interface_mask": interface_mask},
                    "atom_array": None,  # Not needed for collation
                }
            )

        # Collate the masks
        collated = collator(mock_outputs)
        interface_mask = collated["input_features"]["polymer_ligand_interface_mask"]

        # Convert to the same device and dtype as mask_for_loss
        interface_mask = interface_mask.to(
            device=mask_for_loss.device, dtype=mask_for_loss.dtype
        )

        # Combine with original mask using logical AND
        combined_mask = mask_for_loss & interface_mask

        return combined_mask

    def compute(self, S, S_sampled, S_argmax, mask_for_loss, atom_array, **kwargs):
        """
        Compute interface sequence recovery accuracy for both sampled and
        argmax sequences.

        This method computes sequence recovery specifically for residues at
        the polymer-ligand interface and prefixes all output metrics with
        "interface_".

        Args:
            S (torch.Tensor): [B, L] - the ground truth sequence.
            S_sampled (torch.Tensor): [B, L] - the sampled sequence.
            S_argmax (torch.Tensor): [B, L] - the predicted sequence.
            mask_for_loss (torch.Tensor): [B, L] - mask for loss.
            **kwargs: Additional arguments including atom_array.

        Returns:
            metric_dict (dict): Dictionary containing the interface sequence
                recovery metrics with "interface_" prefix.
        """
        # Get the base metrics using parent class compute method
        # Pass atom_array through kwargs for get_per_residue_mask method
        kwargs_with_atom_array = {**kwargs, "atom_array": atom_array}
        base_metrics = super().compute(
            S, S_sampled, S_argmax, mask_for_loss, **kwargs_with_atom_array
        )

        # Add "interface_" prefix to all metric keys
        interface_metrics = {}
        for key, value in base_metrics.items():
            interface_metrics[f"interface_{key}"] = value

        return interface_metrics
