import torch
from atomworks.ml.transforms.base import ConvertToTorch
from mpnn.collate.feature_collator import FeatureCollator
from mpnn.transforms.feature_aggregation.polymer_ligand_interface import (
    FeaturizePolymerLigandInterfaceMask,
)
from mpnn.transforms.polymer_ligand_interface import ComputePolymerLigandInterface

from foundry.metrics.metric import Metric


class NLL(Metric):
    """
    Computes negative log likelihood (NLL) and perplexity for Protein/Ligand
    MPNN.

    This metric computes the NLL loss by averaging the negative log
    probabilities at the true token indices, masked by the loss mask. This
    follows the same computation as LabelSmoothedNLLLoss but without label
    smoothing and with averaging instead of a normalization constant.
    """

    def __init__(
        self,
        return_per_example_metrics=False,
        return_per_residue_metrics=False,
        **kwargs,
    ):
        """
        Initialize the NLL metric.

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
        """
        Map input keys to the compute method arguments.

        Returns:
            dict: Mapping from compute method argument names to nested
                dictionary keys in the input kwargs.
        """
        return {
            "log_probs": ("network_output", "decoder_features", "log_probs"),
            "S": ("network_input", "input_features", "S"),
            "mask_for_loss": ("network_output", "input_features", "mask_for_loss"),
        }

    def get_per_residue_mask(self, mask_for_loss, **kwargs):
        """
        Get the per-residue mask for computing NLL.

        This method can be overridden by subclasses to apply additional masking
        criteria.

        Args:
            mask_for_loss (torch.Tensor): [B, L] - mask for loss
            **kwargs: Additional arguments that may be needed by subclasses
        Returns:
            per_residue_mask (torch.Tensor): [B, L] - per-residue mask for NLL
                computation.
        """
        per_residue_mask = mask_for_loss
        return per_residue_mask

    def compute_nll_metrics(self, S, log_probs, per_residue_mask):
        """
        Compute NLL and perplexity metrics using the provided per-residue mask.
        Args:
            S (torch.Tensor): [B, L] - the ground truth sequence.
            log_probs (torch.Tensor): [B, L, vocab_size] - the log
                probabilities for the sequence.
            per_residue_mask (torch.Tensor): [B, L] - per-residue mask for
                computation of NLL.
        Returns:
            nll_dict (dict): Dictionary containing the NLL metrics.
                - mean_nll [1]: mean NLL over (valid) examples (a valid example
                    is one with at least one valid residue according to the
                    per_residue_mask).
                - nll_per_example [B]: NLL per example, undefined for examples
                    with no valid residues.
                - nll_per_residue [B, L]: NLL per residue (masked, 0 for
                    masked out positions).
                - mean_perplexity [1]: mean perplexity over (valid) examples.
                - perplexity_per_example [B]: perplexity per example, undefined
                    for examples with no valid residues.
                - total_valid_per_example [B]: number of valid residues per
                    example.
                - valid_examples_mask [B]: boolean mask indicating examples
                    with valid residues.
                - per_residue_mask [B, L]: per-residue mask for NLL computation.
        """
        _, _, vocab_size = log_probs.shape
        per_residue_mask = per_residue_mask.float()

        # total_valid_per_example [B] - number of valid residues per example.
        total_valid_per_example = per_residue_mask.sum(dim=-1)

        # valid_examples_mask [B] - boolean mask indicating examples with valid
        # residues.
        valid_examples_mask = total_valid_per_example > 0

        # S_onehot [B, L, vocab_size] - the one-hot encoded sequence.
        S_onehot = torch.nn.functional.one_hot(S, num_classes=vocab_size).float()

        # nll_per_residue [B, L] - the per-residue negative log likelihood,
        # masked by the per_residue_mask.
        nll_per_residue = -torch.sum(S_onehot * log_probs, dim=-1) * per_residue_mask

        # nll_per_example [B] - average NLL per example. Undefined if there are
        # no valid residues.
        nll_per_example = nll_per_residue.sum(dim=-1) / total_valid_per_example

        # mean_nll [1] - mean of per-example NLL values (over valid examples).
        mean_nll = nll_per_example[valid_examples_mask].mean()

        # perplexity_per_example [B] - perplexity per example.
        perplexity_per_example = torch.exp(nll_per_example)

        # mean_perplexity [1] - mean of per-example perplexity values (over
        # valid examples).
        mean_perplexity = perplexity_per_example[valid_examples_mask].mean()

        nll_dict = {
            "mean_nll": mean_nll,
            "nll_per_example": nll_per_example,
            "nll_per_residue": nll_per_residue,
            "mean_perplexity": mean_perplexity,
            "perplexity_per_example": perplexity_per_example,
            "total_valid_per_example": total_valid_per_example,
            "valid_examples_mask": valid_examples_mask,
            "per_residue_mask": per_residue_mask,
        }
        return nll_dict

    def compute(self, log_probs, S, mask_for_loss, **kwargs):
        """
        Compute the negative log likelihood (NLL) and perplexity, meaned
        across all residues that are included in the loss calculation.

        Args:
            S (torch.Tensor): [B, L] - the ground truth sequence.
            log_probs (torch.Tensor): [B, L, vocab_size] - the
                log probabilities for the sequence.
            mask_for_loss (torch.Tensor): [B, L] - mask for loss,
                where True is a residue that is included in the loss
                calculation, and False is a residue that is not included
                in the loss calculation.
            **kwargs: Additional arguments that may be needed by subclasses.
        Returns:
            metric_dict (dict): Dictionary containing the computed metrics.
                - mean_nll [1]: mean NLL over (valid) examples.
                - mean_perplexity [1]: mean perplexity over (valid) examples.
            if self.return_per_example_metrics is True:
                - nll_per_example [B]: NLL per example, undefined for examples
                    with no valid residues.
                - perplexity_per_example [B]: perplexity per example, undefined
                    for examples with no valid residues.
                - total_valid_per_example [B]: number of valid residues per
                    example.
                - valid_examples_mask [B]: boolean mask indicating examples
                    with valid residues.
            if self.return_per_residue_metrics is True:
                - nll_per_residue [B, L]: NLL per residue (masked, 0 for
                    masked out positions).
                - per_residue_mask [B, L]: mask for sequence recovery.
        """
        # per_residue_mask [B, L] - mask for sequence recovery.
        per_residue_mask = self.get_per_residue_mask(mask_for_loss, **kwargs)

        # Compute NLL metrics.
        nll_metrics = self.compute_nll_metrics(S, log_probs, per_residue_mask)

        # Prepare the metric dictionary.
        metric_dict = {
            "mean_nll": nll_metrics["mean_nll"].detach().item(),
            "mean_perplexity": nll_metrics["mean_perplexity"].detach().item(),
        }
        if self.return_per_example_metrics:
            metric_dict.update(
                {
                    "nll_per_example": nll_metrics["nll_per_example"],
                    "perplexity_per_example": nll_metrics["perplexity_per_example"],
                    "total_valid_per_example": nll_metrics["total_valid_per_example"],
                    "valid_examples_mask": nll_metrics["valid_examples_mask"],
                }
            )
        if self.return_per_residue_metrics:
            metric_dict.update(
                {
                    "nll_per_residue": nll_metrics["nll_per_residue"],
                    "per_residue_mask": nll_metrics["per_residue_mask"],
                }
            )
        return metric_dict


class InterfaceNLL(NLL):
    """
    Computes negative log likelihood (NLL) and perplexity for Protein/Ligand
    MPNN specifically for residues at the polymer-ligand interface.

    This metric inherits from NLL but only computes metrics for residues that
    are within a specified distance threshold of ligand atoms. All returned
    metric names are prefixed with "interface_".
    """

    def __init__(
        self,
        interface_distance_threshold: float = 5.0,
        return_per_example_metrics: bool = False,
        return_per_residue_metrics: bool = False,
        **kwargs,
    ):
        """
        Initialize the InterfaceNLL metric.

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
        """
        Map input keys to the compute method arguments.

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
        Get the per-residue mask for computing interface NLL.

        This method computes the interface mask by applying transforms to
        detect polymer-ligand interfaces and combines it with the original
        mask_for_loss using logical AND.

        Args:
            mask_for_loss (torch.Tensor): [B, L] - mask for loss
            **kwargs: Additional arguments including atom_array

        Returns:
            per_residue_mask (torch.Tensor): [B, L] - combined mask for
                interface NLL computation.
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

    def compute(self, log_probs, S, mask_for_loss, atom_array, **kwargs):
        """
        Compute the interface negative log likelihood (NLL) and perplexity,
        averaged across interface residues only.

        This method computes NLL and perplexity specifically for residues at
        the polymer-ligand interface and prefixes all output metrics with
        "interface_".

        Args:
            log_probs (torch.Tensor): [B, L, vocab_size] - the
                log probabilities for the sequence.
            S (torch.Tensor): [B, L] - the ground truth sequence.
            mask_for_loss (torch.Tensor): [B, L] - mask for loss.
            **kwargs: Additional arguments including atom_array.

        Returns:
            metric_dict (dict): Dictionary containing the interface NLL and
                perplexity metrics with "interface_" prefix.
        """
        # Get the base metrics using parent class compute method
        # Pass atom_array through kwargs for get_per_residue_mask method
        kwargs_with_atom_array = {**kwargs, "atom_array": atom_array}
        base_metrics = super().compute(
            log_probs, S, mask_for_loss, **kwargs_with_atom_array
        )

        # Add "interface_" prefix to all metric keys
        interface_metrics = {}
        for key, value in base_metrics.items():
            interface_metrics[f"interface_{key}"] = value

        return interface_metrics
