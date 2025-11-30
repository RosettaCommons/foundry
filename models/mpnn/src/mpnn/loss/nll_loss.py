import torch
import torch.nn as nn


class LabelSmoothedNLLLoss(nn.Module):
    def __init__(self, label_smoothing_eps=0.1, normalization_constant=6000.0):
        """
        Label smoothed negative log likelihood loss for Protein/Ligand MPNN.

        Args:
            label_smoothing_eps (float): The label smoothing factor. Default is
                0.1.
            normalization_constant (float): The normalization constant for the
                loss. As opposed to averaging per sample in the batch, or
                averaging across all tokens, this constant is used to normalize
                the loss. Default is 6000.0.
        """
        super(LabelSmoothedNLLLoss, self).__init__()

        self.label_smoothing_eps = label_smoothing_eps
        self.normalization_constant = normalization_constant

    def forward(self, network_input, network_output, loss_input):
        """
        Given the network_input (same as input_features to the model), network
        output, and loss input, compute the loss.

        Args:
            network_input (dict): The input to the network.
                - input_features (dict): Contains the input features.
                    - S (torch.Tensor): [B, L] - the sequence of residues.
            network_output (dict): The output of the network, a dictionary
                containing several sub-dictionaries; the necessary sub-
                dictionaries and their needed keys are listed below:
                - input_features (dict): Contains the modified input features.
                    - mask_for_loss (torch.Tensor): [B, L] - the mask for the
                        loss computation.
                - decoder_features (dict): Contains the decoder features.
                    - log_probs (torch.Tensor): [B, L, vocab_size] - the log
                        probabilities for the sequence.
            loss_input (dict): Dictionary containing additional inputs needed
                for the loss computation. Unused here.
        Returns:
            The loss and a dictionary containing the loss values.
                - label_smoothed_nll_loss_agg (torch.Tensor): [1] - the
                    aggregated label smoothed negative log likelihood loss,
                    masked by the mask for the loss, summed across the batch and
                    length dimensions, and normalized by the normalization
                    constant. This is the final loss value returned by the loss
                    function.
                - loss_dict (dict): A dictionary containing the loss outputs.
                    - label_smoothed_nll_loss_per_residue (torch.Tensor): [B, L]
                        - the per-residue label smoothed negative log likelihood
                            loss, masked by the mask for loss.
                    - label_smoothed_nll_loss_agg (torch.Tensor): [1] - the
                        aggregated label smoothed negative log likelihood loss,
                        masked by the mask for loss, summed across the batch and
                        length dimensions, and normalized by the normalization
                        constant. This is the final loss value returned by the
                        loss function.

        """
        input_features = network_input["input_features"]

        # Check that the input features contains the necessary keys.
        if "S" not in input_features:
            raise ValueError("Input features must contain 'S' key.")

        # Check that the network output contains the necessary keys.
        if "input_features" not in network_output:
            raise ValueError("Network output must contain 'input_features' key.")
        if "mask_for_loss" not in network_output["input_features"]:
            raise ValueError(
                "Network output must contain'"
                + "mask_for_loss' key in 'input_features'."
            )
        if "decoder_features" not in network_output:
            raise ValueError("Network output must contain 'decoder_features' key.")
        if "log_probs" not in network_output["decoder_features"]:
            raise ValueError(
                "Network output must contain" + "'log_probs' key in 'decoder_features'."
            )

        B, L, vocab_size = network_output["decoder_features"]["log_probs"].shape

        # S_onehot [B, L, vocab_size] - the one-hot encoded sequence.
        S_onehot = torch.nn.functional.one_hot(
            input_features["S"], num_classes=vocab_size
        ).float()

        # label_smoothed_S_onehot [B, L, vocab_size] - the label smoothed
        # encoded sequence.
        label_smoothed_S_onehot = (
            1 - self.label_smoothing_eps
        ) * S_onehot + self.label_smoothing_eps / vocab_size

        # label_smoothed_nll_loss_per_residue [B, L] - the per-residue label
        # smoothed negative log likelihood loss, masked by the mask for loss.
        label_smoothed_nll_loss_per_residue = (
            -torch.sum(
                label_smoothed_S_onehot
                * network_output["decoder_features"]["log_probs"],
                dim=-1,
            )
            * network_output["input_features"]["mask_for_loss"]
        )

        # label_smoothed_nll_loss_agg - the aggregated label smoothed
        # negative log likelihood loss, aggregated across the batch and
        # length dimensions, and normalized by the normalization constant.
        # This is the final loss value returned by the loss function.
        label_smoothed_nll_loss_agg = (
            torch.sum(label_smoothed_nll_loss_per_residue) / self.normalization_constant
        )

        # Construct the output loss dictionary.
        loss_dict = {
            "label_smoothed_nll_loss_per_residue": label_smoothed_nll_loss_per_residue.detach(),
            "label_smoothed_nll_loss_agg": label_smoothed_nll_loss_agg.detach(),
        }

        return label_smoothed_nll_loss_agg, loss_dict
