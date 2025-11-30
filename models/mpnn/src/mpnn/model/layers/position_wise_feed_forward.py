import torch
import torch.nn as nn


class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        """
        Position-wise feed-forward layer.

        Args:
            num_hidden (int): The hidden dimension size of the input and output.
            num_ff (int): The hidden dimension size of the feed-forward layer.
        """
        super(PositionWiseFeedForward, self).__init__()

        # Initialize the linear layers for the position-wise feed-forward
        # layer with bias.
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)

        self.act = torch.nn.GELU()

    def forward(self, h_V):
        """
        Forward pass of the position-wise feed-forward layer.

        Args:
            h_V (torch.Tensor): [B, L, num_hidden] - the hidden embedding
                of the node features.
        Returns:
            feed_forward_output (torch.Tensor): [B, L, num_hidden] - the output
                of the position-wise feed-forward layer.
        """
        # feed_forward_latent [B, L, num_ff] - the input, projected with a
        # linear layer to the feed-forward dimension, and then passed through
        # the activation function.
        feed_forward_latent = self.act(self.W_in(h_V))

        # feed_forward_output [B, L, num_hidden] - the output of the
        # position-wise feed-forward layer, projected back to the hidden
        # dimension with a linear layer.
        feed_forward_output = self.W_out(feed_forward_latent)

        return feed_forward_output
