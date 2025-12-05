import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float


class ConformerEmbeddingWeightedAverage(nn.Module):
    """Learned weighted average of reference conformer embeddings.

    Args:
        atom_level_embedding_dim: Dimension of the input atom-level embeddings (default: 384, for EGRET)
        c_atompair: Dimension of the atom-pair embeddings (default: 16)
        c_atom: Dimension of the output atom embeddings (default: 128)
        n_conformers: Number of conformers to expect (default: 8)
        dropout_rate: Dropout rate for regularization (default: 0.1)
        use_layer_norm: Whether to apply layer normalization to the output (default: True)
    """

    def __init__(
        self,
        atom_level_embedding_dim: int,
        c_atompair: int,
        c_atom: int,
        n_conformers: int = 8,
        dropout_rate: float = 0.1,
        use_layer_norm: bool = True,
    ):
        super().__init__()

        self.n_conformers = n_conformers
        self.atom_level_embedding_dim = atom_level_embedding_dim

        # Downcast MLP from atom_level_embedding_dim to c_atompair
        self.process_atom_level_embedding = nn.Sequential(
            nn.Linear(atom_level_embedding_dim, atom_level_embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(atom_level_embedding_dim // 2, atom_level_embedding_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(atom_level_embedding_dim // 4, atom_level_embedding_dim // 8),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(atom_level_embedding_dim // 8, c_atompair),
        )

        # Final MLP to convert from (n_conformers * c_atompair) to c_atom
        self.conformers_to_atom_single_embedding = nn.Sequential(
            nn.Linear(n_conformers * c_atompair, c_atom, bias=False),
            nn.LayerNorm(c_atom) if use_layer_norm else nn.Identity(),
        )

        # Zero-init the final linear layer to ensure the model starts with identity function (output ≈ 0)
        nn.init.zeros_(self.conformers_to_atom_single_embedding[0].weight)

    def forward(
        self,
        atom_level_embeddings: Float[
            torch.Tensor, "n_conformers n_atom atom_level_embedding_dim"
        ],
    ) -> Float[torch.Tensor, "n_atom c_atom"]:
        """Forward pass: process atom-level embeddings and return the processed result.

        Args:
            atom_level_embeddings: Input tensor of shape [n_conformers, n_atom, atom_level_embedding_dim]

        Returns:
            Processed tensor of shape [n_atom, c_atom] ready for residual addition
        """
        assert (
            atom_level_embeddings.shape[0] == self.n_conformers
        ), "Number of conformers must be consistent"

        # Subset to [:atom_level_embedding_dim]
        if atom_level_embeddings.shape[-1] > self.atom_level_embedding_dim:
            atom_level_embeddings = atom_level_embeddings[
                ..., : self.atom_level_embedding_dim
            ]
        elif atom_level_embeddings.shape[-1] < self.atom_level_embedding_dim:
            raise ValueError(
                f"Atom-level embedding dimension {atom_level_embeddings.shape[-1]} is less than the expected dimension {self.atom_level_embedding_dim}"
            )

        # Process atom-level embeddings to get shape [n_conformers, n_atom, c_atompair]
        processed_embeddings: Float[torch.Tensor, "n_conformers n_atom c_atompair"] = (
            self.process_atom_level_embedding(atom_level_embeddings)
        )

        # Pad with zeros if we don't have enough conformers
        current_n_conformers = processed_embeddings.shape[0]
        if current_n_conformers < self.n_conformers:
            # Pad with zeros at the beginning
            padding_size = self.n_conformers - current_n_conformers
            padding: Float[torch.Tensor, "padding_size n_atom c_atompair"] = (
                torch.zeros(
                    padding_size,
                    processed_embeddings.shape[1],
                    processed_embeddings.shape[2],
                    device=processed_embeddings.device,
                    dtype=processed_embeddings.dtype,
                )
            )
            processed_embeddings = torch.cat([padding, processed_embeddings], dim=0)
        elif current_n_conformers > self.n_conformers:
            # Truncate to n_conformers
            processed_embeddings = processed_embeddings[: self.n_conformers]

        # Reshape from [n_conformers, n_atom, c_atompair] to [n_atom, n_conformers * c_atompair]
        reshaped_embeddings: Float[torch.Tensor, "n_atom n_conformers*c_atompair"] = (
            rearrange(processed_embeddings, "c n d -> n (c d)")
        )

        # Final MLP to get [n_atom, c_atom]
        result: Float[torch.Tensor, "n_atom c_atom"] = (
            self.conformers_to_atom_single_embedding(reshaped_embeddings)
        )

        return result
