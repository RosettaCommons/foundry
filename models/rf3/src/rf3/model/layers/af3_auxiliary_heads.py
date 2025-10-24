import torch
import torch.nn as nn
import torch.nn.functional as F
from rf3.model.RF3_structure import PairformerBlock, linearNoBias

# TODO: Get from RF2AA encoding instead
CHEM_DATA_LEGACY = {"NHEAVY": 23, "aa2num": {"UNK": 20, "GLY": 7, "MAS": 21}}


def discretize_distance_matrix(
    distance_matrix, num_bins=38, min_distance=3.25, max_distance=50.75
):
    # Calculate the bin width
    bin_width = (max_distance - min_distance) / num_bins
    bins = (
        torch.arange(num_bins, device=distance_matrix.device) * bin_width + min_distance
    )

    # Discretize distances into bins (bucketize automatically places out-of-range values in the last bin)
    binned_distances = torch.bucketize(distance_matrix, bins)

    return binned_distances


class ConfidenceHead(nn.Module):
    """Algorithm 31"""

    def __init__(
        self,
        c_s,
        c_z,
        n_pairformer_layers,
        pairformer,
        n_bins_pae,
        n_bins_pde,
        n_bins_plddt,
        n_bins_exp_resolved,
        use_Cb_distances=False,
        use_af3_style_binning_and_final_layer_norms=False,
        symmetrize_Cb_logits=True,
        layer_norm_along_feature_dimension=False,
    ):
        super(ConfidenceHead, self).__init__()
        self.process_s_inputs_right = linearNoBias(449, c_z)
        self.process_s_inputs_left = linearNoBias(449, c_z)
        self.use_af3_style_binning_and_final_layer_norms = (
            use_af3_style_binning_and_final_layer_norms
        )
        self.layer_norm_along_feature_dimension = layer_norm_along_feature_dimension
        if self.use_af3_style_binning_and_final_layer_norms:
            self.layernorm_pde = nn.LayerNorm(c_z)
            self.layernorm_pae = nn.LayerNorm(c_z)
            self.layernorm_plddt = nn.LayerNorm(c_s)
            self.layernorm_exp_resolved = nn.LayerNorm(c_s)
            self.process_pred_distances = linearNoBias(40, c_z)
        else:
            self.process_pred_distances = linearNoBias(11, c_z)

        self.pairformer = nn.ModuleList(
            [
                PairformerBlock(c_s=c_s, c_z=c_z, **pairformer)
                for _ in range(n_pairformer_layers)
            ]
        )

        self.predict_pae = linearNoBias(c_z, n_bins_pae)
        self.predict_pde = linearNoBias(c_z, n_bins_pde)
        self.predict_plddt = linearNoBias(
            c_s, CHEM_DATA_LEGACY["NHEAVY"] * n_bins_plddt
        )
        self.predict_exp_resolved = linearNoBias(
            c_s, CHEM_DATA_LEGACY["NHEAVY"] * n_bins_exp_resolved
        )
        self.use_Cb_distances = use_Cb_distances
        if self.use_Cb_distances:
            self.process_Cb_distances = linearNoBias(25, c_z)
        self.symmetrize_Cb_logits = symmetrize_Cb_logits

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        S_inputs_I,
        S_trunk_I,
        Z_trunk_II,
        X_pred_L,
        seq,
        rep_atoms,
        frame_atom_idxs=None,
    ):
        # stopgrad on S_trunk_I, Z_trunk_II, X_pred_L but not S_inputs_I (4.3.5)
        S_trunk_I = S_trunk_I.detach().float()  # B, L, 384
        Z_trunk_II = Z_trunk_II.detach().float()  # B, L, L, 128
        if X_pred_L is not None:
            X_pred_L = X_pred_L.detach().float()  # B, n_atoms, 3
        S_inputs_I = S_inputs_I.detach().float()  # B, L, 384
        seq = seq.detach()

        if self.layer_norm_along_feature_dimension:
            # do a layer norm on S_trunk_I
            S_trunk_I = F.layer_norm(S_trunk_I, normalized_shape=(S_trunk_I.shape[-1]))
            # do a layer norm on Z_trunk_II
            Z_trunk_II = F.layer_norm(
                Z_trunk_II, normalized_shape=(Z_trunk_II.shape[-1])
            )
            # do a layer norm on S_inputs_I
            S_inputs_I = F.layer_norm(
                S_inputs_I, normalized_shape=(S_inputs_I.shape[-1])
            )
        else:
            S_trunk_I = F.layer_norm(S_trunk_I, normalized_shape=(S_trunk_I.shape))
            Z_trunk_II = F.layer_norm(Z_trunk_II, normalized_shape=(Z_trunk_II.shape))
            S_inputs_I = F.layer_norm(S_inputs_I, normalized_shape=(S_inputs_I.shape))

        # embed S_inputs_I twice
        S_inputs_I_right = self.process_s_inputs_right(S_inputs_I)
        S_inputs_I_left = self.process_s_inputs_left(S_inputs_I)
        # add outer product of two linear embeddings of S_inputs_I  to Z_II
        # TODO: check the unsqueezed dimension is the correct one
        Z_trunk_II = Z_trunk_II + (
            S_inputs_I_right.unsqueeze(-2) + S_inputs_I_left.unsqueeze(-3)
        )

        # embed distances of representative atom from every token
        #    in the pair representation
        # if no coords are input, skip this connection
        if X_pred_L is not None:
            X_pred_rep_I = X_pred_L.index_select(1, rep_atoms)
            dist = torch.cdist(X_pred_rep_I, X_pred_rep_I)
            if not self.use_af3_style_binning_and_final_layer_norms:
                # bins are 3.375 to 20.375 in 1.75 increments according to pseudocode
                dist_one_hot = F.one_hot(
                    discretize_distance_matrix(
                        dist, min_distance=3.375, max_distance=20.875, num_bins=10
                    ),
                    num_classes=11,
                )
            else:
                # published code is 3.25 to 50.75, with 39 bins
                dist_one_hot = F.one_hot(
                    discretize_distance_matrix(
                        dist, min_distance=3.25, max_distance=50.75, num_bins=39
                    ),
                    num_classes=40,
                )

            Z_trunk_II = Z_trunk_II + self.process_pred_distances(dist_one_hot.float())

            if self.use_Cb_distances:
                # embed difference between observed cb and ideal cb positions
                Cb_distances = calc_Cb_distances(
                    X_pred_L, seq, rep_atoms, frame_atom_idxs
                )
                Cb_distances_one_hot = F.one_hot(
                    discretize_distance_matrix(
                        Cb_distances,
                        min_distance=0.0001,
                        max_distance=0.25,
                        num_bins=24,
                    ),
                    num_classes=25,
                )
                Cb_logits = self.process_Cb_distances(Cb_distances_one_hot.float())
                # symmetrize the logits
                if self.symmetrize_Cb_logits:
                    Cb_logits = Cb_logits[:, None, :, :] + Cb_logits[:, :, None, :]
                else:
                    Cb_logits = Cb_logits[:, None, :, :]

                Z_trunk_II = Z_trunk_II + Cb_logits

        if not self.use_af3_style_binning_and_final_layer_norms:
            S_trunk_residual_I = S_trunk_I.clone()
            Z_trunk_residual_II = Z_trunk_II.clone()

        # process with pairformer stack
        for n in range(len(self.pairformer)):
            S_trunk_I, Z_trunk_II = self.pairformer[n](S_trunk_I, Z_trunk_II)

        # despite doing so in their pseudocode, af3's published code does not add the residual back
        if not self.use_af3_style_binning_and_final_layer_norms:
            S_trunk_I = S_trunk_residual_I + S_trunk_I
            Z_trunk_II = Z_trunk_residual_II + Z_trunk_II

            # linearly project for each prediction task
            pde_logits = self.predict_pde(
                Z_trunk_II + Z_trunk_II.transpose(-2, -3)
            )  # BUG: needs to be symmetrized correctly

            pae_logits = self.predict_pae(Z_trunk_II)

            plddt_logits = self.predict_plddt(S_trunk_I)
            exp_resolved_logits = self.predict_exp_resolved(S_trunk_I)

        # af3's published code does not add the residual back and has some additional layernorms before the linear projections
        # they also do the pde slightly differently, adding the transpose after the linear projection
        else:
            left_distance_logits = self.predict_pde(self.layernorm_pde(Z_trunk_II))
            right_distance_logits = left_distance_logits.transpose(-2, -3)
            pde_logits = left_distance_logits + right_distance_logits

            pae_logits = self.predict_pae(self.layernorm_pae(Z_trunk_II))
            plddt_logits = self.predict_plddt(self.layernorm_plddt(S_trunk_I))
            exp_resolved_logits = self.predict_exp_resolved(
                self.layernorm_exp_resolved(S_trunk_I)
            )

        return dict(
            pde_logits=pde_logits,
            pae_logits=pae_logits,
            plddt_logits=plddt_logits,
            exp_resolved_logits=exp_resolved_logits,
        )


def calc_Cb_distances(X_pred_L, seq, rep_atoms, frame_atom_idxs):
    frame_atom_idxs = frame_atom_idxs.unsqueeze(0).expand(X_pred_L.shape[0], -1, -1)

    N = torch.gather(
        X_pred_L, 1, frame_atom_idxs[..., 0].unsqueeze(-1).expand(-1, -1, 3)
    )
    Ca = torch.gather(
        X_pred_L, 1, frame_atom_idxs[..., 1].unsqueeze(-1).expand(-1, -1, 3)
    )
    C = torch.gather(
        X_pred_L, 1, frame_atom_idxs[..., 2].unsqueeze(-1).expand(-1, -1, 3)
    )
    Cb = X_pred_L.index_select(1, rep_atoms)

    is_valid_Cb = (
        (seq != CHEM_DATA_LEGACY.aa2num["UNK"])
        & (seq != CHEM_DATA_LEGACY.aa2num["GLY"])
        & (seq != CHEM_DATA_LEGACY.aa2num["MAS"])
    )

    def _legacy_is_protein(seq):
        return (seq >= 0).all() & (seq < 20).all()

    is_valid_Cb = is_valid_Cb & _legacy_is_protein(seq)

    b = Ca - N
    c = C - Ca
    a = torch.cross(b, c, dim=-1)

    ideal_Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + Ca

    Cb_distances = torch.norm(Cb - ideal_Cb, dim=-1)
    Cb_distances[:, ~is_valid_Cb] = 0.0

    return Cb_distances
