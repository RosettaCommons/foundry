from functools import partial

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from modelhub.model.AF3_blocks import AF3_block, AF3_full_block
from modelhub.model.layers.Attention_module import (
    BiasedAxialAttention,
    BiasedUntiedAxialAttention,
    FeedForwardLayer,
    MSAColAttention,
    MSAColGlobalAttention,
    MSARowAttentionWithBias,
    OldMSAColAttention,
    OldMSAColGlobalAttention,
    TriangleMultiplication,
)
from modelhub.model.layers.outer_product import (
    OuterProductMean,
)  # need to code this correctly
from modelhub.model.layers.SE3_network import FullyConnectedSE3, FullyConnectedSE3_noR
from modelhub.model.layers.structure_bias import structure_bias_factory
from modelhub.training.checkpoint import create_custom_forward
from modelhub.util_module import Dropout


class RF2_block(nn.Module):
    """
    nearly faithful implementation of RF2aa blocks in new paradigm
    unfaithful portions are:
        - ablating adding the positional encodings as biases to the attentions
        - the "is_bonded" boolean feature is no longer embedded with the edge features of the SE3 transformer
    """

    def __init__(
        self, global_config, block_params, is_full, backprop_through_xyz=False, **kwargs
    ):
        super(RF2_block, self).__init__()
        d_msa, d_msa_full, d_pair, d_state = (
            global_config.d_msa,
            global_config.d_msa_full,
            global_config.d_pair,
            global_config.d_state,
        )
        self.is_full = is_full
        self.backprop_through_xyz = backprop_through_xyz

        if self.is_full:
            d_msa = d_msa_full

        self.layer_dropout = block_params.p_drop_layer  # fd layer dropout

        self.norm_pair_bias = nn.LayerNorm(d_pair, bias=block_params.bias_in_attn_norm)
        self.norm_state_bias = nn.LayerNorm(
            d_state, bias=block_params.bias_in_attn_norm
        )
        self.proj_state_bias = nn.Linear(d_state, d_msa)
        self.msa_str_bias = structure_bias_factory["ungated"](
            block_params.d_rbf, d_pair
        )

        self.drop_row = Dropout(broadcast_dim=1, p_drop=block_params.p_drop_row)
        self.drop_col = Dropout(broadcast_dim=2, p_drop=block_params.p_drop_pair)

        self.msa_row_attn = MSARowAttentionWithBias(
            d_msa=d_msa,
            d_pair=d_pair,
            n_head=block_params.n_msa_head,
            d_hidden=block_params.n_msa_channels,
            nseq_normalization=block_params.norm_msa_row,
            bias=block_params.bias_in_attn_norm,
        )
        if self.is_full:
            if block_params.use_flash_attn:
                self.msa_col_attn = MSAColGlobalAttention(
                    d_msa=d_msa,
                    n_head=block_params.n_msa_head,
                    d_hidden=block_params.n_msa_channels,
                    bias=block_params.bias_in_attn_norm,
                )
            else:
                self.msa_col_attn = OldMSAColGlobalAttention(
                    d_msa=d_msa,
                    n_head=block_params.n_msa_head,
                    d_hidden=block_params.n_msa_channels,
                )
        else:
            if block_params.use_flash_attn:
                self.msa_col_attn = MSAColAttention(
                    d_msa=d_msa,
                    n_head=block_params.n_msa_head,
                    d_hidden=block_params.n_msa_channels,
                    bias=block_params.bias_in_attn_norm,
                )
            else:
                self.msa_col_attn = OldMSAColAttention(
                    d_msa=d_msa,
                    n_head=block_params.n_msa_head,
                    d_hidden=block_params.n_msa_channels,
                )
        self.msa_transition = FeedForwardLayer(
            d_msa, 4, p_drop=block_params.msa_transition_drop
        )

        # Pair update parameters
        self.outer_product = OuterProductMean(
            d_msa,
            d_pair,
            d_hidden=block_params.outer_product_channels,
            p_drop=block_params.p_drop_outer_product,
        )

        self.structure_bias = structure_bias_factory["gated"](
            block_params.d_rbf,
            d_state,
            d_pair,
            block_params.structure_bias_gate_channels,
        )
        self.tri_mul_outgoing = TriangleMultiplication(
            d_pair,
            d_hidden=block_params.n_pair_channels,
            outgoing=True,
            bias=block_params.bias_in_attn_norm,
        )
        self.tri_mul_incoming = TriangleMultiplication(
            d_pair,
            d_hidden=block_params.n_pair_channels,
            outgoing=False,
            bias=block_params.bias_in_attn_norm,
        )
        self.pair_row_attn = BiasedAxialAttention(
            d_pair,
            d_pair,
            block_params.n_pair_head,
            block_params.n_pair_channels,
            p_drop=block_params.p_drop_pair,
            is_row=True,
            bias=block_params.bias_in_attn_norm,
        )
        self.pair_col_attn = BiasedAxialAttention(
            d_pair,
            d_pair,
            block_params.n_pair_head,
            block_params.n_pair_channels,
            p_drop=block_params.p_drop_pair,
            is_row=False,
            bias=block_params.bias_in_attn_norm,
        )
        self.pair_transition = FeedForwardLayer(
            d_pair, 2
        )  # HACK: hardcoded value for transition

        self.structure_attn = FullyConnectedSE3(
            d_msa,
            d_pair,
            block_params.d_rbf,
            block_params.n_se3_layers,
            block_params.n_se3_channels,
            block_params.n_se3_degrees,
            block_params.n_se3_head,
            block_params.n_div,
            block_params.l0_in_features,
            block_params.l0_out_features,
            block_params.l1_in_features,
            block_params.l1_out_features,
            block_params.n_se3_edge_features,
            block_params.sc_pred_d_hidden,
            block_params.sc_pred_p_drop,
            block_params.residual_state,
            compute_gradients=block_params.compute_gradients,
        )

    def _unpack_inputs(self, latent_feats):
        pair, state, xyz, is_atom, atom_frames, chirals = (
            latent_feats["pair"],
            latent_feats["state"],
            latent_feats["xyz"],
            latent_feats["is_atom"],
            latent_feats["atom_frames"],
            latent_feats["chirals"],
        )
        if self.is_full:
            msa = latent_feats["msa_full"]
        else:
            msa = latent_feats["msa"]
        bond_feats, dist_matrix, idx = (
            latent_feats["bond_feats"],
            latent_feats["dist_matrix"],
            latent_feats["idx"],
        )
        return (
            msa,
            pair,
            state,
            xyz[..., :3, :],
            is_atom,
            atom_frames,
            chirals,
            bond_feats,
            dist_matrix,
            idx,
            latent_feats["is_motif"],
        )

    def _pack_outputs(self, msa, pair, state, xyz, alpha, quat_update, latent_feats):
        if self.is_full:
            latent_feats["msa_full"] = msa
        else:
            latent_feats["msa"] = msa
        latent_feats["pair"] = pair
        latent_feats["state"] = state
        latent_feats["xyz"] = xyz
        # HACK: appending to growing list, this could cause weird memory problems in pytorch
        # eventually want to refactor this to make it more elegant
        if "xyz_intermediate" not in latent_feats:
            latent_feats["xyz_intermediate"] = [xyz]
        else:
            latent_feats["xyz_intermediate"].append(xyz)

        if "alpha_intermediate" not in latent_feats:
            latent_feats["alpha_intermediate"] = [alpha]
        else:
            latent_feats["alpha_intermediate"].append(alpha)

        if "quat_update" not in latent_feats:
            latent_feats["quat_update"] = [quat_update]
        else:
            latent_feats["quat_update"].append(quat_update)
        return latent_feats

    def _1d_update(self, msa, pair, state, xyz, drop_layer=False):
        weight = 0.0 if drop_layer else 1.0

        pair = self.norm_pair_bias(pair)
        pair = pair + weight * self.msa_str_bias(xyz)

        state = self.norm_state_bias(state)
        state_update = weight * self.proj_state_bias(state)

        msa = msa.type_as(state_update)
        msa = msa.index_add(
            1,
            torch.tensor(
                [
                    0,
                ],
                device=state_update.device,
            ),
            weight * state_update[None],
        )
        msa = msa + weight * self.drop_row(self.msa_row_attn(msa, pair))
        msa = msa + weight * self.msa_col_attn(msa)
        msa = msa + weight * self.msa_transition(msa)

        return msa

    def _2d_update(self, msa, pair, state, xyz, drop_layer=False):
        weight = 0.0 if drop_layer else 1.0

        msa_bias = self.outer_product(msa)
        pair = pair + weight * msa_bias
        str_bias = self.structure_bias(xyz, state)
        pair = pair + weight * self.drop_row(self.tri_mul_outgoing(pair))
        pair = pair + weight * self.drop_row(self.tri_mul_incoming(pair))
        pair = pair + weight * self.drop_row(self.pair_row_attn(pair, str_bias))
        pair = pair + weight * self.drop_col(self.pair_col_attn(pair, str_bias))
        pair = pair + weight * self.pair_transition(pair)
        return pair

    def _3d_update(
        self,
        msa,
        pair,
        state,
        xyz,
        is_atom,
        atom_frames,
        chirals,
        bond_feats,
        dist_matrix,
        idx,
        drop_layer=False,
        is_motif=None,
    ):
        if not self.backprop_through_xyz:
            xyz = xyz.detach()
        block_outputs = self.structure_attn(
            msa,
            pair,
            state,
            xyz,
            is_atom,
            atom_frames,
            chirals,
            idx,
            bond_feats,
            dist_matrix,
            drop_layer=drop_layer,
            is_motif=is_motif,
        )
        return (
            block_outputs["state"],
            block_outputs["xyz"],
            block_outputs["alpha"],
            block_outputs["quat_update"],
        )

    def forward(self, latent_feats, use_checkpoint, use_amp):
        (
            msa,
            pair,
            state,
            xyz,
            is_atom,
            atom_frames,
            chirals,
            bond_feats,
            dist_matrix,
            idx,
            is_motif,
        ) = self._unpack_inputs(latent_feats)
        drop_layer = 0
        if use_checkpoint:
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
                msa = checkpoint.checkpoint(
                    create_custom_forward(self._1d_update, drop_layer=drop_layer),
                    msa,
                    pair,
                    state,
                    xyz,
                    use_reentrant=True,
                )
                pair = checkpoint.checkpoint(
                    create_custom_forward(self._2d_update, drop_layer=drop_layer),
                    msa,
                    pair,
                    state,
                    xyz,
                    use_reentrant=True,
                )
            # 3D track cannot use re-entrant = False because of chiral features call to autograd
            # TODO: allow this to happen since new versions of Pytorch will be using reentrant=False
            msa, pair = msa.float(), pair.float()
            state, xyz, alpha, quat_update = checkpoint.checkpoint(
                create_custom_forward(
                    self._3d_update, drop_layer=drop_layer, is_motif=is_motif
                ),
                msa,
                pair,
                state,
                xyz,
                is_atom,
                atom_frames,
                chirals,
                bond_feats,
                dist_matrix,
                idx,
                use_reentrant=True,
            )
        else:
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
                msa = self._1d_update(msa, pair, state, xyz, drop_layer=drop_layer)
                pair = self._2d_update(msa, pair, state, xyz, drop_layer=drop_layer)
            msa, pair = msa.float(), pair.float()
            state, xyz, alpha, quat_update = self._3d_update(
                msa,
                pair,
                state,
                xyz,
                is_atom,
                atom_frames,
                chirals,
                bond_feats,
                dist_matrix,
                idx,
                drop_layer=drop_layer,
                is_motif=is_motif,
            )

        latent_feats = self._pack_outputs(
            msa, pair, state, xyz, alpha, quat_update, latent_feats
        )
        return latent_feats


class RF2_withgradients(RF2_block):
    def __init__(self, global_config, block_params, is_full, **kwargs):
        super().__init__(global_config, block_params, is_full, **kwargs)
        d_msa, d_msa_full, d_pair = (
            global_config.d_msa,
            global_config.d_msa_full,
            global_config.d_pair,
        )
        self.is_full = is_full

        if self.is_full:
            d_msa = d_msa_full

        self.structure_attn = FullyConnectedSE3_noR(
            d_msa,
            d_pair,
            block_params.d_rbf,
            block_params.n_se3_layers,
            block_params.n_se3_channels,
            block_params.n_se3_degrees,
            block_params.n_se3_head,
            block_params.n_div,
            block_params.l0_in_features,
            block_params.l0_out_features,
            block_params.l1_in_features,
            block_params.l1_out_features,
            block_params.n_se3_edge_features,
            block_params.residual_state,
            compute_gradients=block_params.compute_gradients,
        )

    def _3d_update(
        self,
        msa,
        pair,
        state,
        xyz,
        is_atom,
        atom_frames,
        chirals,
        bond_feats,
        dist_matrix,
        idx,
        drop_layer=False,
        is_motif=None,
    ):
        block_outputs = self.structure_attn(
            msa,
            pair,
            state,
            xyz,
            is_atom,
            atom_frames,
            chirals,
            idx,
            bond_feats,
            dist_matrix,
            drop_layer=drop_layer,
            is_motif=is_motif,
        )
        block_outputs["alpha"] = None
        return (
            block_outputs["state"],
            block_outputs["xyz"],
            block_outputs["alpha"],
            None,
        )

    def _pack_outputs(self, msa, pair, state, xyz, alpha, quat_update, latent_feats):
        if self.is_full:
            latent_feats["msa_full"] = msa
        else:
            latent_feats["msa"] = msa
        latent_feats["pair"] = pair
        latent_feats["state"] = state
        latent_feats["xyz"] = xyz
        ##HACK: appending to growing list, this could cause weird memory problems in pytorch
        ## eventually want to refactor this to make it more elegant
        # if "xyz_intermediate" not in latent_feats:
        # latent_feats["xyz_intermediate"] = [xyz]
        # else:
        # latent_feats["xyz_intermediate"].append(xyz)

        # if "alpha_intermediate" not in latent_feats:
        # latent_feats["alpha_intermediate"] = [alpha]
        # else:
        # latent_feats["alpha_intermediate"].append(alpha)
        return latent_feats


class RF2_untied(RF2_block):
    def __init__(self, global_config, block_params, is_full, **kwargs):
        super().__init__(global_config, block_params, is_full, **kwargs)
        d_pair = global_config.d_pair
        self.pair_row_attn = BiasedUntiedAxialAttention(
            d_pair,
            d_pair,
            block_params.n_pair_head,
            block_params.n_pair_channels,
            is_row=True,
        )
        self.pair_col_attn = BiasedUntiedAxialAttention(
            d_pair,
            d_pair,
            block_params.n_pair_head,
            block_params.n_pair_channels,
            is_row=False,
        )


block_factory = {
    "AF3": AF3_block,
    "AF3_full": AF3_full_block,
    "RF2aa": partial(RF2_block, is_full=False),
    "RF2aa_full": partial(RF2_block, is_full=True),
    "untied_p2p": partial(RF2_untied, is_full=False),
    "untied_p2p_full": partial(RF2_untied, is_full=True),
    "RF2_withgradients": partial(RF2_withgradients, is_full=False),
    "RF2_withgradients_full": partial(RF2_withgradients, is_full=True),
    "RF2_withgradients_R": partial(RF2_block, is_full=False, backprop_through_xyz=True),
}
