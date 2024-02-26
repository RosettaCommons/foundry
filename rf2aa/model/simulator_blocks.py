import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from functools import partial
import numpy as np

from rf2aa.debug import debug_nans
from rf2aa.model.layers.SE3_network import FullyConnectedSE3, FullyConnectedSE3_noR
from rf2aa.model.layers.structure_bias import structure_bias_factory
from rf2aa.model.layers.Attention_module import BiasedAxialAttention, FeedForwardLayer, MSAColAttention, \
    MSARowAttentionWithBias, TriangleMultiplication, MSAColGlobalAttention
from rf2aa.model.layers.outer_product import OuterProductMean # need to code this correctly
from rf2aa.training.checkpoint import create_custom_forward
from rf2aa.util_module import Dropout


class RF2_block(nn.Module):
    """ 
    nearly faithful implementation of RF2aa blocks in new paradigm 
    unfaithful portions are:
        - ablating adding the positional encodings as biases to the attentions
        - the "is_bonded" boolean feature is no longer embedded with the edge features of the SE3 transformer
    """
    
    def __init__(self, global_config, block_params, is_full, **kwargs):
        super(RF2_block, self).__init__()
        d_msa, d_msa_full, d_pair, d_state = global_config.d_msa, global_config.d_msa_full, global_config.d_pair, \
                                                global_config.d_state
        self.is_full = is_full
        
        if self.is_full:
            d_msa = d_msa_full

        self.layer_dropout = block_params.p_drop_layer #fd layer dropout

        self.norm_pair_bias = nn.LayerNorm(d_pair)
        self.norm_state_bias = nn.LayerNorm(d_state)
        self.proj_state_bias = nn.Linear(d_state, d_msa)
        self.msa_str_bias = structure_bias_factory["ungated"](block_params.d_rbf, d_pair)

        self.drop_row = Dropout(broadcast_dim=1, p_drop=block_params.p_drop_row)
        self.drop_col = Dropout(broadcast_dim=2, p_drop=block_params.p_drop_pair)

        self.msa_row_attn = MSARowAttentionWithBias(
            d_msa=d_msa, d_pair=d_pair, n_head=block_params.n_msa_head, d_hidden=block_params.n_msa_channels)
        if self.is_full:
            self.msa_col_attn = MSAColGlobalAttention(
                d_msa=d_msa,
                n_head=block_params.n_msa_head,
                d_hidden=block_params.n_msa_channels
        )
        else:
            self.msa_col_attn = MSAColAttention(
                d_msa=d_msa, 
                n_head=block_params.n_msa_head, 
                d_hidden=block_params.n_msa_channels
            )
        self.msa_transition = FeedForwardLayer(d_msa, 4, p_drop=block_params.msa_transition_drop)
        
        # Pair update parameters
        self.outer_product = OuterProductMean(d_msa, d_pair, d_hidden=block_params.outer_product_channels, \
                                              p_drop=block_params.p_drop_outer_product)
        
        
        self.structure_bias = structure_bias_factory["gated"](block_params.d_rbf, d_state, d_pair, block_params.structure_bias_gate_channels)
        self.tri_mul_outgoing = TriangleMultiplication(d_pair, d_hidden=block_params.n_pair_channels, outgoing=True)
        self.tri_mul_incoming = TriangleMultiplication(d_pair, d_hidden=block_params.n_pair_channels, outgoing=False)
        self.pair_row_attn = BiasedAxialAttention(d_pair, d_pair, block_params.n_pair_head, block_params.n_pair_channels, p_drop=block_params.p_drop_pair, is_row=True)
        self.pair_col_attn = BiasedAxialAttention(d_pair, d_pair, block_params.n_pair_head, block_params.n_pair_channels, p_drop=block_params.p_drop_pair, is_row=False)
        self.pair_transition = FeedForwardLayer(d_pair, 2) # HACK: hardcoded value for transition

        self.structure_attn = FullyConnectedSE3(d_msa, 
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
                                                block_params.sc_pred_p_drop
                                                )
        
    def _unpack_inputs(self, latent_feats):

        pair, state, xyz, is_atom, atom_frames, chirals = \
            latent_feats["pair"], latent_feats["state"], \
            latent_feats["xyz"], latent_feats["is_atom"], \
                latent_feats["atom_frames"], latent_feats["chirals"]
        if self.is_full:
            msa = latent_feats["msa_full"]
        else:
            msa = latent_feats["msa"]
        bond_feats, dist_matrix, idx = latent_feats["bond_feats"], latent_feats["dist_matrix"], latent_feats["idx"]
        return msa, pair, state, xyz[..., :3, :], is_atom, atom_frames, chirals, bond_feats, dist_matrix, idx

    def _pack_outputs(self, msa, pair, state, xyz, alpha, latent_feats):
        if self.is_full:
            latent_feats["msa_full"] = msa
        else:
            latent_feats["msa"] = msa
        latent_feats["pair"] = pair
        latent_feats["state"] = state
        latent_feats["xyz"] = xyz
        #HACK: appending to growing list, this could cause weird memory problems in pytorch
        # eventually want to refactor this to make it more elegant
        if "xyz_intermediate" not in latent_feats:
            latent_feats["xyz_intermediate"] = [xyz]
        else:
            latent_feats["xyz_intermediate"].append(xyz)
        
        if "alpha_intermediate" not in latent_feats:
            latent_feats["alpha_intermediate"] = [alpha]
        else:
            latent_feats["alpha_intermediate"].append(alpha)
        return latent_feats

    def _1d_update(self, msa, pair, state, xyz, drop_layer=False):
        weight = 0. if drop_layer else 1.

        pair = self.norm_pair_bias(pair)
        pair = pair + weight*self.msa_str_bias(xyz)

        state = self.norm_state_bias(state)
        state_update = weight*self.proj_state_bias(state)

        msa = msa.type_as(state_update)
        msa = msa.index_add(1, torch.tensor([0,], device=state_update.device), weight*state_update[None])
        msa = msa + weight*self.drop_row(self.msa_row_attn(msa, pair))
        msa = msa + weight*self.msa_col_attn(msa)
        msa = msa + weight*self.msa_transition(msa)

        return msa

    def _2d_update(self, msa, pair, state, xyz, drop_layer=False):
        weight = 0. if drop_layer else 1.

        msa_bias = self.outer_product(msa)
        pair = pair + weight*msa_bias
        str_bias = self.structure_bias(xyz, state)
        pair = pair + weight*self.drop_row(self.tri_mul_outgoing(pair)) 
        pair = pair + weight*self.drop_row(self.tri_mul_incoming(pair)) 
        pair = pair + weight*self.drop_row(self.pair_row_attn(pair, str_bias)) 
        pair = pair + weight*self.drop_col(self.pair_col_attn(pair, str_bias)) 
        pair = pair + weight*self.pair_transition(pair)
        return pair

    def _3d_update(self, msa, pair, state, xyz, is_atom, atom_frames, chirals, bond_feats, dist_matrix, idx, drop_layer=False):
        block_outputs = self.structure_attn(
            msa, pair, state, xyz.detach(), is_atom, atom_frames, chirals, idx, bond_feats, dist_matrix, drop_layer=drop_layer
        )
        return block_outputs["state"], block_outputs["xyz"], block_outputs["alpha"]

    def forward(self, latent_feats, use_checkpoint):
        msa, pair, state, xyz, is_atom, atom_frames, chirals, bond_feats, dist_matrix, idx = self._unpack_inputs(latent_feats)
        drop_layer = 0
        if use_checkpoint:
            msa  = checkpoint.checkpoint(create_custom_forward(self._1d_update, drop_layer=drop_layer), msa, pair, state, xyz, use_reentrant=True)
            pair = checkpoint.checkpoint(create_custom_forward(self._2d_update, drop_layer=drop_layer), msa, pair, state, xyz, use_reentrant=True)
            # 3D track cannot use re-entrant = False because of chiral features call to autograd
            #TODO: allow this to happen since new versions of Pytorch will be using reentrant=False
            state, xyz, alpha = checkpoint.checkpoint(
                create_custom_forward(self._3d_update, drop_layer=drop_layer),
                msa, pair, state, xyz, is_atom, atom_frames, chirals, bond_feats, dist_matrix, idx,
                use_reentrant=True
            )
        else:
            msa= self._1d_update(msa, pair, state, xyz, drop_layer=drop_layer)
            pair = self._2d_update(msa, pair, state, xyz, drop_layer=drop_layer)
            state, xyz, alpha = self._3d_update(msa, pair, state, xyz, is_atom, atom_frames, chirals,bond_feats, dist_matrix, idx, drop_layer=drop_layer)


        latent_feats = self._pack_outputs(msa, pair, state, xyz, alpha, latent_feats)
        return latent_feats


block_factory = {
    "RF2aa":                    partial(RF2_block, is_full=False),
    "RF2aa_full":               partial(RF2_block, is_full=True)
}
