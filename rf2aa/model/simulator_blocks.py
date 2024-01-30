import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from functools import partial

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
                                                d_state,
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
        return msa, pair, state, xyz[..., :3, :], is_atom, atom_frames, chirals

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

    def _1d_update(self, msa, pair, state, xyz):
        pair = self.norm_pair_bias(pair)
        pair = pair + self.msa_str_bias(xyz)

        state = self.norm_state_bias(state)
        state_update = self.proj_state_bias(state)

        msa = msa.type_as(state_update)
        msa = msa.index_add(1, torch.tensor([0,], device=state_update.device), state_update[None])
        msa = msa + self.drop_row(self.msa_row_attn(msa, pair))
        msa = msa + self.msa_col_attn(msa)
        msa = msa + self.msa_transition(msa)
        return msa

    def _2d_update(self, msa, pair, state, xyz):
        msa_bias = self.outer_product(msa)
        pair = pair + msa_bias
        str_bias = self.structure_bias(xyz, state)
        pair = pair + self.drop_row(self.tri_mul_outgoing(pair)) 
        pair = pair + self.drop_row(self.tri_mul_incoming(pair)) 
        pair = pair + self.drop_row(self.pair_row_attn(pair, str_bias)) 
        pair = pair + self.drop_col(self.pair_col_attn(pair, str_bias)) 
        pair = pair + self.pair_transition(pair)
        return pair

    def _3d_update(self, msa, pair, state, xyz, is_atom, atom_frames, chirals):
        block_outputs = self.structure_attn(msa, pair, state, xyz, is_atom, atom_frames, chirals)
        return block_outputs["state"], block_outputs["xyz"], block_outputs["alpha"]

    def forward(self, latent_feats, use_checkpoint):
        msa, pair, state, xyz, is_atom, atom_frames, chirals = self._unpack_inputs(latent_feats)
        if use_checkpoint:
            msa  = checkpoint.checkpoint(create_custom_forward(self._1d_update), msa, pair, state, xyz)
            pair = checkpoint.checkpoint(create_custom_forward(self._2d_update), msa, pair, state, xyz)
            # 3D track cannot use re-entrant = False because of chiral features call to autograd
            #TODO: allow this to happen since new versions of Pytorch will be using reentrant=False
            state, xyz, alpha = checkpoint.checkpoint(create_custom_forward(self._3d_update), \
                                                    msa, pair, state, xyz, is_atom, atom_frames, chirals)
        else:
            msa= self._1d_update(msa, pair, state, xyz)
            pair = self._2d_update(msa, pair, state, xyz)
            state, xyz, alpha = self._3d_update(msa, pair, state, xyz, is_atom, atom_frames, chirals)
        latent_feats = self._pack_outputs(msa, pair, state, xyz, alpha, latent_feats)
        return latent_feats


class RF2_withgradients(nn.Module):
    """
    this is an updated version of the RF2 block, without computing rotations
    to allow gradients to flow through all blocks
    """
    def __init__(self, global_config=None, block_params=None, is_full=False, **kwargs
                ) -> None:
        super(RF2_withgradients, self).__init__()
        d_msa, d_msa_full, d_pair = global_config.d_msa, global_config.d_msa_full, global_config.d_pair
        self.is_full = is_full
        if self.is_full:
            d_msa = d_msa_full

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
                d_msa=d_msa, n_head=block_params.n_msa_head, d_hidden=block_params.n_msa_channels
            )
        self.drop_row = Dropout(broadcast_dim=1, p_drop=block_params.p_drop_row)
        self.drop_col = Dropout(broadcast_dim=2, p_drop=block_params.p_drop_col)
        self.msa_transition = FeedForwardLayer(d_msa, r_ff=4)
        self.compute_structure_bias = structure_bias_factory[block_params.structure_bias_type](
            d_rbf=block_params.structure_bias_channels,
            d_pair=d_pair
            )
        self.pair_row_attn = BiasedAxialAttention(
            d_pair=d_pair, 
            d_bias=d_pair, 
            n_head=block_params.n_pair_head,  
            d_hidden=block_params.n_pair_channels,
            is_row=True
        )
        self.pair_col_attn = BiasedAxialAttention(
            d_pair=d_pair, 
            d_bias=d_pair, 
            n_head=block_params.n_pair_head,  
            d_hidden=block_params.n_pair_channels,
            is_row=False
        )
        self.tri_mult_incoming = TriangleMultiplication(
            d_pair=d_pair, d_hidden=block_params.n_pair_channels, outgoing=False
        )
        self.tri_mult_outgoing = TriangleMultiplication(
            d_pair=d_pair, d_hidden=block_params.n_pair_channels, outgoing=True
        )
        self.pair_transition = FeedForwardLayer(
            d_pair, r_ff=4
        )
        self.outer_product = OuterProductMean(d_msa, d_pair, d_hidden=block_params.outer_product_channels)
        
        self.structure_attn = FullyConnectedSE3_noR(
            d_msa=d_msa,
            d_pair=d_pair,
            d_rbf=block_params.structure_bias_channels,
            num_layers=block_params.n_se3_layers,
            num_channels=block_params.n_se3_channels,
            num_degrees=block_params.n_se3_degrees,
            n_heads=block_params.n_se3_head,
            div=block_params.n_div,
            l0_in_features=block_params.l0_in_features,
            l0_out_features=block_params.l0_out_features,
            l1_in_features=block_params.l1_in_features,
            l1_out_features=block_params.l1_out_features,
            num_edge_features=block_params.n_se3_edge_features
            )
        self.structure_transition = FeedForwardLayer(
            block_params.l0_out_features, r_ff=4
        )
        self.proj_state = nn.Linear(block_params.l0_out_features, d_msa)
        self.reset_parameter()
    
    def reset_parameter(self):
        pass

    def _unpack_inputs(self, latent_feats):
        pair, xyz, is_atom, atom_frames, chirals = \
            latent_feats["pair"], \
            latent_feats["xyz"], latent_feats["is_atom"], \
                latent_feats["atom_frames"], latent_feats["chirals"]
        if self.is_full:
            msa = latent_feats["msa_full"]
        else:
            msa = latent_feats["msa"]
        return msa, pair, xyz[..., :3, :], is_atom, atom_frames, chirals

    def _pack_outputs(self, msa, pair, state, xyz, latent_feats):
        if self.is_full:
            latent_feats["msa_full"] = msa
        else:
            latent_feats["msa"] = msa
        latent_feats["pair"] = pair
        latent_feats["state"] = state
        latent_feats["xyz"] = xyz
        return latent_feats

    def _1d_update(self, msa, pair):
        msa = msa + self.drop_row(self.msa_row_attn(msa, pair))
        msa = msa + self.drop_col(self.msa_col_attn(msa))
        msa = msa + self.msa_transition(msa)
        msa_bias = self.outer_product(msa)

        pair = pair + msa_bias
        return msa, pair

    def _2d_update(self, pair, xyz):
        # break 3d symmetries with bias from coordinates
        structure_bias = self.compute_structure_bias(xyz)
        pair = pair + self.drop_row(self.pair_row_attn(pair, structure_bias))
        pair = pair + self.drop_col(self.pair_col_attn(pair, structure_bias))

        # provide triangle inductive bias 
        pair = pair + self.drop_row(self.tri_mult_outgoing(pair))
        pair = pair + self.drop_row(self.tri_mult_incoming(pair))
        pair = pair + self.pair_transition(pair)
        return pair

    def _3d_update(self, msa, pair, state, xyz, is_atom, atom_frames, chirals):
        # apply structure attention and update seq features
        state, xyz = self.structure_attn(msa, pair, state, xyz, is_atom, atom_frames, chirals)
        state = state + self.structure_transition(state)

        # state features bias the msa first row features
        state_update = self.proj_state(state)
        msa = msa.type_as(state_update)
        msa = msa.index_add(1, torch.tensor([0,], device=state_update.device), state_update.unsqueeze(1))
        return msa, state, xyz
    
    def forward(self, latent_feats, use_checkpoint):
        msa, pair, xyz, is_atom, atom_frames, chirals = self._unpack_inputs(latent_feats)
        state = None
        if use_checkpoint:
            msa, pair = checkpoint.checkpoint(create_custom_forward(self._1d_update), msa, pair)
            pair = checkpoint.checkpoint(create_custom_forward(self._2d_update), pair, xyz)
            # 3D track cannot use re-entrant = False because of chiral features call to autograd
            #TODO: allow this to happen since new versions of Pytorch will be using reentrant=False
            msa, state, xyz = checkpoint.checkpoint(create_custom_forward(self._3d_update), \
                                                    msa, pair, state, xyz, is_atom, atom_frames, chirals)
        else:
            msa, pair = self._1d_update(msa, pair)
            pair = self._2d_update(pair, xyz)
            msa, state, xyz = self._3d_update(msa, pair, state, xyz, is_atom, atom_frames, chirals)
        latent_feats = self._pack_outputs(msa, pair, state, xyz, latent_feats)
        return latent_feats


        
block_factory = {
    "RF2_withgradients":        partial(RF2_withgradients, is_full=False), 
    "RF2_withgradients_full":  partial(RF2_withgradients, is_full=True),
    "RF2aa":                    partial(RF2_block, is_full=False),
    "RF2aa_full":               partial(RF2_block, is_full=True)
}
