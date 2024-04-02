import torch
import torch.nn as nn

from rf2aa.model.layers.Embeddings import MSA_emb, MSA_emb_nostate, \
            Extra_emb, Bond_emb, Templ_emb, Templ_emb_NoPtwise, recycling_factory
from rf2aa.chemical import ChemicalData as ChemData


class RF2_embedding(nn.Module):
    def __init__(self, global_params, block_params):
        super(RF2_embedding, self).__init__()
        d_msa, d_msa_full, d_pair, d_state = global_params["d_msa"], global_params["d_msa_full"], global_params["d_pair"], global_params["d_state"]
        self.latent_emb = MSA_emb(
            d_msa=d_msa, 
            d_pair=d_pair, 
            d_state=d_state, 
            p_drop=block_params.p_drop, 
            use_same_chain=block_params.use_same_chain
        )
        self.full_emb = Extra_emb(
            d_msa=d_msa_full, 
            d_init=ChemData().NAATOKENS - 1 + 4, #HACK: should define this freom the config (4: ins/del,nterm/cterm feats)
            p_drop=block_params.p_drop
        )
        self.bond_emb = Bond_emb(d_pair=d_pair, d_init=ChemData().NBTYPES)

        self.templ_emb = Templ_emb(d_pair=d_pair, 
                                   d_templ=block_params.d_templ, 
                                   d_state=d_state, 
                                   n_head=block_params.n_head_templ,
                                   d_hidden=block_params.d_hidden_templ, 
                                   p_drop=block_params.templ_p_drop,
                                   symmetrize_repeats=block_params.symmetrize_repeats, # repeat protein stuff 
                                   repeat_length=block_params.repeat_length, 
                                   symmsub_k=block_params.symmsub_k,
                                   sym_method=block_params.sym_method, 
                                   main_block=block_params.main_block, 
                                   copy_main_block=block_params.copy_main_block_template,
                                   additional_dt1d=block_params.additional_dt1d)

        ## Update inputs with outputs from previous forward pass
        self.recycle = recycling_factory[block_params.recycling_type](d_msa=d_msa, d_pair=d_pair, d_state=d_state)
        self.recycling_type = block_params.recycling_type
        assert self.recycling_type == "msa_pair", "no backward compatibility to recycling state"

    def _unpack_inputs(self, rf_inputs):
        msa_latent, msa_full, seq, idx, bond_feats, dist_matrix = \
            rf_inputs["msa_latent"], rf_inputs["msa_full"], rf_inputs["seq"], rf_inputs["idx"], rf_inputs["bond_feats"], \
            rf_inputs["dist_matrix"]
        ## recycling inputs
        msa_prev, pair_prev, state_prev, xyz, sctors, mask_recycle = rf_inputs["msa_prev"], rf_inputs["pair_prev"], None, \
            rf_inputs["xyz"], rf_inputs["sctors"], rf_inputs["mask_recycle"]
        return msa_latent, msa_full, seq, idx, bond_feats, dist_matrix, msa_prev, pair_prev, state_prev, xyz, sctors, mask_recycle
    
    def _add_templ_features(self, rf_inputs, pair, state):
        t1d, t2d, alpha_t, xyz_t, mask_t = rf_inputs["t1d"], rf_inputs["t2d"], \
                                            rf_inputs["alpha_t"], rf_inputs["xyz_t"], \
                                            rf_inputs["mask_t"]
        pair, state = self.templ_emb(t1d, t2d, alpha_t, xyz_t, mask_t, pair, state)
        return pair, state

    def forward(self, rf_inputs):
        msa_latent, msa_full, seq, idx, bond_feats, dist_matrix, msa_prev, pair_prev, state_prev, xyz, sctors, mask_recycle = \
            self._unpack_inputs(rf_inputs)    
        B, N, L = msa_latent.shape[:3]

        dtype = msa_latent.dtype

        msa_latent, pair, state = self.latent_emb(
            msa_latent, seq, idx, bond_feats,  dist_matrix
        )
        msa_full = self.full_emb(msa_full, seq, idx)
        pair = pair + self.bond_emb(bond_feats)

        msa_latent, pair = msa_latent.to(dtype), pair.to(dtype)
        msa_full = msa_full.to(dtype)
        if state is not None: 
            state = state.to(dtype)

        if msa_prev is None:
            msa_prev = torch.zeros_like(msa_latent[:,0])
        if pair_prev is None:
            pair_prev = torch.zeros_like(pair)
        if state_prev is None or self.recycling_type == "msa_pair": #explicitly remove state features if only recycling msa and pair
            state_prev = torch.zeros_like(msa_latent[:, 0])

        msa_recycle, pair_recycle, state_recycle = self.recycle(msa_prev, pair_prev, xyz, state_prev, sctors, mask_recycle)
        
        msa_recycle, pair_recycle = msa_recycle.to(dtype), pair_recycle.to(dtype)

        msa_latent[:,0] = msa_latent[:,0] + msa_recycle.reshape(B,L,-1)
        pair = pair + pair_recycle
        # No support for recycling state
        #state = state + state_recycle # if state is not recycled these will be zeros
        # add template embedding
        pair, state = self._add_templ_features(rf_inputs, pair, state)
        return {
            "msa": msa_latent,
            "msa_full": msa_full,
            "pair": pair,
            "state": state
        }

class RF2_embedding_no_ptwise(RF2_embedding):
    
    def __init__(self, global_params, block_params):
        super(RF2_embedding_no_ptwise, self).__init__(global_params, block_params)
        d_msa, d_msa_full, d_pair, d_state = global_params["d_msa"], global_params["d_msa_full"], global_params["d_pair"], global_params["d_state"]
        self.templ_emb = Templ_emb_NoPtwise(d_pair=d_pair, 
                                   d_templ=block_params.d_templ, 
                                   d_state=d_state, 
                                   n_head=block_params.n_head_templ,
                                   d_hidden=block_params.d_hidden_templ, 
                                   p_drop=block_params.templ_p_drop,
                                   symmetrize_repeats=block_params.symmetrize_repeats, # repeat protein stuff 
                                   repeat_length=block_params.repeat_length, 
                                   symmsub_k=block_params.symmsub_k,
                                   sym_method=block_params.sym_method, 
                                   main_block=block_params.main_block, 
                                   copy_main_block=block_params.copy_main_block_template,
                                   additional_dt1d=block_params.additional_dt1d)


class RF2_embedding_nostate(RF2_embedding):

    def __init__(self, global_params, block_params):
        super(RF2_embedding_nostate, self).__init__(global_params, block_params)
        d_msa, d_msa_full, d_pair, d_state = global_params["d_msa"], global_params["d_msa_full"], global_params["d_pair"], global_params["d_state"]
        self.latent_emb = MSA_emb_nostate(
            d_msa=d_msa, 
            d_pair=d_pair, 
            d_state=d_state, 
            p_drop=block_params.p_drop, 
            use_same_chain=block_params.use_same_chain
        )
        self.templ_emb = None

    def _add_templ_features(self, rf_inputs, pair, state):
        #identity
        return pair, state

# Null module for overloading existing modules with a no-op
class Noop(nn.Module):
    def forward(*args, **kwargs):
        return torch.tensor([0.])

class RF2_embedding_no_ptwise_no_full(RF2_embedding_no_ptwise):
    def __init__(self, global_params, block_params):
        super(RF2_embedding_no_ptwise, self).__init__(global_params, block_params)
        self.full_emb = Noop()


embedding_factory = {
    "rf2aa": RF2_embedding,
    "rf2aa_noptwise": RF2_embedding_no_ptwise,
    "rf2aa_noptwise_no_full": RF2_embedding_no_ptwise_no_full,
    "rf2aa_nostate": RF2_embedding_nostate
}
