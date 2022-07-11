import torch
import torch.nn as nn
from Embeddings import MSA_emb, Extra_emb, Bond_emb, Templ_emb, Recycling
from Track_module import IterativeSimulator
from AuxiliaryPredictor import DistanceNetwork, MaskedTokenNetwork, LDDTNetwork

from chemical import INIT_CRDS,NAATOKENS, NBTYPES

class RoseTTAFoldModule(nn.Module):
    def __init__(
        self, n_extra_block=4, n_main_block=8, n_ref_block=4, n_finetune_block=0,\
        d_msa=256, d_msa_full=64, d_pair=128, d_templ=64,
        n_head_msa=8, n_head_pair=4, n_head_templ=4,
        d_hidden=32, d_hidden_templ=64,
        rbf_sigma=1.0, p_drop=0.15,
        SE3_param={}, SE3_ref_param={},
        atom_type_index=None, aamask=None, ljlk_parameters=None, lj_correction_parameters=None, 
        cb_len=None, cb_ang=None, cb_tor=None,
        num_bonds=None, lj_lin=0.6
    ):
        super(RoseTTAFoldModule, self).__init__()
        #
        # Input Embeddings
        d_state = SE3_param['l0_out_features']
        self.latent_emb = MSA_emb(d_msa=d_msa, d_pair=d_pair,  d_state=d_state, p_drop=p_drop)
        self.full_emb = Extra_emb(d_msa=d_msa_full, d_init=NAATOKENS-1+4, p_drop=p_drop)
        self.bond_emb = Bond_emb(d_pair=d_pair, d_init=NBTYPES)
        self.templ_emb = Templ_emb(d_pair=d_pair, d_templ=d_templ, d_state=d_state, n_head=n_head_templ,
                                   d_hidden=d_hidden_templ, p_drop=0.25)

        # Update inputs with outputs from previous round
        self.recycle = Recycling(d_msa=d_msa, d_pair=d_pair, d_state=d_state, rbf_sigma=rbf_sigma)
        #
        self.simulator = IterativeSimulator(
            n_extra_block=n_extra_block,
            n_main_block=n_main_block,
            n_ref_block=n_ref_block,
            n_finetune_block=n_finetune_block,
            d_msa=d_msa, 
            d_msa_full=d_msa_full,
            d_pair=d_pair, 
            d_hidden=d_hidden,
            n_head_msa=n_head_msa,
            n_head_pair=n_head_pair,
            SE3_param=SE3_param,
            SE3_ref_param=SE3_ref_param,
            rbf_sigma=rbf_sigma,
            p_drop=p_drop,
            atom_type_index=atom_type_index, # change if encoding elements instead of atomtype
            aamask=aamask, 
            ljlk_parameters=ljlk_parameters,
            lj_correction_parameters=lj_correction_parameters, 
            num_bonds=num_bonds,
            cb_len=cb_len,
            cb_ang=cb_ang,
            cb_tor=cb_tor,
            lj_lin=lj_lin
        )

        ##
        self.c6d_pred = DistanceNetwork(d_pair, p_drop=p_drop)
        self.aa_pred = MaskedTokenNetwork(d_msa, p_drop=p_drop)
        self.lddt_pred = LDDTNetwork(d_state)

    def forward(
        self, msa_latent, msa_full, seq, seq_unmasked, xyz, sctors, idx, bond_feats,
        t1d=None, t2d=None, xyz_t=None, alpha_t=None,
        msa_prev=None, pair_prev=None, state_prev=None, 
        return_raw=False, return_full=False,
        use_checkpoint=False
    ):
        B, N, L = msa_latent.shape[:3]
        # Get embeddings
        msa_latent, pair, state = self.latent_emb(msa_latent, seq, idx)
        msa_full = self.full_emb(msa_full, seq, idx)
        pair = pair + self.bond_emb(bond_feats)
        #
        # Do recycling
        if msa_prev == None:
            msa_prev = torch.zeros_like(msa_latent[:,0])
            pair_prev = torch.zeros_like(pair)
            state_prev = torch.zeros_like(state)

        msa_recycle, pair_recycle, state_recycle = self.recycle(msa_prev, pair_prev, xyz, state_prev, sctors)
        msa_latent[:,0] = msa_latent[:,0] + msa_recycle.reshape(B,L,-1)
        pair = pair + pair_recycle
        state = state + state_recycle

        # add template embedding
        pair, state = self.templ_emb(t1d, t2d, alpha_t, xyz_t, pair, state, use_checkpoint=use_checkpoint)

        # Predict coordinates from given inputs
        msa, pair, xyz, alpha_s, xyz_allatom, state = self.simulator(
            seq_unmasked, msa_latent, msa_full, pair, xyz[:,:,:3], state, idx, use_checkpoint=use_checkpoint)

        if return_raw:
            # get last structure
            xyz_last = xyz_allatom[-1].unsqueeze(0)
            return msa[:,0], pair, xyz_last, state, alpha_s[-1]

        # predict masked amino acids
        logits_aa = self.aa_pred(msa)

        # predict distogram & orientograms
        logits = self.c6d_pred(pair)

        # Predict LDDT
        lddt = self.lddt_pred(state)

        return logits, logits_aa, xyz, alpha_s, xyz_allatom, lddt, msa[:,0], pair, state

