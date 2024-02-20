import torch
import torch.nn as nn

from rf2aa.debug import debug_nans
from rf2aa.model.layers.SE3_network import FullyConnectedSE3, get_backbone_offset_vectors, get_chiral_vectors
from rf2aa.model.Track_module import SCPred
from rf2aa.util_module import rbf, make_topk_graph, init_lecun_normal
from rf2aa.chemical import ChemicalData as ChemData

class LocalRefinementSE3(FullyConnectedSE3):

    def __init__(self, global_config, block_params):
        d_msa, d_pair = global_config.d_msa, global_config.d_pair
        d_rbf, num_layers, num_channels, num_degrees, n_heads, div, \
            l0_in_features, l0_out_features, l1_in_features, l1_out_features, \
                  num_edge_features, top_k, sc_pred_d_hidden, sc_pred_p_drop = \
                block_params.d_rbf, block_params.n_se3_layers, block_params.n_se3_channels, \
                    block_params.n_se3_degrees, block_params.n_se3_head, block_params.n_div, \
                        block_params.l0_in_features, block_params.l0_out_features, \
                            block_params.l1_in_features, block_params.l1_out_features, \
                                block_params.n_se3_edge_features, block_params.top_k, \
                                    block_params.sc_pred_d_hidden, block_params.sc_pred_p_drop

        super(LocalRefinementSE3, self).__init__(d_msa, 
                                                 d_pair, 
                                                 d_rbf, 
                                                 num_layers, 
                                                 num_channels, 
                                                 num_degrees, 
                                                 n_heads, 
                                                 div, 
                                                 l0_in_features, 
                                                 l0_out_features, 
                                                 l1_in_features, 
                                                 l1_out_features, 
                                                 num_edge_features,
                                                 sc_pred_d_hidden,
                                                 sc_pred_p_drop
                                                 )
        self.top_k = top_k
        self.reset_parameter() 

    def reset_parameter(self):
        # initialize weights to normal distribution
        self.embed_node = init_lecun_normal(self.embed_node)
        self.embed_edge = init_lecun_normal(self.embed_edge)

        # initialize bias to zeros
        nn.init.zeros_(self.embed_node.bias)
        nn.init.zeros_(self.embed_edge.bias)

        nn.init.ones_(self.norm_msa.weight)
        nn.init.ones_(self.norm_pair.weight)

    def construct_graph(self, xyz, edge):
        L = xyz.shape[1]
        idx = torch.arange(L, device=edge.device)[None]
        G, edge_feats = make_topk_graph(xyz[:,:,1,:], edge, idx, top_k=self.top_k)
        return  G, edge_feats

class RecurrentLocalRefinement(nn.Module):
    
    def __init__(self, global_config, block_params):
        super(RecurrentLocalRefinement, self).__init__()
        self.num_iterations = block_params.num_iterations

        self.se3 = LocalRefinementSE3(global_config, block_params)
    
    def _unpack_inputs(self, latent_feats):
        msa, pair, state, xyz, is_atom, atom_frames, chirals = \
            latent_feats["msa"], latent_feats["pair"], \
            latent_feats["state"], latent_feats["xyz"], latent_feats["is_atom"], \
                latent_feats["atom_frames"], latent_feats["chirals"]
        return msa, pair, state, xyz, is_atom, atom_frames, chirals

    def forward(self, latent_feats):
        B, N, L = latent_feats["msa"].shape[:3]
        xyzs = torch.full((self.num_iterations, L, 3, 3 ), torch.nan, device=latent_feats["msa"].device)
        alphas = torch.full((self.num_iterations, L, ChemData().NTOTALDOFS, 2), torch.nan, device=latent_feats["msa"].device) 

        msa, pair, state, xyz, is_atom, atom_frames, chirals = self._unpack_inputs(latent_feats)

        for i in range(self.num_iterations):
            output = self.se3(msa, pair, state, xyz, is_atom, atom_frames, chirals)
            xyzs[i] = output["xyz"]
            alphas[i] = output["alpha"]
            state, xyz = output["state"], output["xyz"]
        
        return {
            "xyzs": xyzs,
            "alphas": alphas, 
        }

class RecurrentLocalRefinement_w_Adaptor(nn.Module):
    def __init__(self, global_config, block_params):
        super(RecurrentLocalRefinement_w_Adaptor, self).__init__()
        self.num_iterations = block_params.num_iterations

        self.proj_state_in = nn.Linear(block_params.adaptor_features, block_params.l0_in_features)
        self.proj_state_out = nn.Linear(block_params.l0_in_features, block_params.adaptor_features)

        self.se3 = LocalRefinementSE3(global_config, block_params)
    
    def _unpack_inputs(self, latent_feats):
        msa, pair, state, xyz, is_atom, atom_frames, chirals = \
            latent_feats["msa"], latent_feats["pair"], \
            latent_feats["state"], latent_feats["xyz"], latent_feats["is_atom"], \
                latent_feats["atom_frames"], latent_feats["chirals"]
        return msa, pair, state, xyz, is_atom, atom_frames, chirals

    def forward(self, latent_feats):
        B, N, L = latent_feats["msa"].shape[:3]
        xyzs = torch.full((self.num_iterations, L, 3, 3 ), torch.nan, device=latent_feats["msa"].device)
        alphas = torch.full((self.num_iterations, L, ChemData().NTOTALDOFS, 2), torch.nan, device=latent_feats["msa"].device) 

        msa, pair, state, xyz, is_atom, atom_frames, chirals = self._unpack_inputs(latent_feats)

        state = self.proj_state_in(state)
        for i in range(self.num_iterations):
            output = self.se3(msa, pair, state, xyz, is_atom, atom_frames, chirals)
            xyzs[i] = output["xyz"]
            alphas[i] = output["alpha"]
            state, xyz = output["state"], output["xyz"]

        state = self.proj_state_out(state)
        latent_feats["state"] = state

        return {
            "xyzs": xyzs,
            "alphas": alphas, 
        }


refinement_factory ={
    "local": RecurrentLocalRefinement,
    "local_adaptor": RecurrentLocalRefinement_w_Adaptor
}