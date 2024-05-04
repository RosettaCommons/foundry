import torch
import torch.nn as nn

from rf2aa.debug import debug_nans
from rf2aa.model.layers.SE3_network import FullyConnectedSE3, get_backbone_offset_vectors, get_chiral_vectors, SE3TransformerWrapper
from rf2aa.model.Track_module import Str2Str
from rf2aa.util_module import rbf, make_topk_graph, make_full_graph, init_lecun_normal
from rf2aa.util import is_atom,is_nucleic,is_protein
from rf2aa.chemical import ChemicalData as ChemData
from rf2aa.loss.loss import calc_chiral_grads
from rf2aa.model.generative_refinement import GenerativeRefinement

class LocalRefinementSE3(FullyConnectedSE3):

    def __init__(self, global_config, block_params):
        d_msa, d_pair = global_config.d_msa, global_config.d_pair
        d_rbf, num_layers, num_channels, num_degrees, n_heads, div, \
            l0_in_features, l0_out_features, l1_in_features, l1_out_features, \
                  num_edge_features, top_k, sc_pred_d_hidden, sc_pred_p_drop, compute_gradients = \
                block_params.d_rbf, block_params.n_se3_layers, block_params.n_se3_channels, \
                    block_params.n_se3_degrees, block_params.n_se3_head, block_params.n_div, \
                        block_params.l0_in_features, block_params.l0_out_features, \
                            block_params.l1_in_features, block_params.l1_out_features, \
                                block_params.n_se3_edge_features, block_params.top_k, \
                                    block_params.sc_pred_d_hidden, block_params.sc_pred_p_drop, \
                                    block_params.compute_gradients

        residual_state = block_params.residual_state

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
                                                 sc_pred_p_drop,
                                                 residual_state,
                                                 compute_gradients
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

    def construct_graph(self, xyz, edge):
        L = xyz.shape[1]
        idx = torch.arange(L, device=edge.device)[None]
        if self.top_k>0:
            G, edge_feats = make_topk_graph(xyz[:,:,1,:], edge, idx, top_k=self.top_k)
        else:
            G, edge_feats = make_full_graph(xyz[:,:,1,:], edge, idx) # top_k=1 -> fully connected

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
        idx, bond_feats, dist_matrix = latent_feats["idx"], latent_feats["bond_feats"], latent_feats["dist_matrix"]
        return msa, pair, state, xyz[..., :3, :], is_atom, atom_frames, chirals, idx, bond_feats, dist_matrix

    def forward(self, latent_feats):
        B, N, L = latent_feats["msa"].shape[:3]
        msa, pair, state, xyz, is_atom, atom_frames, chirals, idx, bond_feats, dist_matrix = self._unpack_inputs(latent_feats)
        xyzs = []
        alphas = []
        for i in range(self.num_iterations):
            output = self.se3(msa, pair, state, xyz.detach(), is_atom, atom_frames, chirals, idx, bond_feats, dist_matrix)
            xyzs.append(output["xyz"])
            alphas.append(output["alpha"])
            state, xyz = output["state"], output["xyz"]

        return {
            "xyzs": torch.stack(xyzs, dim=0),
            "state": state,
            "alphas": torch.stack(alphas, dim=0)
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
        idx, bond_feats, dist_matrix = latent_feats["idx"], latent_feats["bond_feats"], latent_feats["dist_matrix"]
        return msa, pair, state, xyz[..., :3, :], is_atom, atom_frames, chirals, idx, bond_feats, dist_matrix

    def forward(self, latent_feats):
        B, N, L = latent_feats["msa"].shape[:3]
        xyzs = []
        alphas = []

        msa, pair, state, xyz, is_atom, atom_frames, chirals, idx, bond_feats, dist_matrix = self._unpack_inputs(latent_feats)

        state = self.proj_state_in(state)

        for i in range(self.num_iterations):
            output = self.se3(msa, pair, state, xyz.detach(), is_atom, atom_frames, chirals, idx, bond_feats, dist_matrix)
            xyzs.append(output["xyz"])
            alphas.append(output["alpha"])    
            state, xyz = output["state"], output["xyz"]

        state = self.proj_state_out(state)
        latent_feats["state"] = state

        return {
            "xyzs": torch.stack(xyzs, dim=0),
            "state": state,
            "alphas": torch.stack(alphas, dim=0)
        }

class LegacyRefiner(nn.Module):
    def __init__(self, global_params, block_params):
        super(LegacyRefiner, self).__init__()

        self.str_refiner = Str2Str(
            d_msa=256,
            d_pair=192,
            d_state=64,
            d_rbf=64,
            nextra_l1=3,
            SE3_param={
                "num_layers": 2,
                "num_channels": 32,
                "num_degrees": 2,
                "l0_in_features": 64, 
                "l0_out_features": 64, 
                "l1_in_features": 3,
                "l1_out_features": 2,
                "num_edge_features": 64,
                "n_heads": 4,
                "div": 4
            }
        )
        self.refiner_topk = 64
    def _unpack_latents(self, latent_feats):
        msa, pair, xyz, state, idx, rotation_mask, bond_feats, dist_matrix, atom_frames, chirals = \
        latent_feats["msa"], latent_feats["pair"], latent_feats["xyz"], latent_feats["state"], \
        latent_feats["idx"], latent_feats["is_atom"], latent_feats["bond_feats"], latent_feats["dist_matrix"],  \
        latent_feats["atom_frames"], latent_feats["chirals"]
        return msa, pair, xyz, state, idx, rotation_mask, bond_feats, dist_matrix, atom_frames, chirals
    
    def forward(self, latent_feats):
        msa, pair, xyz, state, idx, rotation_mask, bond_feats, dist_matrix, atom_frames, chirals = \
        self._unpack_latents(latent_feats)
        is_motif = torch.zeros_like(state[..., 0][0], device=msa.device).bool()
        xyzs = []
        alphas = []
        for i in range(4):
            extra_l0 = None
            extra_l1 = []

            dchiraldxyz, = calc_chiral_grads(xyz.detach(),chirals)
            #extra_l1 = torch.cat((dljdxyz[0].detach(), dchiraldxyz[0].detach()), dim=1)
            extra_l1.append(dchiraldxyz[0].detach())
            extra_l1 = torch.cat(extra_l1, dim=1)

            xyz, state, alpha, quat = self.str_refiner(
                        msa.float(), pair.float(), xyz.detach().float(), state.float(), idx,
                        rotation_mask, bond_feats,  dist_matrix, atom_frames, 
                        is_motif, extra_l0, extra_l1.float(), top_k=self.refiner_topk, use_atom_frames=True
                    )
            xyzs.append(xyz)
            alphas.append(alpha)

        return {
            "xyzs": torch.stack(xyzs, dim=0),
            "state": state,
            "alphas": torch.stack(alphas, dim=0)
        }

refinement_factory ={
    "local": RecurrentLocalRefinement,
    "local_adaptor": RecurrentLocalRefinement_w_Adaptor,
    "legacy": LegacyRefiner,
    "generative": GenerativeRefinement
}
