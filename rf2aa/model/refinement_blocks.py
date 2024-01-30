import torch
import torch.nn as nn

from rf2aa.debug import debug_nans
from rf2aa.model.layers.SE3_network import FullyConnectedSE3, get_backbone_offset_vectors, get_chiral_vectors
from rf2aa.model.Track_module import SCPred
from rf2aa.util import NTOTAL, NTOTALDOFS
from rf2aa.util_module import rbf, make_topk_graph, init_lecun_normal

class LocalRefinementSE3(FullyConnectedSE3):

    def __init__(self, global_config, block_params):
        d_msa, d_pair, d_state = global_config.d_msa, global_config.d_pair, global_config.d_state
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
                                                 d_state,
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

   #def reset_parameter(self):
        ## initialize weights to normal distribution
        #self.embed_node = init_lecun_normal(self.embed_node)
        #self.embed_edge = init_lecun_normal(self.embed_edge)

        ## initialize bias to zeros
        #nn.init.zeros_(self.embed_node.bias)
        #nn.init.zeros_(self.embed_edge.bias)

        #nn.init.ones_(self.norm_msa.weight)
        #nn.init.ones_(self.norm_pair.weight)


    #def forward(self, msa, pair, state, xyz, is_atom, atom_frames, chirals):
        #B, N, L = msa.shape[:3]
        #seq = self.norm_msa(msa[:, 0])
        #pair = self.norm_pair(pair)

        #node = self.embed_node(torch.cat((seq, state), dim=-1))
        #node = node + self.ff_node(node)
        #node = self.norm_node(node)

        ##NOTE: Ablating providing the positional encoding at every step
        ## we introduced this and I do not think it is in RF2
        ##neighbor = get_seqsep_protein_sm(idx, bond_feats, dist_matrix, rotation_mask)
        #cas = xyz[:,:,1].contiguous()
        #rbf_feat = rbf(torch.cdist(cas, cas))
        #edge = torch.cat((pair, rbf_feat), dim=-1)
        #edge = self.embed_edge(edge)
        #edge = edge + self.ff_edge(edge)
        #edge = self.norm_edge(edge)

        #idx = torch.arange(L, device=edge.device)[None]
        #G, edge_feats = make_topk_graph(xyz[:,:,1,:], edge, idx, top_k=self.top_k)

        ##TODO: get extra l1 feats automatically and populate the extra l1 dimension
        #l1_feats = torch.cat(
            #[
                #get_backbone_offset_vectors(xyz, is_atom, atom_frames),
                #get_chiral_vectors(xyz, chirals)
            #], dim=1
        #)

        #shift = self.se3(G, node.reshape(B*L, -1, 1), l1_feats, edge_feats)
        
        #state = shift["0"].reshape(B, L, -1)
        #offset = shift["1"].reshape(B, L, 2, 3)
        #T = offset[:,:,0,:] / 10
        #R = offset[:,:,1,:] / 100.0

        #Qnorm = torch.sqrt( 1 + torch.sum(R*R, dim=-1) )
        #qA, qB, qC, qD = 1/Qnorm, R[:,:,0]/Qnorm, R[:,:,1]/Qnorm, R[:,:,2]/Qnorm

        #v = xyz - xyz[:,:,1:2,:]
        #Rout = torch.zeros((B,L,3,3), device=xyz.device)
        #Rout[:,:,0,0] = qA*qA+qB*qB-qC*qC-qD*qD
        #Rout[:,:,0,1] = 2*qB*qC - 2*qA*qD
        #Rout[:,:,0,2] = 2*qB*qD + 2*qA*qC
        #Rout[:,:,1,0] = 2*qB*qC + 2*qA*qD
        #Rout[:,:,1,1] = qA*qA-qB*qB+qC*qC-qD*qD
        #Rout[:,:,1,2] = 2*qC*qD - 2*qA*qB
        #Rout[:,:,2,0] = 2*qB*qD - 2*qA*qC
        #Rout[:,:,2,1] = 2*qC*qD + 2*qA*qB
        #Rout[:,:,2,2] = qA*qA-qB*qB-qC*qC+qD*qD
        #I = torch.eye(3, device=Rout.device).expand(B,L,3,3)
        #Rout = torch.where(is_atom.reshape(B, L, 1,1), I, Rout)
        #xyz = torch.einsum('blij,blaj->blai', Rout,v)+xyz[:,:,1:2,:]+T[:,:,None,:]

        #alpha = self.sc_predictor(msa[:,0], state)
        #return {
            #"state": state,
            #"xyz": xyz,
            #"alpha": alpha
        #}

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
        alphas = torch.full((self.num_iterations, L, NTOTALDOFS, 2), torch.nan, device=latent_feats["msa"].device) 

        msa, pair, state, xyz, is_atom, atom_frames, chirals = self._unpack_inputs(latent_feats)

        for i in range(self.num_iterations):
            output = self.se3(msa, pair, state, xyz, is_atom, atom_frames, chirals)
            xyzs[i] = output["xyz"]
            alphas[i] = output["alpha"]
            latent_feats["state"] = output["state"]
        
        return {
            "xyzs": xyzs,
            "alphas": alphas, 
        }

refinement_factory ={
    "local": RecurrentLocalRefinement
}