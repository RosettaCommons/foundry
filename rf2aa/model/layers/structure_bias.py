from rf2aa.util_module import rbf, init_lecun_normal

import torch

import torch.nn as nn
from opt_einsum import contract as einsum

class StructureBias(torch.nn.Module):

    def __init__(self, d_rbf, d_pair) -> None:
        super(StructureBias, self).__init__()
        self.proj_rbf = nn.Linear(d_rbf, d_pair)

        self.reset_parameter()
    
    def reset_parameter(self):
        self.proj_rbf = init_lecun_normal(self.proj_rbf)
        nn.init.zeros_(self.proj_rbf.bias)

    def forward(self, xyz):
        cas = xyz[:,:,1].contiguous()
        rbf_feat = rbf(torch.cdist(cas, cas))
        bias = self.proj_rbf(rbf_feat)
        return bias


class GatedStructureBias(torch.nn.Module):

    def __init__(self, d_rbf, d_state, d_pair, d_hidden_gate) -> None:
        super(GatedStructureBias, self).__init__()
        self.norm_state = nn.LayerNorm(d_state)
        self.proj_rbf = nn.Linear(d_rbf, d_pair)
        self.proj_left = nn.Linear(d_state, d_hidden_gate)
        self.proj_right = nn.Linear(d_state, d_hidden_gate)
        self.to_gate = nn.Linear(d_hidden_gate*d_hidden_gate, d_pair)

        self.reset_parameter()
    
    def reset_parameter(self):
        pass

    def forward(self, xyz, state):
        B, L = xyz.shape[:2]
        cas = xyz[:,:,1].contiguous()

        rbf_feat = rbf(torch.cdist(cas, cas))
        rbf_feat = self.proj_rbf(rbf_feat)

        state = self.norm_state(state)
        left = self.proj_left(state)
        right = self.proj_right(state)
        gate = einsum('bli,bmj->blmij', left, right).reshape(B,L,L,-1)
        gate = torch.sigmoid(self.to_gate(gate))
        rbf_feat = gate*rbf_feat

        return rbf_feat


structure_bias_factory = {
        "ungated": StructureBias,
        "gated": GatedStructureBias
    }