import torch
import torch.nn as nn

from rf2aa.chemical import NAATOKENS

class DistanceNetwork(nn.Module):
    def __init__(self, n_feat, p_drop=0.1):
        super(DistanceNetwork, self).__init__()
        #
        self.proj_symm = nn.Linear(n_feat, 61+37) # must match bin counts defined in kinematics.py
        self.proj_asymm = nn.Linear(n_feat, 37+19)
    
        self.reset_parameter()
    
    def reset_parameter(self):
        # initialize linear layer for final logit prediction
        nn.init.zeros_(self.proj_symm.weight)
        nn.init.zeros_(self.proj_asymm.weight)
        nn.init.zeros_(self.proj_symm.bias)
        nn.init.zeros_(self.proj_asymm.bias)

    def forward(self, x):
        # input: pair info (B, L, L, C)

        # predict theta, phi (non-symmetric)
        logits_asymm = self.proj_asymm(x)
        logits_theta = logits_asymm[:,:,:,:37].permute(0,3,1,2)
        logits_phi = logits_asymm[:,:,:,37:].permute(0,3,1,2)

        # predict dist, omega
        logits_symm = self.proj_symm(x)
        logits_symm = logits_symm + logits_symm.permute(0,2,1,3)
        logits_dist = logits_symm[:,:,:,:61].permute(0,3,1,2)
        logits_omega = logits_symm[:,:,:,37:].permute(0,3,1,2)

        return logits_dist, logits_omega, logits_theta, logits_phi

class MaskedTokenNetwork(nn.Module):
    def __init__(self, n_feat, p_drop=0.1):
        super(MaskedTokenNetwork, self).__init__()

        #fd note this predicts probability for the mask token (which is never in ground truth)
        #   it should be ok though(?)
        self.proj = nn.Linear(n_feat, NAATOKENS)
        
        self.reset_parameter()
    
    def reset_parameter(self):
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, N, L = x.shape[:3]
        logits = self.proj(x).permute(0,3,1,2).reshape(B, -1, N*L)

        return logits

class LDDTNetwork(nn.Module):
    def __init__(self, n_feat, n_bin_lddt=50):
        super(LDDTNetwork, self).__init__()
        self.proj = nn.Linear(n_feat, n_bin_lddt)

        self.reset_parameter()

    def reset_parameter(self):
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        logits = self.proj(x) # (B, L, 50)

        return logits.permute(0,2,1)

class PAENetwork(nn.Module):
    def __init__(self, n_feat, n_bin_pae=64):
        super(PAENetwork, self).__init__()
        self.proj = nn.Linear(n_feat, n_bin_pae)
        self.reset_parameter()
    def reset_parameter(self):
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        logits = self.proj(x) # (B, L, L, 64)

        return logits.permute(0,3,1,2)


class BinderNetwork(nn.Module):
    """BinderNetwork This class is for predicting binder non/binder
    from the input. It's basically just a logistic regression head
    that learns to downsample from the pair and state features.
    """
    def __init__(self, d_pair: int = 128, d_state: int = 64):
        super(BinderNetwork, self).__init__()
        self.downsample_pair = torch.nn.Linear(d_pair, 1)
        self.downsample_state = torch.nn.Linear(d_state, 1)
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.zeros_(self.downsample_pair.weight)
        nn.init.zeros_(self.downsample_pair.bias)
        nn.init.zeros_(self.downsample_state.weight)
        nn.init.zeros_(self.downsample_state.bias)

    def forward(self, pair_features, same_chain, state):
        logits_pairwise = self.downsample_pair(pair_features)

        # Note: same_chain is an L x L matrix that indicates which residues are
        # part of the same chain. The following line computes only the features
        # that are between the protein and the ligand. It's not clear whether
        # we should do this, or use all of the pairwise features.
        interchain_mask = same_chain == 0
        if torch.sum(interchain_mask) == 0:
            interchain_mask = same_chain == 1
        logits_between_ligand_and_protein = logits_pairwise[interchain_mask]
        logits_between_ligand_and_protein = torch.reshape(logits_between_ligand_and_protein, (pair_features.shape[0], -1))

        # Note: it is still not clear to me which should come first:
        # taking the mean over length, and downsampling via learned weights.
        # My intuition is that taking the mean first destroys a lot of useful information
        # and it would be more useful to apply the linear head to every position, but
        # it's hard to say without trying both.
        mean_logit_over_length = torch.mean(logits_between_ligand_and_protein, axis=-1)

        downsampled_state = self.downsample_state(state)
        mean_state_over_length = torch.mean(downsampled_state, dim=(1, 2))

        # The result above should be size (batch_size, )
        # and contain a single prediction per batch.
        squashed_logits = torch.sigmoid(mean_logit_over_length + mean_state_over_length)

        return squashed_logits
