import torch
import torch.nn as nn


class AF3_with_rollout(nn.Module):
    """ Implements rollout on each training step """
    def __init__(self, model, confidence, sampler):
        super(AF3_with_rollout, self).__init__()
        self.model = model
        self.confidence = confidence
        self.sampler = sampler
        self.num_timesteps = 20

    def forward(self, input, n_cycle, no_sync):
        # first do forward pass
        trunk_output = self.model(input, n_cycle, no_sync)
        # save embeddings

        # do rollout conditioned on embeddings
        # with nograd? 
        with torch.no_grad():
            noise_schedule = self.sampler.construct_noise_schedule(self.num_timesteps, 0, 1).to(input["f"]["msa"].device)
            diffusion_output = self.sampler.sample_diffusion(
                input["f"],
                trunk_output["S_inputs_I"].clone(),
                trunk_output["S_I"].clone(),
                trunk_output["Z_II"].clone(), 
                noise_schedule,
                step_scale=1.5 # int he paper it says they changed this during the rollout
            )
        
        # find ground truth permutation
        # run diffusion training by noising the ground truth permutation closest to the rollout
        # run confidence model on embeddings and output structure
        confidence = self.confidence(
            trunk_output["S_inputs_I"],
            trunk_output["S_I"],
            trunk_output["Z_II"],
            diffusion_output["X_L"],
            input["f"],
        )
        # return output
        return dict(
            X_pred_L=trunk_output["X_pred_L"],
            X_pred_rollout_L=diffusion_output["X_L"],
            plddt=confidence["plddt"],
            pae=confidence["pae"],
            pde=confidence["pde"],
            exp_resolved=confidence["exp_resolved"],
            distogram=trunk_output["distogram"],
        )

class ErrorPredictionLoss(nn.Module):

    def __init__(self, loss_fn):
        super(ErrorPredictionLoss, self).__init__()
    
    def forward(
        self,
        network_input, 
        network_output,
        loss_input,
    ):
        # take network prediction and deconvolute symmetry
        # convert network prediction to I dimension
        # use symmetry algorithm to find relabeled ground truth
        # if is_atom(Seq):
        # resolve_equiv_natives(xs, natstack, maskstack)
        # else:
        # get Ls_prot and Ls_sm from f features
        # resolve_equiv_natives_asmb(xs, natstack, maskstack, ch_label, Ls_prot, Ls_sm)

        # convert ground truth back to L dimension

        # calculate lddt, pae, pde with respect to symmetry resolved ground truth

        # calculate loss based on predicted bins from network output
        pass

    def calc_lddt(self, X_pred_L, X_gt_L, crd_mask_I, seq):
        pass

    def calc_pae(self, X_pred_L, X_gt_L, crd_mask_I, seq):
        pass

    def calc_pde(self, X_pred_L, X_gt_L, crd_mask_I, seq):
        pass

