import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from rf2aa.training.checkpoint import create_custom_forward


class AF3_with_rollout(nn.Module):
    """Implements rollout on each training step"""

    def __init__(self, model, confidence, sampler, batch_size_rollout):
        super(AF3_with_rollout, self).__init__()
        self.model = model
        self.confidence = confidence
        self.sampler = sampler
        self.num_timesteps = 20
        self.batch_size_rollout = batch_size_rollout

    def forward(
        self, input, n_cycle, seq, rep_atom_idxs, no_sync, frame_atom_idxs=None
    ):
        # first do forward pass
        with torch.no_grad():
            trunk_output = self.model.trunk_forward(input, n_cycle, no_sync)

            noise_schedule = self.sampler.construct_noise_schedule(
                self.num_timesteps, 0, 1
            ).to(input["f"]["msa"].device)
            diffusion_output = self.sampler.sample_diffusion(
                input["f"],
                trunk_output["S_inputs_I"].clone().detach(),
                trunk_output["S_I"].clone().detach(),
                trunk_output["Z_II"].clone().detach(),
                noise_schedule,
                step_scale=1.5,  # int he paper it says they changed this during the rollout
                D=self.batch_size_rollout,
            )

        # run confidence model on embeddings and output structure
        # Bug in deepspeed backwards requires us to do it with batch size 1
        confidence_stack = {}
        confidence_stack["plddt_logits"] = None
        confidence_stack["pae_logits"] = None
        confidence_stack["pde_logits"] = None
        confidence_stack["exp_resolved_logits"] = None
        with torch.enable_grad():
            for i in range(self.batch_size_rollout):
                confidence = checkpoint.checkpoint(
                    create_custom_forward(
                        self.confidence, frame_atom_idxs=frame_atom_idxs
                    ),
                    trunk_output["S_inputs_I"],
                    trunk_output["S_I"],
                    trunk_output["Z_II"],
                    diffusion_output["X_L"][i].unsqueeze(0),
                    seq,
                    rep_atom_idxs,
                    use_reentrant=False,
                )

                for k, v in confidence.items():
                    if confidence_stack[k] is not None:
                        confidence_stack[k] = torch.cat((confidence_stack[k], v), dim=0)
                    else:
                        confidence_stack[k] = v

        return dict(
            X_L=None,
            X_pred_rollout_L=diffusion_output["X_L"],
            plddt=confidence_stack["plddt_logits"],
            pae=confidence_stack["pae_logits"],
            pde=confidence_stack["pde_logits"],
            exp_resolved=confidence_stack["exp_resolved_logits"],
            distogram=None,
        )


class ConfidenceLoss(nn.Module):
    def __init__(
        self,
        plddt,
        pae,
        pde,
        exp_resolved,
        weight=1,
        rank_loss=None,
        log_statistics=False,
    ):
        super(ConfidenceLoss, self).__init__()
        self.weight = weight
        self.plddt = plddt
        self.pae = pae
        self.pde = pde
        self.exp_resolved = exp_resolved
        self.cce = nn.CrossEntropyLoss(reduction="none")
        self.eps = 1e-6
        self.rank_loss = rank_loss
        self.log_statistics = log_statistics

    def forward(
        self,
        network_input,
        network_output,
        loss_input,
    ):
        raise NotImplementedError(
            "ConfidenceLoss is not implemented here; see rf2aa/loss/af3_confidence_loss.py for implementation"
        )
