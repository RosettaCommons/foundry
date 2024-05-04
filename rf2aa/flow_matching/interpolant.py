import logging
import torch
import copy
import math
import functools as fn
import torch.nn.functional as F
from collections import defaultdict
from rf2aa.flow_matching import so3_utils, data_utils as du #, all_atom
from rf2aa.chemical import ChemicalData as ChemData
#from cogen.data import utils as du
from scipy.spatial.transform import Rotation
from scipy.optimize import linear_sum_assignment
from torch import autograd
#from cogen.motif_scaffolding import twisting

logger = logging.getLogger(__name__)

def _centered_gaussian(num_batch, num_res, device):
    noise = torch.randn(num_batch, num_res, 3, device=device)
    return noise - torch.mean(noise, dim=-2, keepdims=True)


def _uniform_so3(num_batch, num_res, device):
    return torch.tensor(
        Rotation.random(num_batch*num_res).as_matrix(),
        device=device,
        dtype=torch.float32,
    ).reshape(num_batch, num_res, 3, 3)


def _masked_categorical(num_batch, num_res, device):
    return torch.ones(
        num_batch, num_res, device=device) * ChemData().MASK_INDEX


def _trans_diffuse_mask(trans_t, trans_1, diffuse_mask):
    return trans_t * diffuse_mask[..., None] + trans_1 * (1 - diffuse_mask[..., None])


def _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask):
    return (
        rotmats_t * diffuse_mask[..., None, None]
        + rotmats_1 * (1 - diffuse_mask[..., None, None])
    )


def _aatypes_diffuse_mask(aatypes_t, aatypes_1, diffuse_mask):
    return aatypes_t * diffuse_mask + aatypes_1 * (1 - diffuse_mask)


class Interpolant:

    def __init__(self, cfg):
        self._cfg = cfg
        self._rots_cfg = cfg.rots
        self._trans_cfg = cfg.trans
        self._aatypes_cfg = cfg.aatypes
        self._sample_cfg = cfg.sampling
        self._igso3 = None

        self.num_tokens = self._aatypes_cfg.num_tokens+1 if self._aatypes_cfg.interpolant_type == "masking" else self._aatypes_cfg.num_tokens

    @property
    def igso3(self):
        if self._igso3 is None:
            sigma_grid = torch.linspace(0.1, 1.5, 1000)
            self._igso3 = so3_utils.SampleIGSO3(
                1000, sigma_grid, cache_dir='.cache')
        return self._igso3

    def set_device(self, device):
        self._device = device

    def sample_t(self, num_batch):
        t = torch.rand(num_batch, device=self._device)
        return t * (1 - 2*self._cfg.min_t) + self._cfg.min_t

    def _corrupt_trans(self, trans_1, t, res_mask, diffuse_mask):
        trans_nm_0 = _centered_gaussian(*res_mask.shape, self._device)
        trans_0 = trans_nm_0 * du.NM_TO_ANG_SCALE
        if self._trans_cfg.batch_ot:
            trans_0 = self._batch_ot(trans_0, trans_1, diffuse_mask)
        if self._trans_cfg.train_schedule == 'linear':
            trans_t = (1 - t[..., None]) * trans_0 + t[..., None] * trans_1
        elif self._trans_cfg.train_schedule == 'vpsde':
            # t (B,1)
            # trans_0 (B, N, 3)
            bmin = self._trans_cfg.vpsde_bmin
            bmax = self._trans_cfg.vpsde_bmax
            alpha_t = torch.exp(- bmin * (1-t) - 0.5 * (1-t)**2 * (bmax - bmin)) # (B,1)
            trans_t = torch.sqrt(alpha_t[..., None]) * trans_1 + torch.sqrt(1 - alpha_t[..., None]) * trans_0
        else:
            raise ValueError(
                f'Unknown trans schedule {self._trans_cfg.train_schedule}')
        trans_t = _trans_diffuse_mask(trans_t, trans_1, diffuse_mask)
        return trans_t * res_mask[..., None]
    
    def _batch_ot(self, trans_0, trans_1, res_mask):
        num_batch, num_res = trans_0.shape[:2]
        noise_idx, gt_idx = torch.where(
            torch.ones(num_batch, num_batch))
        batch_nm_0 = trans_0[noise_idx]
        batch_nm_1 = trans_1[gt_idx]
        batch_mask = res_mask[gt_idx]
        aligned_nm_0, aligned_nm_1, _ = du.batch_align_structures(
            batch_nm_0, batch_nm_1, mask=batch_mask
        ) 

        # fd shortcut with batch 1
        if num_batch == 1:
            return aligned_nm_0

        aligned_nm_0 = aligned_nm_0.reshape(num_batch, num_batch, num_res, 3)
        aligned_nm_1 = aligned_nm_1.reshape(num_batch, num_batch, num_res, 3)

        # Compute cost matrix of aligned noise to ground truth
        batch_mask = batch_mask.reshape(num_batch, num_batch, num_res)

        #fd ensure masked positions are zeroed out
        cost_matrix = torch.sum(
            torch.linalg.norm(
                batch_mask[...,None] * (
                    aligned_nm_0.nan_to_num() - 
                    aligned_nm_1.nan_to_num()
                ), dim=-1
            ), dim=-1
        ) / torch.sum(batch_mask, dim=-1)

        #fd gpu->cpu slowdown
        noise_perm, gt_perm = linear_sum_assignment(du.to_numpy(cost_matrix))
        return aligned_nm_0[(tuple(gt_perm), tuple(noise_perm))]
    
    def _corrupt_rotmats(self, rotmats_1, t, res_mask, diffuse_mask):
        num_batch, num_res = res_mask.shape
        noisy_rotmats = self.igso3.sample(
            torch.tensor([1.5]),
            num_batch*num_res
        ).to(self._device)
        noisy_rotmats = noisy_rotmats.reshape(num_batch, num_res, 3, 3)
        rotmats_0 = torch.einsum(
            "...ij,...jk->...ik", rotmats_1, noisy_rotmats)
        
        so3_schedule = self._rots_cfg.train_schedule
        if so3_schedule == 'exp':
            so3_t = 1 - torch.exp(-t*self._rots_cfg.exp_rate)
        elif so3_schedule == 'linear':
            so3_t = t
        else:
            raise ValueError(f'Invalid schedule: {so3_schedule}')
        rotmats_t = so3_utils.geodesic_t(so3_t[..., None], rotmats_1, rotmats_0)
        identity = torch.eye(3, device=self._device)
        rotmats_t = (
            rotmats_t * res_mask[..., None, None]
            + identity[None, None] * (1 - res_mask[..., None, None])
        )
        return _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask)

    def _corrupt_aatypes(self, aatypes_1, t, res_mask, diffuse_mask):
        num_batch, num_res = res_mask.shape
        assert aatypes_1.shape == (num_batch, num_res)
        assert t.shape == (num_batch, 1)
        assert res_mask.shape == (num_batch, num_res)
        assert diffuse_mask.shape == (num_batch, num_res)

        if self._aatypes_cfg.interpolant_type == "masking":
            u = torch.rand(num_batch, num_res, device=self._device)
            aatypes_t = aatypes_1.clone()
            if self._aatypes_cfg.schedule == 'linear':
                corruption_mask = u < (1 - t) # (B, N)
            elif self._aatypes_cfg.schedule == 'exp':
                coeff = self._aatypes_cfg.schedule_exp_rate
                corruption_mask = u < (1 - (1 / (math.exp(coeff) - 1)) * (torch.exp(coeff * t) - 1))
            else:
                raise ValueError(f"Unknown aatypes schedule {self._aatypes_cfg.schedule}")

            aatypes_t[corruption_mask] = ChemData().MASKINDEX
            aatypes_t = aatypes_t * res_mask + ChemData().MASKINDEX * (1 - res_mask)

            if self._aatypes_cfg.train_extra_mask > 0.0:
                u2 = torch.rand(num_batch, num_res, device=self._device)
                extra_mask = u2 < self._aatypes_cfg.train_extra_mask
                aatypes_t[extra_mask] = ChemData().MASKINDEX
        elif self._aatypes_cfg.interpolant_type == "uniform":
            u = torch.rand(num_batch, num_res, device=self._device)
            aatypes_t = aatypes_1.clone()
            assert self._aatypes_cfg.schedule == 'linear'
            corruption_mask = u < (1-t) # (B, N)
            uniform_sample = torch.randint_like(aatypes_t, low=0, high=du.NUM_TOKENS)
            aatypes_t[corruption_mask] = uniform_sample[corruption_mask]

            aatypes_t = aatypes_t * res_mask + du.MASK_TOKEN_INDEX * (1 - res_mask)
        else:
            raise ValueError(f"Unknown aatypes interpolant type {self._aatypes_cfg.interpolant_type}")

        return _aatypes_diffuse_mask(aatypes_t, aatypes_1, diffuse_mask)

    def corrupt_batch(self, batch):
        noisy_batch = copy.deepcopy(batch)

        # [B, N, 3]
        trans_1 = batch['trans_1']  # Angstrom

        # [B, N, 3, 3]
        rotmats_1 = batch['rotmats_1']

        # [B, N]
        aatypes_1 = batch['aatypes_1']

        # [B, N]
        res_mask = batch['res_mask']
        diffuse_mask = batch['diffuse_mask']
        diffuse_mask_seq = batch['diffuse_mask_seq']
        num_batch, num_res = diffuse_mask.shape

        # [B, 1]
        if self._cfg.separate_t:
            if self._cfg.hierarchical_t:
                max_t = torch.rand(num_batch, device=self._device) * (1 - self._cfg.min_t)
                so3_t = self._cfg.min_t + torch.rand(num_batch, device=self._device) * (max_t - self._cfg.min_t)
                r3_t = self._cfg.min_t + torch.rand(num_batch, device=self._device) * (max_t - self._cfg.min_t)
                cat_t = self._cfg.min_t + torch.rand(num_batch, device=self._device) * (max_t - self._cfg.min_t)
                so3_t = so3_t[:, None]
                r3_t = r3_t[:, None]
                cat_t = cat_t[:, None]
            else:
                so3_t = self.sample_t(num_batch)[:, None]
                r3_t = self.sample_t(num_batch)[:, None]
                cat_t = self.sample_t(num_batch)[:, None]
        elif self._cfg.codesign_separate_t:
            u = torch.rand((num_batch,), device=self._device)
            forward_fold_mask = (u < self._cfg.codesign_forward_fold_prop).float()
            inverse_fold_mask = (u < self._cfg.codesign_inverse_fold_prop + self._cfg.codesign_forward_fold_prop).float() * \
                (u >= self._cfg.codesign_forward_fold_prop).float()

            normal_structure_t = self.sample_t(num_batch)
            inverse_fold_structure_t = torch.ones((num_batch,), device=self._device)
            normal_cat_t = self.sample_t(num_batch)
            forward_fold_cat_t = torch.ones((num_batch,), device=self._device)

            # If we are forward folding, then cat_t should be 1
            # If we are inverse folding or codesign then cat_t should be uniform
            cat_t = forward_fold_mask * forward_fold_cat_t + (1 - forward_fold_mask) * normal_cat_t

            # If we are inverse folding, then structure_t should be 1
            # If we are forward folding or codesign then structure_t should be uniform
            structure_t = inverse_fold_mask * inverse_fold_structure_t + (1 - inverse_fold_mask) * normal_structure_t

            so3_t = structure_t[:, None]
            r3_t = structure_t[:, None]
            cat_t = cat_t[:, None]
        else:
            t = self.sample_t(num_batch)[:, None]
            so3_t = t
            r3_t = t
            cat_t = t
        noisy_batch['so3_t'] = so3_t
        noisy_batch['r3_t'] = r3_t
        noisy_batch['cat_t'] = cat_t

        # Apply corruptions
        if self._trans_cfg.corrupt:
            trans_t = self._corrupt_trans(
                trans_1, r3_t, res_mask, diffuse_mask)
        else:
            trans_t = trans_1
        if torch.any(torch.isnan(trans_t)):
            raise ValueError('NaN in trans_t during corruption')
        noisy_batch['trans_t'] = trans_t

        if self._rots_cfg.corrupt:
            rotmats_t = self._corrupt_rotmats(rotmats_1, so3_t, res_mask, diffuse_mask)
        else:
            rotmats_t = rotmats_1
        if torch.any(torch.isnan(rotmats_t)):
            raise ValueError('NaN in rotmats_t during corruption')
        noisy_batch['rotmats_t'] = rotmats_t
        if self._aatypes_cfg.corrupt:
            aatypes_t = self._corrupt_aatypes(aatypes_1, cat_t, res_mask, diffuse_mask_seq)
        else:
            aatypes_t = aatypes_1
        noisy_batch['aatypes_t'] = aatypes_t
        noisy_batch['trans_sc'] = torch.zeros_like(trans_1)
        noisy_batch['aatypes_sc'] = torch.zeros_like(
            aatypes_1)[..., None].repeat(1, 1, self.num_tokens)
        return noisy_batch
    
    def rot_sample_kappa(self, t):
        if self._rots_cfg.sample_schedule == 'exp':
            return 1 - torch.exp(-t*self._rots_cfg.exp_rate)
        elif self._rots_cfg.sample_schedule == 'linear':
            return t
        else:
            raise ValueError(
                f'Invalid schedule: {self._rots_cfg.sample_schedule}')

    def _trans_vector_field(self, t, trans_1, trans_t):
        if self._trans_cfg.sample_schedule == 'linear':
            trans_vf = (trans_1 - trans_t) / (1 - t)
        elif self._trans_cfg.sample_schedule == 'vpsde':
            bmin = self._trans_cfg.vpsde_bmin
            bmax = self._trans_cfg.vpsde_bmax
            bt = bmin + (bmax - bmin) * (1-t) # scalar
            alpha_t = torch.exp(- bmin * (1-t) - 0.5 * (1-t)**2 * (bmax - bmin)) # scalar
            trans_vf = 0.5 * bt * trans_t + \
                0.5 * bt * (torch.sqrt(alpha_t) * trans_1 - trans_t) / (1 - alpha_t)
        else:
            raise ValueError(
                f'Invalid sample schedule: {self._trans_cfg.sample_schedule}'
            )
        return trans_vf

    def _trans_euler_step(self, d_t, t, trans_1, trans_t):
        assert d_t >= 0
        trans_vf = self._trans_vector_field(t, trans_1, trans_t)
        return trans_t + trans_vf * d_t

    def _rots_euler_step(self, d_t, t, rotmats_1, rotmats_t):
        if self._rots_cfg.sample_schedule == 'linear':
            scaling = 1 / (1 - t)
        elif self._rots_cfg.sample_schedule == 'exp':
            scaling = self._rots_cfg.exp_rate
        else:
            raise ValueError(
                f'Unknown sample schedule {self._rots_cfg.sample_schedule}')
        # TODO: Add in SDE.
        return so3_utils.geodesic_t(
            scaling * d_t, rotmats_1, rotmats_t)

    def _regularize_step_probs(self, step_probs, aatypes_t):
        batch_size, num_res, S = step_probs.shape
        device = step_probs.device
        assert aatypes_t.shape == (batch_size, num_res)

        step_probs = torch.clamp(step_probs, min=0.0, max=1.0)
        # TODO replace with torch._scatter
        step_probs[
            torch.arange(batch_size, device=device).repeat_interleave(num_res),
            torch.arange(num_res, device=device).repeat(batch_size),
            aatypes_t.long().flatten()
        ] = 0.0
        step_probs[
            torch.arange(batch_size, device=device).repeat_interleave(num_res),
            torch.arange(num_res, device=device).repeat(batch_size),
            aatypes_t.long().flatten()
        ] = 1.0 - torch.sum(step_probs, dim=-1).flatten()
        step_probs = torch.clamp(step_probs, min=0.0, max=1.0)
        return step_probs

    def _aatypes_euler_step(self, d_t, t, logits_1, aatypes_t):
        # S = 21
        batch_size, num_res, S = logits_1.shape
        assert aatypes_t.shape == (batch_size, num_res)
        if self._aatypes_cfg.interpolant_type == "masking":
            assert S == 21
            device = logits_1.device
            
            mask_one_hot = torch.zeros((S,), device=device)
            mask_one_hot[du.MASK_TOKEN_INDEX] = 1.0

            # TODO just change the network to predict 20...
            if S == 21:
                logits_1[:, :, du.MASK_TOKEN_INDEX] = -1e9

            pt_x1_probs = F.softmax(logits_1 / self._aatypes_cfg.temp, dim=-1) # (B, D, S)


            aatypes_t_is_mask = (aatypes_t == du.MASK_TOKEN_INDEX).view(batch_size, num_res, 1).float()
            if self._aatypes_cfg.schedule == 'linear':
                step_probs = d_t * pt_x1_probs * ((1+ self._aatypes_cfg.noise*t) / ((1 - t))) # (B, D, S)
            elif self._aatypes_cfg.schedule == 'exp':
                coeff = self._aatypes_cfg.schedule_exp_rate
                alpha_t = (1 / (math.exp(coeff) - 1)) * (torch.exp(coeff * t) - 1)
                dt_alpha_t = (1 / (math.exp(coeff) - 1)) * (coeff * torch.exp(coeff * t))
                step_probs = d_t * pt_x1_probs * ((dt_alpha_t + self._aatypes_cfg.noise * alpha_t) / ((1 - alpha_t))) # (B, D, S)
            step_probs += d_t * (1 - aatypes_t_is_mask) * mask_one_hot.view(1, 1, -1) * self._aatypes_cfg.noise

            step_probs = self._regularize_step_probs(step_probs, aatypes_t)

            return torch.multinomial(step_probs.view(-1, S), num_samples=1).view(batch_size, num_res)
        elif self._aatypes_cfg.interpolant_type == "uniform":
            assert S == 20
            assert aatypes_t.max() < 20, "No UNK tokens allowed in the uniform sampling step!"
            device = logits_1.device

            pt_x1_probs = F.softmax(logits_1 / self._aatypes_cfg.temp, dim=-1) # (B, D, S)

            pt_x1_eq_xt_prob = torch.gather(pt_x1_probs, dim=-1, index=aatypes_t.long().unsqueeze(-1)) # (B, D, 1)
            assert pt_x1_eq_xt_prob.shape == (batch_size, num_res, 1)

            N = self._aatypes_cfg.noise
            step_probs = d_t * (pt_x1_probs * ((1 + N + N * (S - 1) * t) / (1-t)) + N * pt_x1_eq_xt_prob )

            step_probs = self._regularize_step_probs(step_probs, aatypes_t)

            return torch.multinomial(step_probs.view(-1, S), num_samples=1).view(batch_size, num_res)
        else:
            raise ValueError(f"Unknown aatypes interpolant type {self._aatypes_cfg.interpolant_type}")

    def _aatypes_euler_step_purity(self, d_t, t, logits_1, aatypes_t):
        batch_size, num_res, S = logits_1.shape
        assert aatypes_t.shape == (batch_size, num_res)
        assert S == self._aatypes_cfg.num_tokens + 1
        assert self._aatypes_cfg.interpolant_type == "masking"
        device = logits_1.device

        S = logits_1.shape[2]
        s = torch.arange(S)
        logits_1_wo_mask = logits_1[:, :, s != du.MASK_TOKEN_INDEX] # (B, D, S-1)
        pt_x1_probs = F.softmax(logits_1_wo_mask / self._aatypes_cfg.temp, dim=-1) # (B, D, S-1)
        # assert self._aatypes_cfg.schedule == 'linear'
        # step_probs = (d_t * pt_x1_probs * (1/(1-t))).clamp(max=1) # (B, D, S-1)
        max_logprob = torch.max(torch.log(pt_x1_probs), dim=-1)[0] # (B, D)
        # bias so that only currently masked positions get chosen to be unmasked
        max_logprob = max_logprob - (aatypes_t != du.MASK_TOKEN_INDEX).float() * 1e9
        sorted_max_logprobs_idcs = torch.argsort(max_logprob, dim=-1, descending=True) # (B, D)

        if self._aatypes_cfg.schedule == 'linear':
            unmask_probs = (d_t * ( (1 + self._aatypes_cfg.noise * t) / (1-t)).to(device)).clamp(max=1) # scalar
        elif self._aatypes_cfg.schedule == 'exp':
            assert self._aatypes_cfg.noise == 0.0, "If you want noise, double check the calcs for exp schedule"
            coeff = self._aatypes_cfg.schedule_exp_rate
            alpha_t = (1 / (math.exp(coeff) - 1)) * (torch.exp(coeff * t) - 1)
            dt_alpha_t = (1 / (math.exp(coeff) - 1)) * (coeff * torch.exp(coeff * t))
            unmask_probs = (d_t * (dt_alpha_t / (1 - alpha_t))).to(device).clamp(max=1) # scalar
        elif self._aatypes_cfg.schedule == 'never_decode':
            unmask_probs = torch.zeros_like(t).to(device)
        else:
            raise Exception(f'unknown interpolant.aatypes.schedule {self._aatypes_cfg.schedul}')

        number_to_unmask = torch.binomial(count=torch.count_nonzero(aatypes_t == du.MASK_TOKEN_INDEX, dim=-1).float(),
                                          prob=unmask_probs)
        unmasked_samples = torch.multinomial(pt_x1_probs.view(-1, S-1), num_samples=1).view(batch_size, num_res)

        # Vectorized version of:
        # for b in range(B):
        #     for d in range(D):
        #         if d < number_to_unmask[b]:
        #             aatypes_t[b, sorted_max_logprobs_idcs[b, d]] = unmasked_samples[b, sorted_max_logprobs_idcs[b, d]]

        D_grid = torch.arange(num_res, device=device).view(1, -1).repeat(batch_size, 1)
        mask1 = (D_grid < number_to_unmask.view(-1, 1)).float()
        inital_val_max_logprob_idcs = sorted_max_logprobs_idcs[:, 0].view(-1, 1).repeat(1, num_res)
        masked_sorted_max_logprobs_idcs = (mask1 * sorted_max_logprobs_idcs + (1-mask1) * inital_val_max_logprob_idcs).long()
        mask2 = torch.zeros((batch_size, num_res), device=device)
        mask2.scatter_(dim=1, index=masked_sorted_max_logprobs_idcs, src=torch.ones((batch_size, num_res), device=device))
        unmask_zero_row = (number_to_unmask == 0).view(-1, 1).repeat(1, num_res).float()
        mask2 = mask2 * (1 - unmask_zero_row)
        aatypes_t = aatypes_t * (1 - mask2) + unmasked_samples * mask2

        # re-mask
        u = torch.rand(batch_size, num_res, device=self._device)
        re_mask_mask = (u < d_t * self._aatypes_cfg.noise).float()
        aatypes_t = aatypes_t * (1 - re_mask_mask) + du.MASK_TOKEN_INDEX * re_mask_mask

        return aatypes_t


    def sample(
            self,
            num_batch,
            num_res,
            model,
            num_timesteps=None,
            trans_potential=None,
            trans_0=None,
            rotmats_0=None,
            aatypes_0=None,
            trans_1=None,
            rotmats_1=None,
            aatypes_1=None,
            diffuse_mask=None,
            chain_idx=None,
            res_idx=None,
            t_nn=None,
            forward_folding=False,
            inverse_folding=False,
            separate_t=False,
            feats_1d=None,
            bond_feats=None,
            chirals=None,
            atom_frames=None,
            atom_frames_full=None,
            atom_frame_2d=None,
            is_sm=None,
            indep=None,
            masks_1d=None,
            diffuse_mask_seq=None,
        ):
        # TODO: Refactor into its own class.

        res_mask = torch.ones(num_batch, num_res, device=self._device)

        if trans_potential is None:
            trans_potential = self._trans_potential

        # Set-up initial prior samples

        if trans_0 is None:
            trans_0 = _centered_gaussian(
                num_batch, num_res, self._device) * du.NM_TO_ANG_SCALE
        if rotmats_0 is None:
            rotmats_0 = _uniform_so3(num_batch, num_res, self._device)
        if aatypes_0 is None:
            if self._aatypes_cfg.interpolant_type == "masking":
                aatypes_0 = _masked_categorical(num_batch, num_res, self._device)
            elif self._aatypes_cfg.interpolant_type == "uniform":
                aatypes_0 = torch.randint_like(res_mask, low=0, high=self.num_tokens)
            else:
                raise ValueError(f"Unknown aatypes interpolant type {self._aatypes_cfg.interpolant_type}")
        if res_idx is None:
            res_idx = torch.arange(
                num_res,
                device=self._device,
                dtype=torch.float32)[None].repeat(num_batch, 1)

        if chain_idx is None:
            chain_idx = res_mask

        if diffuse_mask is None:
            diffuse_mask = res_mask
        
        if diffuse_mask_seq is None:
            raise Exception(f'diffuse_mask_seq is None.')

        trans_sc = torch.zeros(num_batch, num_res, 3, device=self._device)
        aatypes_sc = torch.zeros(
            num_batch, num_res, self.num_tokens, device=self._device)
        batch = {
            'res_mask': res_mask,
            'diffuse_mask': diffuse_mask,
            'chain_idx': chain_idx,
            'res_idx': res_idx,
            'trans_sc': trans_sc,
            'aatypes_sc': aatypes_sc,
            'feats_1d': feats_1d,
            'bond_feats': bond_feats,
            'chirals': chirals,
            'atom_frames': atom_frames,
            'atom_frames_full': atom_frames_full,
            'atom_frame_2d': atom_frame_2d,
            'is_sm': is_sm,
            'indep': indep,
            'masks_1d': masks_1d,
        }

        motif_scaffolding = False
        if diffuse_mask is not None and trans_1 is not None and rotmats_1 is not None and aatypes_1 is not None:
            motif_scaffolding = True
            motif_mask = ~diffuse_mask.bool()
        if motif_scaffolding and not self._cfg.twisting.use: # amortisation 
            diffuse_mask = diffuse_mask.expand(num_batch, -1) # shape = (B, num_residue)
            batch['diffuse_mask'] = diffuse_mask
            rotmats_0 = _rots_diffuse_mask(rotmats_0, rotmats_1, diffuse_mask)

            # Center translations based on motif
            motif_1 = trans_1 * motif_mask[..., None]
            motif_com = torch.sum(motif_1, dim=1) / (torch.sum(motif_mask) + 1)
            trans_1 -= motif_com[:, None, :]
            trans_0 = _trans_diffuse_mask(trans_0, trans_1, diffuse_mask)
            aatypes_0 = _aatypes_diffuse_mask(aatypes_0, aatypes_1, diffuse_mask_seq.long())
            batch['trans_sc'] = _trans_diffuse_mask(
                trans_sc, trans_1, diffuse_mask)
            logits_1 = torch.nn.functional.one_hot(
                aatypes_1.long(),
                num_classes=self.num_tokens
            ).float()
            batch['aatypes_sc'] = _trans_diffuse_mask(
                aatypes_sc, logits_1, diffuse_mask_seq.long())
        else:
            trans_1 = torch.zeros(num_batch, num_res, 3, device=self._device)
            rotmats_1 = torch.eye(3, device=self._device)[None, None].repeat(num_batch, num_res, 1, 1)
            aatypes_1 = torch.zeros((num_batch, num_res), device=self._device).long()
            logits_1 = torch.nn.functional.one_hot(
                aatypes_1,
                num_classes=self.num_tokens
            ).float()

        if forward_folding:
            assert aatypes_1 is not None
            assert self._aatypes_cfg.noise == 0
        if forward_folding and separate_t:
            aatypes_0 = aatypes_1
        if inverse_folding:
            assert trans_1 is not None
            assert rotmats_1 is not None
        if inverse_folding and separate_t:
            trans_0 = trans_1
            rotmats_0 = rotmats_1

        logs_traj = defaultdict(list)
        if motif_scaffolding and self._cfg.twisting.use: # sampling / guidance
            raise NotImplementedError
            assert trans_1.shape[0] == 1 # assume only one motif
            motif_mask = motif_mask.squeeze(0)
            trans_motif = trans_1[:, motif_mask]  # [1, motif_res, 3]
            trans_motif -= trans_motif.mean(dim=[-2], keepdim=True)
            R_motif = rotmats_1[:, motif_mask]  # [1, motif_res, 3, 3]

        # Set-up time
        if num_timesteps is None:
            num_timesteps = self._sample_cfg.num_timesteps
        ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps)
        t_1 = ts[0]

        frames_to_atom37 = lambda x,y: all_atom.atom37_from_trans_rot(x, y, res_mask).detach().cpu()
        trans_t_1, rotmats_t_1, aatypes_t_1 = trans_0, rotmats_0, aatypes_0
        prot_traj = [(frames_to_atom37(trans_t_1, rotmats_t_1), aatypes_0.detach().cpu().long())] 
        clean_traj = []
        for t_2 in ts[1:]:

            # Run model.
            if self._trans_cfg.corrupt:
                batch['trans_t'] = trans_t_1
            else:
                if trans_1 is None:
                    raise ValueError('Must provide trans_1 if not corrupting.')
                batch['trans_t'] = trans_1

            if self._rots_cfg.corrupt:
                batch['rotmats_t'] = rotmats_t_1
            else:
                if rotmats_1 is None:
                    raise ValueError('Must provide rotmats_1 if not corrupting.')
                batch['rotmats_t'] = rotmats_1

            if self._aatypes_cfg.corrupt:
                batch['aatypes_t'] = aatypes_t_1
            else:
                if aatypes_1 is None:
                    raise ValueError('Must provide aatype if not corrupting.')
                batch['aatypes_t'] = aatypes_1


            t = torch.ones((num_batch, 1), device=self._device) * t_1
            
            if t_nn is not None:
                batch['r3_t'], batch['so3_t'], batch['cat_t'] = torch.split(t_nn(t), -1)
            else:

                if self._cfg.provide_kappa:
                    batch['so3_t'] = self.rot_sample_kappa(t)
                else:
                    batch['so3_t'] = t
                batch['r3_t'] = t
                batch['cat_t'] = t
            if forward_folding and separate_t:
                batch['cat_t'] = (1 - self._cfg.min_t) * torch.ones_like(batch['cat_t'])
            if inverse_folding and separate_t:
                batch['r3_t'] = (1 - self._cfg.min_t) * torch.ones_like(batch['r3_t'])
                batch['so3_t'] = (1 - self._cfg.min_t) * torch.ones_like(batch['so3_t'])

            d_t = t_2 - t_1

            use_twisting = self._cfg.twisting.use and t_1 >= self._cfg.twisting.t_min

            if use_twisting: # Reconstruction guidance
                with torch.inference_mode(False):
                    batch, Log_delta_R, delta_x = twisting.perturbations_for_grad(batch)
                    model_out = model(batch)
                    t = batch['r3_t'] #TODO: different time for SO3?
                    trans_t_1, rotmats_t_1, logs_traj = self.guidance(trans_t_1, rotmats_t_1, model_out, motif_mask, R_motif, trans_motif, Log_delta_R, delta_x, t, d_t, logs_traj)

            else:
                with torch.no_grad():
                    model_out = model(batch)

            # Process model output.
            pred_trans_1 = model_out['pred_trans']
            pred_rotmats_1 = model_out['pred_rotmats']
            pred_aatypes_1 = model_out['pred_aatypes']
            pred_logits_1 = model_out['pred_logits']
            clean_traj.append((frames_to_atom37(pred_trans_1, pred_rotmats_1), pred_aatypes_1.detach().cpu()))
            if forward_folding:
                pred_logits_1 = 100.0 * logits_1
            if inverse_folding:
                pred_trans_1 = trans_1
                pred_rotmats_1 = rotmats_1
            

            if self._cfg.self_condition:
                batch['trans_sc'] = _trans_diffuse_mask(
                    pred_trans_1, trans_1, diffuse_mask)
                if forward_folding:
                    batch['aatypes_sc'] = logits_1
                else:
                    batch['aatypes_sc'] = _trans_diffuse_mask(
                        pred_logits_1, logits_1, diffuse_mask)

            # Take reverse step            
            trans_t_2 = self._trans_euler_step(
                d_t, t_1, pred_trans_1, trans_t_1)
            if trans_potential is not None:
                with torch.inference_mode(False):
                    grad_pred_trans_1 = pred_trans_1.clone().detach().requires_grad_(True)
                    pred_trans_potential = autograd.grad(outputs=trans_potential(grad_pred_trans_1), inputs=grad_pred_trans_1)[0]
                if self._trans_cfg.potential_t_scaling:
                    trans_t_2 -= t_1 / (1 - t_1) * pred_trans_potential * d_t
                else:
                    trans_t_2 -= pred_trans_potential * d_t
            rotmats_t_2 = self._rots_euler_step(
                d_t, t_1, pred_rotmats_1, rotmats_t_1)

            logger.debug(f'{self._aatypes_cfg.do_purity=}')
            if self._aatypes_cfg.do_purity:
                aatypes_t_2 = self._aatypes_euler_step_purity(d_t, t_1, pred_logits_1, aatypes_t_1)
            else:
                aatypes_t_2 = self._aatypes_euler_step(d_t, t_1, pred_logits_1, aatypes_t_1)

            if not self._cfg.twisting.use:
                trans_t_2 = _trans_diffuse_mask(trans_t_2, trans_1, diffuse_mask)
                rotmats_t_2 = _rots_diffuse_mask(rotmats_t_2, rotmats_1, diffuse_mask)
                aatypes_t_2 = _aatypes_diffuse_mask(aatypes_t_2, aatypes_1, diffuse_mask)
            trans_t_1, rotmats_t_1, aatypes_t_1 = trans_t_2, rotmats_t_2, aatypes_t_2
            prot_traj.append((frames_to_atom37(trans_t_2, rotmats_t_2), aatypes_t_2.cpu().detach().long()))

            t_1 = t_2

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]

        if self._trans_cfg.corrupt:
            batch['trans_t'] = trans_t_1
        else:
            if trans_1 is None:
                raise ValueError('Must provide trans_1 if not corrupting.')
            batch['trans_t'] = trans_1

        if self._rots_cfg.corrupt:
            batch['rotmats_t'] = rotmats_t_1
        else:
            if rotmats_1 is None:
                raise ValueError('Must provide rotmats_1 if not corrupting.')
            batch['rotmats_t'] = rotmats_1

        if self._aatypes_cfg.corrupt:
            batch['aatype_t'] = aatypes_t_1
        else:
            if aatypes_1 is None:
                raise ValueError('Must provide aatype if not corrupting.')
            batch['aatype_t'] = aatypes_1

        with torch.no_grad():
            model_out = model(batch)
        pred_trans_1 = model_out['pred_trans']
        pred_rotmats_1 = model_out['pred_rotmats']
        pred_aatypes_1 = model_out['pred_aatypes']
        if forward_folding:
            pred_aatypes_1 = aatypes_1
        if inverse_folding:
            pred_trans_1 = trans_1
            pred_rotmats_1 = rotmats_1
        pred_atom37 = frames_to_atom37(pred_trans_1, pred_rotmats_1)
        clean_traj.append((pred_atom37, pred_aatypes_1.detach().cpu().long()))
        prot_traj.append((pred_atom37, pred_aatypes_1.detach().cpu().long()))
        return prot_traj, clean_traj, model_out | dict(trans_t=batch['trans_t'], rotmats_t=batch['rotmats_t'], r3_t=batch['r3_t'], so3_t=batch['so3_t'])


    def guidance(self, trans_t, rotmats_t, model_out, motif_mask, R_motif, trans_motif, Log_delta_R, delta_x, t, d_t, logs_traj):
        # Select motif
        motif_mask = motif_mask.clone()
        trans_pred = model_out['pred_trans'][:, motif_mask]  # [B, motif_res, 3]
        R_pred = model_out['pred_rotmats'][:, motif_mask]  # [B, motif_res, 3, 3]

        # Proposal for marginalising motif rotation
        F = twisting.motif_rots_vec_F(trans_motif, R_motif, self._cfg.twisting.num_rots, align=self._cfg.twisting.align, scale=self._cfg.twisting.scale_rots, device=self._device, dtype=torch.float32)

        # Estimate p(motif|predicted_motif)
        grad_Log_delta_R, grad_x_log_p_motif, logs = twisting.grad_log_lik_approx(R_pred, trans_pred, R_motif, trans_motif, Log_delta_R, delta_x, None, None, None, F, twist_potential_rot=self._cfg.twisting.potential_rot, twist_potential_trans=self._cfg.twisting.potential_trans)

        with torch.no_grad():
            # Choose scaling
            t_trans = t
            t_so3 = self.rot_sample_kappa(t) if self._cfg.twisting.provide_kappa else t
            if self._cfg.twisting.scale_w_t == 'ot':
                var_trans = ((1 - t_trans) / t_trans)[:, None]
                var_rot = ((1 - t_so3) / t_so3)[:, None, None]
            elif self._cfg.twisting.scale_w_t == 'linear':
                var_trans = (1 - t)[:, None]
                var_rot = (1 - t_so3)[:, None, None]
            elif self._cfg.twisting.scale_w_t == 'constant':
                num_batch = trans_pred.shape[0]
                var_trans = torch.ones((num_batch, 1, 1)).to(R_pred.device)
                var_rot = torch.ones((num_batch, 1, 1, 1)).to(R_pred.device)
            var_trans = var_trans + self._cfg.twisting.obs_noise ** 2
            var_rot = var_rot + self._cfg.twisting.obs_noise ** 2

            trans_scale_t = self._cfg.twisting.scale / var_trans
            rot_scale_t = self._cfg.twisting.scale / var_rot

            # Compute update
            trans_t, rotmats_t = twisting.step(trans_t, rotmats_t, grad_x_log_p_motif, grad_Log_delta_R, d_t, trans_scale_t, rot_scale_t, self._cfg.twisting.update_trans, self._cfg.twisting.update_rot)

        # delete unsused arrays to prevent from any memory leak
        del grad_Log_delta_R
        del grad_x_log_p_motif
        del Log_delta_R
        del delta_x
        for key, value in model_out.items():
            model_out[key] = value.detach().requires_grad_(False)
        # # log metrics
        # for k, v in logs.items():
        #     logs_traj[k].append(v)
        # grad_x_log_p_motif_norm = torch.norm(grad_x_log_p_motif, dim=[-1]).mean(-1)#.item()
        # grad_Log_delta_R_norm = torch.norm(grad_Log_delta_R, dim=[-1, -2]).mean(-1)#.item()
        # logs_traj['grad_x_log_p_motif_norm'].append(grad_x_log_p_motif_norm)
        # logs_traj['grad_Log_delta_R'].append(grad_Log_delta_R_norm)
        # logs_traj['trans_scale_t'].append(trans_scale_t.squeeze())
        # logs_traj['rot_scale_t'].append(rot_scale_t.squeeze())
        # logs_traj['guidance_trans_norm'].append(grad_x_log_p_motif_norm * trans_scale_t.squeeze())
        # logs_traj['guidance_Log_delta_R_norm'].append(grad_Log_delta_R_norm * rot_scale_t.squeeze())

        return trans_t, rotmats_t, logs_traj
