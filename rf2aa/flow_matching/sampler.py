import torch
import tree
import torch.nn.functional as F
from typing import Any, Dict, Tuple
import warnings
from rf2aa.data.rotation_augmentation import centre_random_augmentation
from rf2aa.flow_matching.interpolant import _centered_gaussian, _uniform_so3
import rf2aa.flow_matching.data_utils as du
from rf2aa.flow_matching import data_transforms
from rf2aa.training.recycling import recycle_step_packed, recycle_step_gen
from rf2aa.chemical import ChemicalData as ChemData
from rf2aa.training.recycling import unpack_outputs
from rf2aa.util import rigid_from_3_points, writepdb_file


class Sampler:
    def __init__(self, model, num_timesteps, min_t, interpolant, xyz_converter, is_training) -> None:
        raise NotImplementedError("Sampler has been retired")
        self.model = model
        self.num_timesteps = num_timesteps
        self.min_t = min_t
        self.interpolant = interpolant
        self.device = self.interpolant._device
        self.xyz_converter = xyz_converter
        self.is_training = is_training

    def sample(self, inputs: Tuple[str, Any], use_amp=False) -> Dict[str, Any]:
        # first receive inputs from dataloader
        # convert them into features
        network_input = self._get_network_input(inputs)
        ts = torch.linspace(self.min_t, 1.0, self.num_timesteps)
        t_1 = ts[0]
        # create prior
        rotmats_t_1, trans_t_1 = self._setup_prior(network_input["seq_unmasked"])
        # track px1 and xt 
        px1s = []
        xts = []
        # run the model for n_steps
        for t_2 in ts[1:]:
            d_t = t_2-t_1
            # collect features for each step
            updated_features = self._construct_xt_features(rotmats_t_1, trans_t_1, network_input) 
            network_input.update(updated_features)
            # run model
            output_i = recycle_step_packed(self.model, network_input,1, use_amp=use_amp, nograds=True)
            xyz = output_i[5][-1]
            N, Ca, C = xyz[...,0, :], xyz[...,1, :], xyz[...,2, :]
            px1s.append(xyz)
            xts.append(network_input["xyz_t"][0])
            pred_rotmats, pred_trans = rigid_from_3_points(N, Ca, C, is_na=None) 
            # take euler step
            rotmats_t_2, trans_t_2 = self._take_step(pred_rotmats, pred_trans, 
                                                     rotmats_t_1, trans_t_1, d_t, t_1)
            # set prev_rots, prev_trans to curr
            rotmats_t_1, trans_t_1 = rotmats_t_2, trans_t_2
            t_1 = t_2
        # 

        updated_features = self._construct_xt_features(rotmats_t_1, trans_t_1, network_input) 
        network_input.update(updated_features)
        # run model
        output_i = recycle_step_packed(self.model, network_input,1, use_amp=use_amp, nograds=True)
        # return the updated positions
        mask = torch.ones(xyz.shape[:-1], device=xyz.device).bool()
        mask = F.pad(mask, (0,33))
        mask = mask[0]
        self.write_traj(px1s, mask, network_input["seq_unmasked"])
        return output_i

    def _get_network_input(self, inputs):
        out = prepare_input_fm(inputs, self.interpolant, self.xyz_converter, device=self.device)
        network_input = out[2]
        return network_input

    def _setup_prior(self, seq_unmasked):
        B, L = seq_unmasked.shape[:2]
        trans_0 = _centered_gaussian(
            B, L, self.device) * du.NM_TO_ANG_SCALE

        rotmats_0 = _uniform_so3(B, L, self.device)
        return rotmats_0, trans_0 

    def _take_step(self, pred_rotmats_1, pred_trans_1, rotmats_t_1, trans_t_1, d_t, t_1):
        rotmats_t_2 = self.interpolant._rots_euler_step(
                d_t, t_1, pred_rotmats_1, rotmats_t_1)


        trans_t_2 = self.interpolant._trans_euler_step(
                d_t, t_1, pred_trans_1, trans_t_1)
        return rotmats_t_2, trans_t_2

    def _construct_xt_features(self, rotmats, trans, network_input):
        """ overwrite template features with previous xt prediction """
        xyz = data_transforms.rigids_to_xyz(rotmats, trans)
        xyz = F.pad(xyz, (0, 0, 0, 33))

        mask_t = torch.ones(xyz.shape[:3], device=xyz.device).bool()
        mask_t[...,3:] = False
        #TODO: center backbone here
        t1d, seq_unmasked, atom_frames = network_input["t1d"], network_input["seq_unmasked"], network_input["atom_frames"]
        t2d, mask_t_2d, alpha_t, alpha_prev = construct_template_feats(xyz[None], mask_t[None], \
                    t1d, seq_unmasked, atom_frames, self.xyz_converter, use_atom_frames=False)

        t1d = torch.zeros_like(t1d)
        alpha_t, alpha_prev = torch.zeros_like(alpha_t), torch.zeros_like(alpha_prev)

        updated_features = {
            "t1d": t1d,
            "t2d": t2d,
            "xyz_t": xyz[...,1, :][None],
            "mask_t": mask_t_2d

        }
        return updated_features

    def write_traj(self, xyz_list, mask, seq):
        f = open("traj.pdb", "w+")
        for i, xyz in enumerate(xyz_list):
            writepdb_file(f, xyz, seq, modelnum=i, atom_mask=mask)


class AllAtomSampler(Sampler):
    """ sampler for model which predicts all atom positions, not frames/torsions """
    def __init__(self, model, num_timesteps, min_t, interpolant, xyz_converter, is_training) -> None:
        super().__init__(model, num_timesteps, min_t, interpolant, xyz_converter, is_training)
        raise NotImplementedError("AllAtomSampler has been retired")
        self.allatom_mask = ChemData().allatom_mask.to(self.device)

    def sample(self, inputs: Tuple[str, Any], n_cycle=1, use_amp=False) -> Dict[str, Any]:
        # first receive inputs from dataloader
        # convert them into features
        network_input = self._get_network_input(inputs)
        ts = torch.linspace(self.min_t, 1.0, self.num_timesteps)
        # create prior
        seq_unmasked = network_input["seq_unmasked"].to(self.device)
        trans_t_1 = self._setup_prior(seq_unmasked)
        network_input["trans_t"] = trans_t_1[None]
        # run first model fwd pass to get evoformer features
        output_i = recycle_step_gen(self.model, network_input, n_cycle, use_amp=use_amp, nograds=True)
        latent_feats = {
            "msa": output_i[-3],
            "pair": output_i[-2],
            "state": output_i[-1],
            "seq_unmasked": seq_unmasked,
            "dist_matrix": network_input["dist_matrix"].to(self.device),
            "idx": network_input["idx"].to(self.device),
            "trans_t": trans_t_1[None],
            "t": network_input["t"]
        }
        output_i_trunk = output_i
        # run the model refinement for n_steps
        output_i, px1, xts = self.run_refiner(latent_feats, ts)
        # HACK: get features from evoformer, this needs to become a dictionary to allow for assignment
        output_i = list(output_i)
        for i in range(len(output_i)):
            if output_i[i] is None:
                output_i[i] = output_i_trunk[i]
        return tuple(output_i)

    def _setup_prior(self, seq_unmasked):
        B, L = seq_unmasked.shape
        xyz = torch.zeros(B, L, ChemData().NTOTAL, 3, device=self.device)
        is_real_atom = self.allatom_mask[seq_unmasked]
        num_atoms = is_real_atom.sum() 
        xyz[is_real_atom] = _centered_gaussian(B, num_atoms, self.device) * du.NM_TO_ANG_SCALE
        return xyz

    def _get_network_input(self, inputs):
        out = prepare_input_fm_allatom(inputs, self.interpolant, self.xyz_converter, device=self.device)
        network_input = out[2]
        return network_input

    def _take_step(self, pred_trans_1, trans_t_1, d_t, t_1, seq_unmasked):
        is_real_atom = self.allatom_mask[seq_unmasked]
        trans_t_2_rolled = pred_trans_1.clone()
        pred_trans_1 = pred_trans_1[is_real_atom]
        trans_t_1 = trans_t_1[is_real_atom]
        trans_t_2 = self.interpolant._trans_euler_step(
                d_t, t_1, pred_trans_1, trans_t_1)
        trans_t_2_rolled[is_real_atom] = trans_t_2
        return trans_t_2_rolled
    
    def run_refiner(self, latent_feats, ts):
        px1s = []
        xts = []
        t_1 = ts[0]
        trans_t_1 = latent_feats["trans_t"][0]
        for t_2 in ts[1:]:
            d_t = t_2-t_1
            # collect features for each step
            pred_trans_1 = self._run_diffusion_step(latent_feats)
            xts.append(trans_t_1)
            trans_t_2 = self._take_step(pred_trans_1, trans_t_1, d_t, t_1, seq_unmasked=latent_feats["seq_unmasked"])
            trans_t_1 = trans_t_2
            latent_feats["trans_t"] = trans_t_1[None]
            t_1 = t_2
        outputs = {}
        outputs["xyz"] = pred_trans_1
        outputs["state"] = latent_feats["state"]
        output_i = unpack_outputs(outputs, latent_feats, return_raw=False)
        return output_i, px1s, xts

    def _run_diffusion_step(self, latent_feats):
        outputs = self.model.module.model.refinement(latent_feats)
        pred_trans_1 = outputs["xyz"]
        return pred_trans_1


class AF3Sampler:

    def __init__(self, config, model, confidence=None):
        self.config = config
        self.model = model
        self.device = next(model.parameters()).device
        self.confidence = confidence


    def sample(self, inputs: Tuple[str, Any], n_cycle=1, use_amp=False) -> Dict[str, Any]:
        # first receive inputs from dataloader
        # convert them into features
        network_input = self._get_network_input(inputs)

        # send network input to gpu
        #network_input=tree.map_structure(lambda x: x.to(self.device) if hasattr(x, 'cpu') else x, network_input)
        def _inmap(path, x):
            if hasattr(x, 'cpu') and path != ('f','msa_stack'):
                return x.to(self.device) 
            else:
                return x
        network_input = tree.map_structure_with_path(_inmap, network_input)


        # run model to get evoformer features
        recycle_inputs = self.model.pre_recycle(**network_input)
        for i in range(n_cycle):
            # run the model for n_steps
            recycle_inputs["f"]["msa"] = network_input["f"]["msa_stack"][i].to(self.device)
            recycle_inputs = self.model.recycle(**recycle_inputs)

        n_diff_steps = self.config.af3_inference.num_steps

        noise_schedule = self.construct_noise_schedule(n_diff_steps, 0, 1)
        noise_schedule = noise_schedule.to(self.device)
        post_recycle_outputs = recycle_inputs

        if self.config.af3_inference.solver == "af3":
            X_L = self.sample_diffusion(
                network_input["f"], post_recycle_outputs["S_inputs_I"],
                post_recycle_outputs["S_I"], post_recycle_outputs["Z_II"],
                noise_schedule
            ) 
        elif self.config.af3_inference.solver == "simple":
            X_L = self.sample_diffusion_simple(
                network_input["f"], post_recycle_outputs["S_inputs_I"],
                post_recycle_outputs["S_I"], post_recycle_outputs["Z_II"],
                noise_schedule
            ) 
        elif self.config.af3_inference.solver == "euler":
            X_L = self.sample_diffusion_euler(
                network_input["f"], post_recycle_outputs["S_inputs_I"],
                post_recycle_outputs["S_I"], post_recycle_outputs["Z_II"],
                noise_schedule
            ) 
        elif self.config.af3_inference.solver == "heun":
            X_L = self.sample_diffusion_heun(
                network_input["f"], post_recycle_outputs["S_inputs_I"],
                post_recycle_outputs["S_I"], post_recycle_outputs["Z_II"],
                noise_schedule
            ) 

        outputs = self.model.post_recycle(
            **recycle_inputs,
            is_training=False,
        )
        outputs.update(X_L)

        #run confidence
        if self.confidence is not None:
            confidence_feats = self._get_confidence_feats(inputs)
            confidence_feats = tree.map_structure_with_path(_inmap, confidence_feats)
            outputs["confidence"] = self.confidence(post_recycle_outputs["S_inputs_I"], post_recycle_outputs["S_I"], post_recycle_outputs["Z_II"], X_L["X_L"], confidence_feats["rf2aa_seq"], confidence_feats["rep_atom_idx"], frame_atom_idxs=confidence_feats["frame_atom_idxs"])
            #add the chain label so we can calculate ipae later
            outputs["confidence"]["chain_iid_token_lvl"] = confidence_feats["chain_iid_token_lvl"]
            outputs["confidence"]["is_real_atom"] = confidence_feats["is_real_atom"]
            outputs["confidence"]["rf2aa_seq"] = confidence_feats["rf2aa_seq"]

        return outputs

    def construct_noise_schedule(self, num_timesteps, min_t, max_t):
        t = torch.linspace(min_t, max_t, num_timesteps)
        sigma_data = 16
        s_min = 4e-4
        s_max = 160
        p = 7
        t_hat = sigma_data * ((s_max)**(1/p) + t*(s_min**(1/p) - s_max**(1/p)))**p
        return t_hat

    def sample_diffusion(self, f, s_inputs_I, s_trunk_I, Z_trunk_II, noise_schedule, \
                         gamma_0=0.8, gamma_min=1.0, noise_scale=1.003, step_scale=1.5, D=None):
        D = D if D is not None else self.config.dataset_params["diffusion_batch_size_valid"]
        L = f["ref_pos"].shape[0]
        X_L = self._get_initial_structure(f, noise_schedule, D, L, self.device)
        X_noisy_L_traj = []
        X_denoised_L_traj = []  
        t_hats = []
        for c_t_minus_1, c_t in zip(noise_schedule, noise_schedule[1:]):
            X_exists_L = torch.ones((D, L)).bool()
            s_trans = 1.0
            X_L = centre_random_augmentation(X_L, X_exists_L, s_trans)
            gamma = gamma_0 if c_t > gamma_min else 0

            t_hat = c_t_minus_1 * (gamma + 1)

            epsilon_L = noise_scale * torch.sqrt(
                torch.square(t_hat) - torch.square(c_t_minus_1)
            ) * torch.normal(
                mean=0.0, std=1.0, size=X_L.shape, device=X_L.device
            )
            X_noisy_L = X_L + epsilon_L
            X_denoised_L = self.model.diffusion_module(X_noisy_L, t_hat.tile(D), f, s_inputs_I, s_trunk_I, Z_trunk_II)
            delta_L =  (X_noisy_L - X_denoised_L) / t_hat
            d_t = c_t - t_hat
            X_L = X_noisy_L + step_scale * d_t * delta_L

            X_noisy_L_traj.append(X_noisy_L)
            X_denoised_L_traj.append(X_denoised_L)
            t_hats.append(t_hat)
        return dict(
            X_L= X_L,
            X_noisy_L_traj= X_noisy_L_traj,
            X_denoised_L_traj= X_denoised_L_traj,
            t_hats= t_hats
        )

    def sample_diffusion_simple(self, f, s_inputs_I, s_trunk_I, Z_trunk_II, noise_schedule, D=None):
        D = D if D is not None else self.config.dataset_params["diffusion_batch_size_valid"]
        L = f["ref_pos"].shape[0]
        X_L = self._get_initial_structure(f, noise_schedule, D, L, self.device)
        X_noisy_L_traj = []
        X_denoised_L_traj = []  
        t_hats = []
        for c_t_minus_1, c_t in zip(noise_schedule, noise_schedule[1:]):
            X_exists_L = torch.ones((D, L)).bool()
            s_trans = 1.0
            X_L = centre_random_augmentation(X_L, X_exists_L, s_trans)

            if (c_t_minus_1 != noise_schedule[0]):
                epsilon_L = c_t_minus_1 * torch.normal(
                    mean=0.0, std=1.0, size=X_L.shape, device=X_L.device
                )
                X_noisy_L = X_L + epsilon_L
            else:
                X_noisy_L = X_L
            X_denoised_L = self.model.diffusion_module(X_noisy_L, c_t_minus_1.tile(D), f, s_inputs_I, s_trunk_I, Z_trunk_II)
            X_L = X_denoised_L

            X_noisy_L_traj.append(X_noisy_L)
            X_denoised_L_traj.append(X_denoised_L)
            t_hats.append(c_t_minus_1)
        return dict(
            X_L= X_L,
            X_noisy_L_traj= X_noisy_L_traj,
            X_denoised_L_traj= X_denoised_L_traj,
            t_hats= t_hats
        )

    def sample_diffusion_euler(self, f, s_inputs_I, s_trunk_I, Z_trunk_II, noise_schedule, D=None):
        D = D if D is not None else self.config.dataset_params["diffusion_batch_size_valid"]
        L = f["ref_pos"].shape[0]
        X_L = self._get_initial_structure(f, noise_schedule, D, L, self.device)
        X_noisy_L_traj = []
        X_denoised_L_traj = []  
        t_hats = []
        for c_t_minus_1, c_t in zip(noise_schedule, noise_schedule[1:]):
            X_exists_L = torch.ones((D, L)).bool()
            s_trans = 1.0
            X_L = centre_random_augmentation(X_L, X_exists_L, s_trans)

            X_noisy_L = X_L
            X_denoised_L = self.model.diffusion_module(X_noisy_L, c_t_minus_1.tile(D), f, s_inputs_I, s_trunk_I, Z_trunk_II)
            delta_L =  (X_noisy_L - X_denoised_L) / c_t_minus_1
            d_t = c_t - c_t_minus_1
            X_L = X_noisy_L + d_t * delta_L

            X_noisy_L_traj.append(X_noisy_L)
            X_denoised_L_traj.append(X_denoised_L)
            t_hats.append(c_t_minus_1)
        return dict(
            X_L= X_L,
            X_noisy_L_traj= X_noisy_L_traj,
            X_denoised_L_traj= X_denoised_L_traj,
            t_hats= t_hats
        )

    def sample_diffusion_heun(self, f, s_inputs_I, s_trunk_I, Z_trunk_II, noise_schedule, D=None):
        D = D if D is not None else self.config.dataset_params["diffusion_batch_size_valid"]
        L = f["ref_pos"].shape[0]
        X_L = self._get_initial_structure(f, noise_schedule, D, L, self.device)
        X_noisy_L_traj = []
        X_denoised_L_traj = []  
        t_hats = []
        for c_t_minus_1, c_t in zip(noise_schedule, noise_schedule[1:]):
            X_exists_L = torch.ones((D, L)).bool()
            s_trans = 1.0
            X_L = centre_random_augmentation(X_L, X_exists_L, s_trans)

            X_noisy_L = X_L
            d_t = c_t - c_t_minus_1

            X_denoised_L1 = self.model.diffusion_module(X_noisy_L, c_t_minus_1.tile(D), f, s_inputs_I, s_trunk_I, Z_trunk_II)
            delta_L1 =  (X_noisy_L - X_denoised_L1) / c_t_minus_1

            if (c_t != noise_schedule[-1]):
                print (c_t_minus_1,'->',c_t,'2nd order correction')
                X_L1 = X_noisy_L + d_t * delta_L1
                X_denoised_L2 = self.model.diffusion_module(X_L1, c_t.tile(D), f, s_inputs_I, s_trunk_I, Z_trunk_II)
                delta_L2 =  (X_L1 - X_denoised_L2) / c_t
                X_L = X_noisy_L + d_t/2 * (delta_L1 + delta_L2)
            else:
                print (c_t_minus_1,'->',c_t,'1st order only')
                X_L = X_noisy_L + d_t * delta_L1
                X_denoised_L2 = X_denoised_L1

            X_noisy_L_traj.append(X_noisy_L)
            X_denoised_L_traj.append(X_denoised_L2)
            t_hats.append(c_t_minus_1)
        return dict(
            X_L= X_L,
            X_noisy_L_traj= X_noisy_L_traj,
            X_denoised_L_traj= X_denoised_L_traj,
            t_hats= t_hats
        )

    def _get_initial_structure(self, f, noise_schedule, D, L, device):
        X_L = noise_schedule[0] * torch.normal(mean=0.0, std=1.0, size=(D, L, 3), device=device)
        return X_L 

    def _get_network_input(self, inputs, t=None):
        #network_input, loss_input = prepare_input_af3(inputs, **self.config.af3_data_prep, device="cpu")
        example = inputs[0]
        network_input = dict(
            f = example["feats"],
            X_noisy_L= None,
            t = None,
        )
        return network_input
    
    def _get_confidence_feats(self, inputs):
        example = inputs[0]
        confidence_feats = dict(
            rf2aa_seq = example["confidence_feats"]["rf2aa_seq"],
            frame_atom_idxs = example["confidence_feats"]["pae_frame_idx_token_lvl_from_atom_lvl"],
            rep_atom_idx = example["ground_truth"]["rep_atom_idxs"],
            chain_iid_token_lvl = example["ground_truth"]["chain_iid_token_lvl"],
            is_real_atom = example["confidence_feats"]["is_real_atom"]
        )
        return confidence_feats
    
class AF3PartialSampler(AF3Sampler):

    def __init__(self, config, model):
        super().__init__(config, model)
        self.partial_t = config.af3_data_prep["partial_t"]

    def _get_initial_structure(self, f, noise_schedule, D, L, device):
        noise = torch.normal(mean=0.0, std=noise_schedule[0], size=(D, L, 3), device=device)
        X_L = f["xyz_guess"] + noise 
        #+ noise_schedule[0] * torch.normal(mean=0.0, std=1.0, size=(D, L, 3), device=device)

        return X_L
    
    def construct_noise_schedule(self, num_timesteps, min_t, max_t):
        full_schedule_min_t = 0
        full_schedule_max_t = 1
        full_noise_schedule = super().construct_noise_schedule(num_timesteps, full_schedule_min_t, full_schedule_max_t)
        assert self.partial_t < num_timesteps
        return full_noise_schedule[self.partial_t:]
    
    def _get_network_input(self, inputs):
        #network_input, loss_input = prepare_input_af3(inputs, **self.config.af3_data_prep, device="cpu")
        example = inputs[0]
        network_input = dict(
            f = example["feats"],
            X_noisy_L= None,
            t = None,
        )
        return network_input

