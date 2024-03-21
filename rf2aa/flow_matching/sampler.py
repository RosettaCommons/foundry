import torch
import torch.nn.functional as F
from typing import Any, Dict, Tuple
from rf2aa.flow_matching.interpolant import _centered_gaussian, _uniform_so3
import rf2aa.flow_matching.data_utils as du
from rf2aa.flow_matching import data_transforms
from rf2aa.training.recycling import recycle_step_packed
from rf2aa.data.dataloader_adaptor import prepare_input_fm, construct_template_feats
from rf2aa.util import rigid_from_3_points, writepdb_file


class Sampler:
    def __init__(self, model, num_timesteps, min_t, interpolant, xyz_converter, is_training) -> None:
        self.model = model
        self.num_timesteps = num_timesteps
        self.min_t = min_t
        self.interpolant = interpolant
        self.device = self.interpolant._device
        self.xyz_converter = xyz_converter
        self.is_training = is_training

    def sample(self, inputs: Tuple[str, Any]) -> Dict[str, Any]:
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
            output_i = recycle_step_packed(self.model, network_input,1, use_amp=True, nograds=True)
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
        # 
        # only integrated to min_t, still need to take final step

        updated_features = self._construct_xt_features(rotmats_t_1, trans_t_1, network_input) 
        network_input.update(updated_features)
        # run model
        output_i = recycle_step_packed(self.model, network_input,1, use_amp=True, nograds=True)
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

        