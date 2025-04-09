from typing import Any

import numpy as np
import torch

from modelhub.alignment import weighted_rigid_align
from modelhub.metrics.base import Metric


class BackboneMetrics(Metric):

    def __init__(self):
        super().__init__()
        self.clash_threshold = 1.0
        self.float_threshold = 3.0  # maximum closest-neighbour distance before considered a floating atom 
        self.standard_ca_dist = 3.8

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {
            "X_L": ("network_output", "X_L"), # [D, L, 3]
            "tok_idx": ("network_input", "f", "atom_to_token_map"),
            'f': ('network_input', 'f'),
            # Available keys:
            # "network_input": ("network_input",), # dict_keys(['X_noisy_L', 't', 'f'])
            # "network_output": ("network_output",), # dict_keys(['X_L', 'X_noisy_L_traj', 'X_denoised_L_traj', 't_hats'])
            # 'extra_info': ('extra_info',),  # dict_keys(['X_gt_L', 'crd_mask_L', 'X_rep_atoms_I', 'crd_mask_rep_atoms_I', 
            # 'chain_iid_token_lvl', 'example_id', 'extra_info', 'ground_truth_atom_array'])
            # network_input['f'].keys() = dict_keys(['residue_index', 'token_index', 'asym_id', 
            # 'entity_id', 'sym_id', 'restype', 'is_protein', 'is_rna', 'is_dna', 'is_ligand', 
            # 'is_backbone', 'is_sidechain', 'is_virtual', 'is_central', 'ref_pos', 'ref_mask', 
            # 'ref_element', 'ref_charge', 'ref_atom_name_chars', 'ref_space_uid', 'ref_automorphs', 
            # 'ref_automorphs_mask', 'ref_pos_is_ground_truth', 'atom_to_token_map', 'token_bonds'])
        }
    
    def compute(self, X_L, tok_idx, f):
        o = {}
        # o['batch_size'] = X_L.shape[0]
        
        xyz = X_L.detach().cpu().numpy()
        tok_idx = tok_idx.cpu().numpy()
        dists = np.linalg.norm(xyz[..., :, None, :] - xyz[..., None, :, :], axis=-1)  # N_atoms x N_atoms
        
        is_protein = f['is_protein'][tok_idx].cpu().numpy()  # n_atoms

        mask = np.zeros_like(dists, dtype=bool)
        mask = mask |  (np.eye(dists.shape[-1], dtype=bool) )[None]
        mask = mask |  (tok_idx[:, None] == tok_idx[None, :])[None]
        mask = mask | ~(is_protein[:, None] & is_protein[None, :])[None]
        dists[mask] = 999

        num_clashes_L = (dists.min(axis=-1) < self.clash_threshold).astype(float)  # B, L
        o['frac_clashing'] = float(num_clashes_L.mean(-1).mean())
        o['n_clashing'] = float(num_clashes_L.sum(-1).mean())

        if 'is_backbone' in f:
            is_backbone = f['is_backbone'].cpu().numpy()
            mask = np.zeros_like(dists, dtype=bool)
            mask = mask |  (tok_idx[:, None] == tok_idx[None, :])[None]
            mask = mask | ~(is_backbone[:, None] & is_backbone[None, :])[None]
            dists[mask] = 999
            o['frac_backbone_clashing'] = float((dists.min(axis=-1) < self.clash_threshold).astype(float).mean(-1).mean())
            o['n_backbone_clashing'] = float((dists.min(axis=-1) < self.clash_threshold).astype(float).sum(-1).mean())

        # Num floating
        dists = np.linalg.norm(xyz[..., :, None, :] - xyz[..., None, :, :], axis=-1)  # N_atoms x N_atoms
        mask = np.zeros_like(dists, dtype=bool) 
        mask = mask & np.eye(dists.shape[-1], dtype=bool)[None]
        dists[mask] = 999

        is_floating = dists.min(axis=-1) > self.float_threshold
        o['frac_floating'] = float(is_floating.mean(-1).mean())
        
        if 'is_ca' in f:
            # Calculate CA
            idx_mask = f['is_ca'].cpu() & f['is_protein'][tok_idx].cpu()
            xyz = X_L.cpu()[:, idx_mask]
            
            ca_dists = torch.norm(xyz[:, 1:] - xyz[:, :-1], dim=-1)
            deviation = torch.abs(ca_dists - self.standard_ca_dist)  # B, (I-1)
            is_chainbreak = deviation > 1.0

            o['max_ca_deviation'] = float(deviation.max(-1).values.mean())
            o['fraction_chainbreaks'] = float(is_chainbreak.float().mean(-1).mean()) 
            o['n_chainbreaks'] = float(is_chainbreak.float().sum(-1).mean())

        return o

# class MotifMetrics(BackboneMetrics):
#     def compute(self, X_L, tok_idx, f):
#         o = {}
#         if 'is_masked_token' in f:
#             # Calculate loss on the unmasked tokens
#             is_unmasked = ~f['is_masked_token'][tok_idx].cpu().numpy()
#             if np.sum(is_unmasked) > 0:
#                 xyz_unmasked = X_L.cpu().numpy()[:, is_unmasked]  # B, N_unmasked, 3
#                 xyz_input = f['gt_pos'][is_unmasked].cpu()
#                 if not len(xyz_input.shape) == len(xyz_unmasked.shape):
#                     xyz_input = xyz_input.unsqueeze(1).expand(xyz_unmasked.shape[0], xyz_unmasked.shape[1], 3)

#                 # Align
#                 xyz_input = weighted_rigid_align(xyz_unmasked, xyz_input)

#                 rmsd = np.sqrt(np.mean(np.linalg.norm(xyz_unmasked - xyz_input, axis=-1), axis=-1))  # B, 
#                 o['rmsd_to_unmasked_pos'] = float(rmsd.mean())  # Average RMSD of unmasked positions
#                 print("RMSD:", rmsd)
#         return o