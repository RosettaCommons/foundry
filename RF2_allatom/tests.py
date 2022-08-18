import unittest
import torch
from torch.utils import data

# from chemical import NFRAMES
from data_loader import get_train_valid_set, Dataset, DatasetNAComplex, DatasetRNA, DatasetSMComplex, loader_pdb, loader_na_complex, loader_rna, loader_sm_compl,set_data_loader_params
from kinematics import xyz_to_c6d, xyz_to_t2d
from chemical import num2aa, aa2elt, aa2num
from loss import compute_general_FAPE, resolve_equiv_natives, calc_str_loss
from util import get_frames, frame_indices, is_atom, xyz_to_frame_xyz, xyz_t_to_frame_xyz, long2alt, writepdb

class LossTestCase(unittest.TestCase):

	def setUp(self):
		self.loader_param = set_data_loader_params({})
		(
			pdb_items, fb_items, compl_items, neg_items, na_compl_items, na_neg_items, rna_items,
			sm_compl_items, valid_pdb, valid_homo, valid_compl, valid_neg, valid_na_compl, 
			valid_na_neg, valid_rna, valid_sm_compl, homo
		) = get_train_valid_set(self.loader_param)

		pdb_IDs, pdb_weights, pdb_dict = pdb_items
		na_compl_IDs, na_compl_weights, na_compl_dict = na_compl_items
		rna_IDs, rna_weights, rna_dict = rna_items
		sm_compl_IDs, sm_compl_weights, sm_compl_dict = sm_compl_items
		self.homo = homo

		valid_pdb_set = Dataset(
			list(valid_pdb.keys()),
			loader_pdb, valid_pdb,
			self.loader_param, homo, p_homo_cut=-1.0
		)
		valid_na_compl_set = DatasetNAComplex(
			list(valid_na_compl.keys()),
			loader_na_complex, valid_na_compl,
			self.loader_param, negative=False, native_NA_frac=1.0
		)
		valid_sm_compl_set = DatasetSMComplex(
			list(sm_compl_dict.keys()),
			loader_sm_compl, sm_compl_dict,
			self.loader_param
		)
		self.valid_pdb_loader = data.DataLoader(valid_pdb_set)
		self.valid_na_compl_loader = data.DataLoader(valid_na_compl_set)
		self.valid_sm_compl_loader = data.DataLoader(valid_sm_compl_set)

	def test_compute_general_FAPE(self):
		with self.subTest("test that FAPE loss is correctly calculated for proteins"):
			for seq, msa, msa_masked, msa_full, mask_msa, true_crds, atom_mask, idx_pdb, xyz_t, t1d, xyz_prev, same_chain, unclamp, negative, atom_frames, bond_feats in self.valid_pdb_loader:
				# first assert that same structure gives you 0 loss

				frames, frame_mask = get_frames(
					true_crds, atom_mask, msa[:, 0, 0], frame_indices, atom_frames
					)

				l_fape = compute_general_FAPE(
					true_crds,
					true_crds,
					atom_mask,
					frames,
					frame_mask
				)
				self.assertAlmostEqual(int(l_fape.numpy()),0)
				fapes = []
				for i in range(5):
					perturbed_crds = true_crds+(torch.rand(true_crds.shape)*(i+1))
					l_fape = compute_general_FAPE(
						perturbed_crds,
						true_crds,
						atom_mask,
						frames,
						frame_mask
					)
					fapes.append(l_fape)
				for i in range(1,5):
					self.assertLess(fapes[i-1], fapes[i])
				break
				#add noise and make sure increasing noise increases loss
		with self.subTest("test that FAPE loss is correctly calculated for protein/NA complexes"):
			for seq, msa, msa_masked, msa_full, mask_msa, true_crds, atom_mask, idx_pdb, xyz_t, t1d, xyz_prev, same_chain, unclamp, negative, atom_frames, bond_feats in self.valid_na_compl_loader:
				# first assert that same structure gives you 0 loss

				frames, frame_mask = get_frames(
					true_crds, atom_mask, msa[:, 0, 0], frame_indices, atom_frames
					)

				l_fape = compute_general_FAPE(
					true_crds,
					true_crds,
					atom_mask,
					frames,
					frame_mask
				)
				self.assertAlmostEqual(int(l_fape.numpy()),0)
				fapes = []
				for i in range(5):
					perturbed_crds = true_crds+(torch.rand(true_crds.shape)*(i+1))
					l_fape = compute_general_FAPE(
						perturbed_crds,
						true_crds,
						atom_mask,
						frames,
						frame_mask
					)
					fapes.append(l_fape)
				for i in range(1,5):
					self.assertLess(fapes[i-1], fapes[i])
				break
		with self.subTest("test that FAPE loss is correctly calculated for protein/SM complexes"):
			for seq, msa, msa_masked, msa_full, mask_msa, true_crds, atom_mask, idx_pdb, xyz_t, t1d, xyz_prev, same_chain, unclamp, negative, atom_frames, bond_feats in self.valid_sm_compl_loader:
				# first assert that same structure gives you 0 loss
				true_crds, atom_mask = resolve_equiv_natives(true_crds[0, 0].unsqueeze(0), true_crds, atom_mask)
				frames, frame_mask = get_frames(
					true_crds, atom_mask, msa[:, 0, 0], frame_indices, atom_frames
					)
				l_fape = compute_general_FAPE(
					true_crds,
					true_crds,
					atom_mask,
					frames,
					frame_mask
				)
				self.assertAlmostEqual(int(l_fape.numpy()),0)

				fapes = []
				for i in range(5):
					perturbed_crds = true_crds+(torch.rand(true_crds.shape)*(i+1))
					l_fape = compute_general_FAPE(
						perturbed_crds,
						true_crds,
						atom_mask,
						frames,
						frame_mask
					)
					fapes.append(l_fape)
				for i in range(1,5):
					self.assertLess(fapes[i-1], fapes[i])
				break
		with self.subTest("test that protein backbone FAPE loss can be calculated with compute_general_FAPE"):
			for seq, msa, msa_masked, msa_full, mask_msa, true_crds, atom_mask, idx_pdb, xyz_t, t1d, xyz_prev, same_chain, unclamp, negative, atom_frames, bond_feats in self.valid_pdb_loader:
				frames, frame_mask = get_frames(
					true_crds, atom_mask, msa[:, 0, 0], frame_indices, atom_frames
					)
				frame_mask[...,1:] = False

				l_fape = compute_general_FAPE(
					true_crds,
					true_crds,
					atom_mask,
					frames,
					frame_mask
				)
				self.assertAlmostEqual(int(l_fape.numpy()),0)

				res_mask = ~((atom_mask[:,:,:3].sum(dim=-1) < 3.0) * ~(is_atom(msa[:,0,0])))
				mask_2d = res_mask[:,None,:] * res_mask[:,:,None]

				fapes = []
				for i in range(5):
					perturbed_crds = true_crds+(torch.rand(true_crds.shape)*(i+1))
					l_fape = compute_general_FAPE(
						perturbed_crds,
						true_crds,
						atom_mask,
						frames,
						frame_mask
					)

					fapes.append(l_fape)

					tot_str, str_loss = calc_str_loss(perturbed_crds.unsqueeze(0), true_crds, mask_2d, same_chain, negative=False,
											  A=10.0, d_clamp_intra=10.0, d_clamp_inter=10.0, gamma=1.0, eps=1e-4)
					self.assertAlmostEqual(int(l_fape.numpy()), int(tot_str.numpy()))
				for i in range(1,5):
					self.assertLess(fapes[i-1], fapes[i])
				break
		with self.subTest("test that FAPE loss over only the atoms can be calculated"):
			for seq, msa, msa_masked, msa_full, mask_msa, true_crds, atom_mask, idx_pdb, xyz_t, t1d, xyz_prev, same_chain, unclamp, negative, atom_frames, bond_feats in self.valid_sm_compl_loader:
				label_aa_s = msa[:, 0]
				seq = label_aa_s[:,0].clone()
				true_crds, atom_mask = resolve_equiv_natives(true_crds[0, 0].unsqueeze(0), true_crds, atom_mask)
				frames, frame_mask = get_frames(
					true_crds, atom_mask, seq, frame_indices, atom_frames
				)

				rotation_mask = is_atom(seq)
				atom_fape = compute_general_FAPE(
						true_crds[:,rotation_mask[0],:,:3],
						true_crds[:,rotation_mask[0],:,:3],
						atom_mask[:,rotation_mask[0]],
						frames[:,rotation_mask[0]],
						frame_mask[:,rotation_mask[0]]
					)
				self.assertAlmostEqual(int(atom_fape.numpy()),0)

	def test_get_frames(self):
		"""test that nodes in atom frames are relatively close to each other (because they should be bonded)"""
		NFRAMES = 8
		for seq, msa, msa_masked, msa_full, mask_msa, true_crds, atom_mask, idx_pdb, xyz_t, t1d, xyz_prev, same_chain, unclamp, negative, atom_frames, bond_feats in self.valid_sm_compl_loader:
			true_crds, atom_mask = resolve_equiv_natives(true_crds[0, 0].unsqueeze(0), true_crds, atom_mask)
			frames, frame_mask = get_frames(
				true_crds, atom_mask, msa[:, 0, 0], frame_indices, atom_frames
			)
			N, L, natoms, _ = true_crds.shape
			# flatten middle dims so can gather across residues
			X_prime = true_crds.reshape(N, L*natoms, -1, 3).repeat(1,1,NFRAMES,1)

			frames_reindex = torch.zeros(frames.shape[:-1])
			for i in range(L):
				frames_reindex[:, i, :, :] = (i+frames[..., i, :, :, 0])*natoms + frames[..., i, :, :, 1]
			frames_reindex = frames_reindex.long()

			frame_mask *= torch.all(
				torch.gather(atom_mask.reshape(1, L*natoms),1,frames_reindex.reshape(1,L*NFRAMES*3)).reshape(1,L,-1,3),
				axis=-1)

			X_x = torch.gather(X_prime, 1, frames_reindex[...,0:1].repeat(N,1,1,3))
			X_y = torch.gather(X_prime, 1, frames_reindex[...,1:2].repeat(N,1,1,3))
			X_z = torch.gather(X_prime, 1, frames_reindex[...,2:3].repeat(N,1,1,3))
			atoms = is_atom(msa[:, 0,0])

			frame_distance1 = torch.cdist(X_x[atoms], X_y[atoms])
			frame_distance2 = torch.cdist(X_y[atoms], X_z[atoms])
			self.assertTrue(torch.all(frame_distance1[:,0,0] <2))
			self.assertTrue(torch.all(frame_distance2[:,0,0] <2))
			break

	def test_xyz_to_c6d(self):
		for seq, msa, msa_masked, msa_full, mask_msa, true_crds, atom_mask, idx_pdb, xyz_t, t1d, xyz_prev, same_chain, unclamp, negative, atom_frames, bond_feats in self.valid_sm_compl_loader:
			true_crds, atom_mask = resolve_equiv_natives(true_crds[0, 0].unsqueeze(0), true_crds, atom_mask)
			# atoms = is_atom(msa[:, 0,0])
			# atom_crds = true_crds[atoms]
			# atom_L, natoms, _ = atom_crds.shape
			# frames_reindex = torch.zeros(atom_frames.shape[:-1])
			# for i in range(atom_L):
			# 	frames_reindex[:, i, :] = (i+atom_frames[..., i, :, 0])*natoms + atom_frames[..., i, :, 1]
			# frames_reindex = frames_reindex.long()
			# true_crds[atoms, :, :3] = atom_crds.reshape(atom_L*natoms, 3)[frames_reindex]
			true_crds = xyz_to_frame_xyz(true_crds, msa[:, 0,0], atom_frames)
			c6d, _ = xyz_to_c6d(true_crds)

			xyz_t = xyz_t_to_frame_xyz(xyz_t, msa[:, 0,0].squeeze(0), atom_frames)
			t2d = xyz_to_t2d(xyz_t)
			break

	def test_res_mask(self):
		"""updated res_mask to not mask "atom" nodes that only have one backbone atom filled in """
		for seq, msa, msa_masked, msa_full, mask_msa, true_crds, atom_mask, idx_pdb, xyz_t, t1d, xyz_prev, same_chain, unclamp, negative, atom_frames, bond_feats in self.valid_sm_compl_loader:
			true_crds, atom_mask = resolve_equiv_natives(true_crds[0, 0].unsqueeze(0), true_crds, atom_mask)
			res_mask = ~((atom_mask[:,:,:3].sum(dim=-1) < 3.0) * ~(is_atom(msa[:,0,0])))
			B, L = true_crds.shape[:2]
			self.assertEqual(res_mask.shape[0], B)
			self.assertEqual(res_mask.shape[1], L)
			print(is_atom(msa[:,0,0]))
			print(res_mask)
			print(true_crds[res_mask][:,:23])
			break
	
	def test_resolve_equiv_natives(self):
		""" test that resolve_equiv_natives works"""
		for seq, msa, msa_masked, msa_full, mask_msa, true_crds, atom_mask, idx_pdb, xyz_t, t1d, xyz_prev, same_chain, unclamp, negative, atom_frames, bond_feats in self.valid_sm_compl_loader:
			true_crds, atom_mask = resolve_equiv_natives(torch.randn(true_crds[0, 0].unsqueeze(0).shape), true_crds, atom_mask)
			print(true_crds)
			break

class DataLoaderTestCase(unittest.TestCase):

	def setUp(self) -> None:
		super().setUp()
		self.loader_param = set_data_loader_params({})
		(
			pdb_items, fb_items, compl_items, neg_items, na_compl_items, na_neg_items, rna_items,
			sm_compl_items, valid_pdb, valid_homo, valid_compl, valid_neg, valid_na_compl, 
			valid_na_neg, valid_rna, valid_sm_compl, homo
		) = get_train_valid_set(self.loader_param)

		pdb_IDs, pdb_weights, pdb_dict = pdb_items
		na_compl_IDs, na_compl_weights, na_compl_dict = na_compl_items
		rna_IDs, rna_weights, rna_dict = rna_items
		sm_compl_IDs, sm_compl_weights, sm_compl_dict = sm_compl_items
		self.homo = homo

		valid_pdb_set = Dataset(
			list(valid_pdb.keys()),
			loader_pdb, valid_pdb,
			self.loader_param, homo, p_homo_cut=-1.0
		)
		valid_sm_compl_set = DatasetSMComplex(
			list(sm_compl_dict.keys()),
			loader_sm_compl, sm_compl_dict,
			self.loader_param
		)
		self.valid_pdb_loader = data.DataLoader(valid_pdb_set)
		self.valid_sm_compl_loader = data.DataLoader(valid_sm_compl_set)

	def test_vaporize_protein(self):
		for seq, msa, msa_masked, msa_full, mask_msa, true_crds, atom_mask, idx_pdb, xyz_t, t1d, xyz_prev, same_chain, unclamp, negative, atom_frames, bond_feats in self.valid_pdb_loader:
			B, L = msa[:, 0, 0].shape
			i = torch.randint(2, L-3,(1,))
			residues_atomize = msa[:, 0, 0, i-2:i+2]
			residues_atomize = [aa2elt[num][:14] for num in residues_atomize[0]]

			true_alt = torch.zeros_like(true_crds)
			true_alt.scatter_(2, long2alt[msa[:, 0, 0],:,None].repeat(1,1,1,3), true_crds)
			coords_stack = torch.stack((true_crds[:, i-2:i+2], true_alt[:, i-2:i+2]), dim=0)
			swaps = (coords_stack[0] == coords_stack[1]).all(dim=2).all(dim=2).squeeze() #checks whether theres a swap at each position
			print((coords_stack[0] == coords_stack[1]).all(dim=2).all(dim=2).squeeze())
			print(coords_stack.shape)
			swaps = torch.nonzero(~swaps).squeeze() # indices with a swap
			print(swaps)
			if swaps.numel() != 0:
				combs = torch.combinations(torch.tensor([0,1]), r=swaps.numel(), with_replacement=True)
				stack = torch.stack((combs, swaps.unsqueeze(0).repeat(swaps.numel()+1,1)), dim =2)
				coords_stack = coords_stack.repeat(swaps.numel()+1,1,1,1,1).squeeze(1)
				atom_coords = true_crds.repeat(swaps.numel()+1,1,1,1)
				print(coords_stack.shape)
				true_crds[:, i-2:i+2][swaps] = coords_stack[stack[...,0], stack[...,1]]
				print(atom_coords)
			else:
				continue

			lig_seq = []
			ra = []
			for idx in range(len(residues_atomize)):
				for jdx in range(14):
					if residues_atomize[idx][jdx]:
						ra.append((i-2+idx, jdx))
						lig_seq.append(aa2num[residues_atomize[idx][jdx]])
			lig_seq = torch.tensor(lig_seq)
			ins = torch.zeros_like(lig_seq)
			print(lig_seq)
			ra = torch.tensor(ra)
			r,a = ra.T
			lig_xyz = torch.zeros((len(ra), 3))
			lig_xyz = true_crds[:, r, a]
			lig_mask = atom_mask[:, r, a]
			# print(lig_xyz)
			# print(lig_mask)
			#NEED TO FIGURE OUT XYZ_T, T1D set everything vaporized into NaN and then create new NaN features for length, set t1D into gaps
			msa = torch.cat((msa[:, :, :, :i-2], msa[:, :, :, i+2:]), dim=3)
			msa_masked = torch.cat((msa_masked[:, :, :, :i-2], msa_masked[:, :, :, i+2:]), dim=3)
			msa_full = torch.cat((msa_full[:, :, :, :i-2], msa_full[:, :, :, i+2:]), dim=3)
			mask_msa = torch.cat((mask_msa[:, :, :, :i-2], mask_msa[:, :, :, i+2:]), dim=3)

			true_crds = torch.cat((true_crds[ :, :i-2], true_crds[ :, i+2:]), dim=1)
			atom_mask = torch.cat((atom_mask[ :, :i-2], atom_mask[ :, i+2:]), dim=1)

			idx_pdb = torch.cat((idx_pdb[ :, :i-2], idx_pdb[ :, i+2:]), dim=1)
			xyz_t = torch.cat((xyz_t[ :, :, :i-2], xyz_t[:, :, i+2:]), dim=2)
			t1d = torch.cat((t1d[ :, :, :i-2], t1d[:, :, i+2:]), dim=2)
			xyz_prev = torch.cat((xyz_prev[ :, :i-2], xyz_prev[:, i+2:]), dim=1)
			same_chain = torch.cat((same_chain[ :, :i-2], same_chain[:, i+2:]), dim=1)
			same_chain = torch.cat((same_chain[ :, :, :i-2], same_chain[:, :, i+2:]), dim=2)

			bond_feats = torch.cat((bond_feats[ :, :i-2], bond_feats[:, i+2:]), dim=1)
			bond_feats = torch.cat((bond_feats[ :, :, :i-2], bond_feats[:, :, i+2:]), dim=2)


			break

	def test_writepdb(self):
		counter = 0
		for seq, msa, msa_masked, msa_full, mask_msa, true_crds, atom_mask, idx_pdb, xyz_t, t1d, xyz_prev, same_chain, unclamp, negative, atom_frames, bond_feats in self.valid_sm_compl_loader:
			print(msa.shape)
			print(true_crds[0, 0].shape)
			writepdb(f"{counter}.pdb",  true_crds[0, 0].unsqueeze(0),msa[0,0,0])
			counter = counter +1
			if counter > 20:
				break
		
if __name__ == '__main__':
	unittest.main()