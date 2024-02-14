import torch

from rf2aa.kinematics import xyz_to_t2d
from rf2aa.symmetry import symm_subunit_matrix, find_symm_subs
from rf2aa.util import  is_atom, \
    Ls_from_same_chain_2d, xyz_t_to_frame_xyz, get_prot_sm_mask
from rf2aa.chemical import ChemicalData as ChemData



def prepare_input(inputs, xyz_converter, gpu):
        (
            seq, msa, msa_masked, msa_full, mask_msa, true_crds, mask_crds, idx_pdb, 
            xyz_t, t1d, mask_t, xyz_prev, mask_prev, same_chain, unclamp, negative, 
            atom_frames, bond_feats, dist_matrix, chirals, ch_label, symmgp, task, item
        ) = inputs

        # transfer inputs to device
        B, _, N, L = msa.shape

        idx_pdb = idx_pdb.to(gpu, non_blocking=True) # (B, L)
        true_crds = true_crds.to(gpu, non_blocking=True) # (B, L, 27, 3)
        mask_crds = mask_crds.to(gpu, non_blocking=True) # (B, L, 27)
        same_chain = same_chain.to(gpu, non_blocking=True)

        xyz_t = xyz_t.to(gpu, non_blocking=True)
        t1d = t1d.to(gpu, non_blocking=True)
        mask_t = mask_t.to(gpu, non_blocking=True)
        
        #fd --- use black hole initialization
        xyz_prev = ChemData().INIT_CRDS.reshape(1,1,ChemData().NTOTAL,3).repeat(1,L,1,1).to(gpu, non_blocking=True)
        mask_prev = torch.zeros((1,L,ChemData().NTOTAL), dtype=torch.bool).to(gpu, non_blocking=True)

        atom_frames = atom_frames.to(gpu, non_blocking=True)
        bond_feats = bond_feats.to(gpu, non_blocking=True)
        dist_matrix = dist_matrix.to(gpu, non_blocking=True)
        chirals = chirals.to(gpu, non_blocking=True)
        assert (len(symmgp)==1)
        symmgp = symmgp[0]

        # symmetry - reprocess (many) inputs
        if (symmgp != 'C1'):
            if (symmgp[0]=='C'):
                Osub = min(3, int(symmgp[1:]))
            elif (symmgp[0]=='D'):
                Osub = min(5, 2*int(symmgp[1:]))
            else:
                Osub = 6
            Lasu = L//Osub

            # load symm data from symmetry group
            symmids, symmRs, symmmeta, symmoffset = symm_subunit_matrix(symmgp)
            symmids = symmids.to(gpu, non_blocking=True)
            symmRs = symmRs.to(gpu, non_blocking=True)
            symmoffset = symmoffset.to(gpu, non_blocking=True)
            symmmeta = (
                [x.to(gpu, non_blocking=True) for x in symmmeta[0]],
                symmmeta[1])
            O = symmids.shape[0]

            # offset initial model away from symmetry center
            xyz_prev = xyz_prev + symmoffset*Lasu**(1/3)

            # find contacting subunits
            xyz_prev, symmsub = find_symm_subs(xyz_prev[:,:Lasu], symmRs, symmmeta)
            symmsub = symmsub.to(gpu, non_blocking=True)

        else:
            Lasu = L
            Osub = 1
            symmids = None
            symmsub = None
            symmRs = None
            symmmeta = None

        # processing template features
        mask_t_2d = get_prot_sm_mask(mask_t, seq[0][0]) 
        mask_t_2d = mask_t_2d[:,:,None]*mask_t_2d[:,:,:,None] # (B, T, L, L)

        # we can provide sm_templates so we want to allow interchain templates bw protein chain 1 and sms
        # specifically the templates are found for the query protein chain
        Ls = Ls_from_same_chain_2d(same_chain)
        prot_ch1_to_sm_2d = torch.zeros_like(same_chain) 
        prot_ch1_to_sm_2d[:, :Ls[0], is_atom(seq)[0][0]] = 1
        prot_ch1_to_sm_2d[:, is_atom(seq)[0][0], :Ls[0]] = 1

        is_possible_t2d = same_chain.clone()
        is_possible_t2d[prot_ch1_to_sm_2d.bool()] = 1

        mask_t_2d = mask_t_2d.float() * is_possible_t2d.float()[:,None] # (ignore inter-chain region between proteins)
        xyz_t_frame = xyz_t_to_frame_xyz(xyz_t, msa[:, 0,0], atom_frames)
        t2d = xyz_to_t2d(xyz_t_frame, mask_t_2d)

        # get torsion angles from templates
        seq_tmp = t1d[...,:-1].argmax(dim=-1).reshape(-1,Lasu*Osub)
        alpha, _, alpha_mask, _ = xyz_converter.get_torsions(xyz_t.reshape(-1,Lasu*Osub,ChemData().NTOTAL,3), seq_tmp, mask_in=mask_t.reshape(-1,Lasu*Osub,ChemData().NTOTAL))
        alpha = alpha.reshape(B,-1,Lasu*Osub,ChemData().NTOTALDOFS,2)
        alpha_mask = alpha_mask.reshape(B,-1,Lasu*Osub,ChemData().NTOTALDOFS,1)
        alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(B, -1, Lasu*Osub, 3*ChemData().NTOTALDOFS)
        alpha_prev = torch.zeros((B,Lasu*Osub,ChemData().NTOTALDOFS,2))

        network_input = {}
        network_input['msa_latent'] = msa_masked
        network_input['msa_full'] = msa_full
        network_input['seq'] = seq
        network_input['seq_unmasked'] = msa[:,0,0]
        network_input['idx'] = idx_pdb
        network_input['t1d'] = t1d
        network_input['t2d'] = t2d
        network_input['xyz_t'] = xyz_t[:,:,:,1]
        network_input['alpha_t'] = alpha_t
        network_input['mask_t'] = mask_t_2d
        network_input['same_chain'] = same_chain
        network_input['bond_feats'] = bond_feats
        network_input['dist_matrix'] = dist_matrix

        network_input['chirals'] = chirals
        network_input['atom_frames'] = atom_frames

        network_input['symmids'] = symmids
        network_input['symmsub'] = symmsub
        network_input['symmRs'] = symmRs
        network_input['symmmeta'] = symmmeta

        network_input["xyz_prev"] = xyz_prev
        network_input["alpha_prev"] = alpha_prev
        network_input["mask_recycle"] = None
        return task, item, network_input, true_crds, mask_crds, msa, mask_msa, unclamp, negative, symmRs, Lasu, ch_label


def get_loss_calc_items(inputs,device="cpu"):
    (
        seq, msa, msa_masked, msa_full, mask_msa, true_crds, mask_crds, idx_pdb, 
        xyz_t, t1d, mask_t, xyz_prev, mask_prev, same_chain, unclamp, negative, 
        atom_frames, bond_feats, dist_matrix, chirals, ch_label, symmgp, task, item
    ) = inputs

    return seq.to(device), same_chain.to(device), idx_pdb.to(device), bond_feats.to(device), dist_matrix.to(device), atom_frames.to(device) 