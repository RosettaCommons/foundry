import torch
import torch.nn as nn
from collections import OrderedDict

from rf2aa.kinematics import xyz_to_c6d, c6d_to_bins
from rf2aa.loss.loss import resolve_equiv_natives, resolve_equiv_natives_asmb, \
    resolve_symmetry_predictions, resolve_symmetry, mask_unresolved_frames, \
    compute_general_FAPE, torsionAngleLoss, calc_lddt, calc_allatom_lddt_loss, \
    calc_crd_rmsd, calc_BB_bond_geom, calc_l1_clash_loss, calc_atom_bond_loss
from rf2aa.util import is_atom, is_protein, Ls_from_same_chain_2d, get_prot_sm_mask, \
    xyz_to_frame_xyz, get_frames, writepdb
from rf2aa.chemical import ChemicalData as ChemData

cce_loss = nn.CrossEntropyLoss(reduction='none')


def get_loss_and_misc(
        trainer,
        output_i, true_crds, atom_mask, same_chain,
        seq, msa, mask_msa, idx_pdb, bond_feats, dist_matrix, atom_frames, unclamp, negative, task, item, symmRs, Lasu, ch_label, 
        loss_param
    ):
    logit_s, logit_aa_s, logit_pae, logit_pde, p_bind, pred_crds, alphas, pred_allatom, pred_lddts, _, _, _ = output_i

    if pred_allatom is None:
        _, pred_allatom = trainer.xyz_converter.compute_all_atom(msa[0][0][None],pred_crds[-1], alphas[-1])
        #pred_crds = pred_crds[:, None]
        #alphas = alphas[:, None]

    if (symmRs is not None):
        ###
        # resolve symmetry
        ###
        true_crds = true_crds[:,0]
        atom_mask = atom_mask[:,0]
        mapT2P = resolve_symmetry_predictions(pred_crds, true_crds, atom_mask, Lasu) # (Nlayer, Ltrue)

        # update all derived data to only include subunits mapping to native
        logit_s_new = []
        for li in logit_s:
            li=torch.gather(li,2,mapT2P[-1][None,None,:,None].repeat(1,li.shape[1],1,li.shape[-1]))
            li=torch.gather(li,3,mapT2P[-1][None,None,None,:].repeat(1,li.shape[1],li.shape[2],1))
            logit_s_new.append(li)
        logit_s = tuple(logit_s_new)

        logit_aa_s = logit_aa_s.view(1,ChemData().NAATOKENS,msa.shape[-2],msa.shape[-1])
        logit_aa_s = torch.gather(logit_aa_s,3,mapT2P[-1][None,None,None,:].repeat(1,ChemData().NAATOKENS,logit_aa_s.shape[-2],1))
        logit_aa_s = logit_aa_s.view(1,ChemData().NAATOKENS,-1)

        msa = torch.gather(msa,2,mapT2P[-1][None,None,:].repeat(1,msa.shape[-2],1))
        mask_msa = torch.gather(mask_msa,2,mapT2P[-1][None,None,:].repeat(1,mask_msa.shape[-2],1))

        logit_pae=torch.gather(logit_pae,2,mapT2P[-1][None,None,:,None].repeat(1,logit_pae.shape[1],1,logit_pae.shape[-1]))
        logit_pae=torch.gather(logit_pae,3,mapT2P[-1][None,None,None,:].repeat(1,logit_pae.shape[1],logit_pae.shape[2],1))

        logit_pde=torch.gather(logit_pde,2,mapT2P[-1][None,None,:,None].repeat(1,logit_pde.shape[1],1,logit_pde.shape[-1]))
        logit_pde=torch.gather(logit_pde,3,mapT2P[-1][None,None,None,:].repeat(1,logit_pde.shape[1],logit_pde.shape[2],1))

        pred_crds = torch.gather(pred_crds,2,mapT2P[:,None,:,None,None].repeat(1,1,1,3,3))
        pred_allatom = torch.gather(pred_allatom,1,mapT2P[-1,None,:,None,None].repeat(1,1,ChemData().NTOTAL,3))
        alphas = torch.gather(alphas,2,mapT2P[:,None,:,None,None].repeat(1,1,1,ChemData().NTOTALDOFS,2))

        same_chain=torch.gather(same_chain,1,mapT2P[-1][None,:,None].repeat(1,1,same_chain.shape[-1]))
        same_chain=torch.gather(same_chain,2,mapT2P[-1][None,None,:].repeat(1,same_chain.shape[1],1))

        bond_feats=torch.gather(bond_feats,1,mapT2P[-1][None,:,None].repeat(1,1,bond_feats.shape[-1]))
        bond_feats=torch.gather(bond_feats,2,mapT2P[-1][None,None,:].repeat(1,bond_feats.shape[1],1))

        dist_matrix=torch.gather(dist_matrix,1,mapT2P[-1][None,:,None].repeat(1,1,dist_matrix.shape[-1]))
        dist_matrix=torch.gather(dist_matrix,2,mapT2P[-1][None,None,:].repeat(1,dist_matrix.shape[1],1))

        pred_lddts = torch.gather(pred_lddts,2,mapT2P[-1][None,None,:].repeat(1,pred_lddts.shape[-2],1))
        idx_pdb = torch.gather(idx_pdb,1,mapT2P[-1][None,:])
    elif 'sm_compl' in task[0] or 'metal_compl' in task[0]:
        sm_mask = is_atom(seq[0,0])
        Ls_prot = Ls_from_same_chain_2d(same_chain[:,~sm_mask][:,:,~sm_mask])
        Ls_sm = Ls_from_same_chain_2d(same_chain[:,sm_mask][:,:,sm_mask])

        true_crds, atom_mask = resolve_equiv_natives_asmb(
            pred_allatom, true_crds, atom_mask, ch_label, Ls_prot, Ls_sm)
    else:
        true_crds, atom_mask = resolve_equiv_natives(pred_crds[-1], true_crds, atom_mask)

    res_mask = get_prot_sm_mask(atom_mask, msa[0,0])
    mask_2d = res_mask[:,None,:] * res_mask[:,:,None]

    true_crds_frame = xyz_to_frame_xyz(true_crds, msa[:, 0], atom_frames)
    c6d = xyz_to_c6d(true_crds_frame)
    c6d = c6d_to_bins(c6d, same_chain, negative=negative)

    # contact accuray not as useful to track anymore
    #prob = self.active_fn(logit_s[0]) # distogram
    #acc_s = self.calc_acc(prob, c6d[...,0], idx_pdb, mask_2d)
    loss, loss_dict = calc_loss(
        trainer, logit_s, c6d,
        logit_aa_s, msa, mask_msa, logit_pae, logit_pde, p_bind,
        pred_crds, alphas, pred_allatom, true_crds, 
        atom_mask, res_mask, mask_2d, same_chain,
        pred_lddts, idx_pdb, bond_feats, dist_matrix,
        atom_frames=atom_frames,unclamp=unclamp, negative=negative,
        item=item, task=task, **loss_param
    )
    
    return loss, loss_dict 


def calc_loss(trainer, logit_s, label_s,
                  logit_aa_s, label_aa_s, mask_aa_s, logit_pae, logit_pde, p_bind,
                  pred, pred_tors, pred_allatom, true,
                  mask_crds, mask_BB, mask_2d, same_chain,
                  pred_lddt, idx, bond_feats, dist_matrix, atom_frames=None, unclamp=False, 
                  negative=False, interface=False,
                  w_dist=1.0, w_aa=1.0, w_str=1.0, w_inter_fape=0.0, w_lig_fape=1.0, w_lddt=1.0, 
                  w_bond=1.0, w_clash=0.0, w_atom_bond=0.0, w_skip_bond=0.0, w_rigid=0.0, w_hb=0.0, w_bind=0.0,
                  w_pae=0.0, w_pde=0.0, lj_lin=0.85, eps=1e-6, binder_loss_label_smoothing = 0.0, item=None, task=None, out_dir='./'
    ):
        gpu = pred.device

        # track losses for printing to local log and uploading to WandB
        loss_dict = OrderedDict()

        B, L, natoms = true.shape[:3]
        seq = label_aa_s[:,0].clone()

        assert (B==1) # fd - code assumes a batch size of 1

        tot_loss = 0.0
        # set up frames
        frames, frame_mask = get_frames(
            pred_allatom[-1,None,...], mask_crds, seq, trainer.fi_dev, atom_frames)

        # update frames and frames_mask to only include BB frames (have to update both for compatibility with compute_general_FAPE)
        frames_BB = frames.clone()
        frames_BB[..., 1:, :, :] = 0
        frame_mask_BB = frame_mask.clone()
        frame_mask_BB[...,1:] =False

        # c6d loss
        for i in range(4):
            loss = cce_loss(logit_s[i], label_s[...,i]) # (B, L, L)
            if i==0: # apply distogram loss to all residue pairs with valid BB atoms
                mask_2d_ = mask_2d
            else: 
                # apply anglegram loss only when both residues have valid BB frames (i.e. not metal ions, and not examples with unresolved atoms in frames)
                _, bb_frame_good = mask_unresolved_frames(frames_BB, frame_mask_BB, mask_crds) # (1, L, nframes)
                bb_frame_good = bb_frame_good[...,0] # (1,L)
                loss_mask_2d = bb_frame_good & bb_frame_good[...,None]
                mask_2d_ = mask_2d & loss_mask_2d

            if negative.item():
                # Don't compute inter-chain distogram losses
                # for negative examples.
                mask_2d_ = mask_2d_ * same_chain
            loss = (mask_2d_*loss).sum() / (mask_2d_.sum() + eps)
            tot_loss += w_dist*loss
            loss_dict[f'c6d_{i}'] = loss.detach()

        # masked token prediction loss
        loss = cce_loss(logit_aa_s, label_aa_s.reshape(B, -1))
        loss = loss * mask_aa_s.reshape(B, -1)
        loss = loss.sum() / (mask_aa_s.sum() + 1e-8)
        tot_loss += w_aa*loss
        loss_dict['aa_cce'] = loss.detach()

        # col 4: binder loss
        # only apply binding loss to complexes
        # note that this will apply loss to positive sets w/o a corresponding negative set
        #   (e.g., homomers).  Maybe want to change this?
        if "binder" in trainer.config.model.auxiliary_predictors or trainer.config.experiment.trainer =="legacy":
            if (torch.sum(same_chain==0) > 0):
                bce = torch.nn.BCELoss()
                target = torch.tensor(
                    [abs(float(not negative) - binder_loss_label_smoothing)],
                    device=p_bind.device
                )
                loss = bce(p_bind,target)
            else:
                # avoid unused parameter error
                loss = 0.0 * p_bind.sum()

            tot_loss += w_bind * loss
            loss_dict['binder_bce_loss'] = loss.detach()


        ### GENERAL LAYERS
        # Structural loss (layer-wise backbone FAPE)
        dclamp = 300.0 if unclamp else 30.0 # protein & NA FAPE distance cutoffs
        dclamp_sm, Z_sm = 4, 4  # sm mol FAPE distance cutoffs
        dclamp_prot = 10
        # residue mask for FAPE calculation only masks unresolved protein backbone atoms
        # whereas other losses also maks unresolved ligand atoms (mask_BB)
        # frames with unresolved ligand atoms are masked in compute_general_FAPE
        res_mask = ~((mask_crds[:,:,:3].sum(dim=-1) < 3.0) * ~(is_atom(seq)))

        # create 2d masks for intrachain and interchain fape calculations
        nframes = frame_mask.shape[-1]
        frame_atom_mask_2d_allatom = torch.einsum('bfn,bra->bfnra', frame_mask_BB, mask_crds).bool() # B, L, nframes, L, natoms
        frame_atom_mask_2d = frame_atom_mask_2d_allatom[:, :, :, :, :3]
        frame_atom_mask_2d_intra_allatom = frame_atom_mask_2d_allatom * same_chain[:, :,None, :, None].bool().expand(-1,-1,nframes,-1, ChemData().NTOTAL)
        frame_atom_mask_2d_intra = frame_atom_mask_2d_intra_allatom[:, :, :, :, :3]
        different_chain = ~same_chain.bool()
        frame_atom_mask_2d_inter = frame_atom_mask_2d*different_chain[:, :,None, :, None].expand(-1,-1,nframes,-1, 3)

        if task[0] in ['tf','neg_tf'] or res_mask.sum() == 0:
            tot_str = 0.0 * pred.sum(axis=(1,2,3,4))
            pae_loss = 0.0 * logit_pae.sum()
            pde_loss = 0.0 * logit_pde.sum()
        elif negative: # inter-chain fapes should be ignored for negative cases
            if logit_pae is not None:
                logit_pae = logit_pae[:,:,res_mask[0]][:,:,:,res_mask[0]]
            if logit_pde is not None:
                logit_pde = logit_pde[:,:,res_mask[0]][:,:,:,res_mask[0]]
                
            tot_str, pae_loss, pde_loss = compute_general_FAPE(
                pred[:,res_mask,:,:3],
                true[:,res_mask[0],:3],
                mask_crds[:,res_mask[0],:3],
                frames_BB[:,res_mask[0]],
                frame_mask_BB[:,res_mask[0]],
                frame_atom_mask_2d=frame_atom_mask_2d_intra[:, res_mask[0]][:, :, :, res_mask[0]],
                dclamp=dclamp,
                logit_pae=logit_pae,
                logit_pde=logit_pde,
            )

        else:

            if logit_pae is not None:
                logit_pae = logit_pae[:,:,res_mask[0]][:,:,:,res_mask[0]]
            if logit_pde is not None:
                logit_pde = logit_pde[:,:,res_mask[0]][:,:,:,res_mask[0]]

            # change clamp for intra protein to 10, leave rest at 30
            dclamp_2d = torch.full_like(frame_atom_mask_2d_allatom, dclamp, dtype=torch.float32)
            if not unclamp:
                is_prot = is_protein(seq) # (1,L)
                same_chain_clamp_mask = same_chain[:, :, None, :, None].bool().repeat(1,1,nframes,1, natoms)
                # zero out rows and columns with small molecules
                same_chain_clamp_mask[:, ~is_prot[0]] = 0
                same_chain_clamp_mask[:,:, :,  ~is_prot[0]] = 0 
                dclamp_2d *= ~same_chain_clamp_mask.bool()
                dclamp_2d += same_chain_clamp_mask*dclamp_prot

            tot_str, pae_loss, pde_loss = compute_general_FAPE(
                pred[:,res_mask,:,:3],
                true[:,res_mask[0],:3],
                mask_crds[:,res_mask[0],:3],
                frames_BB[:,res_mask[0]],
                frame_mask_BB[:,res_mask[0]],
                dclamp=None,
                dclamp_2d=dclamp_2d[:, res_mask[0]][:, :, :, res_mask[0],:3], 
                logit_pae=logit_pae,
                logit_pde=logit_pde,
            )

            # free up big intermediate data tensors
            del dclamp_2d
            if not unclamp:
                del same_chain_clamp_mask

        num_layers = pred.shape[0]
        gamma = 1.0 # equal weighting of fape across all layers
        w_bb_fape = torch.pow(torch.full((num_layers,), gamma, device=pred.device), torch.arange(num_layers, device=pred.device))
        w_bb_fape = torch.flip(w_bb_fape, (0,))
        w_bb_fape = w_bb_fape / w_bb_fape.sum()
        bb_l_fape = (w_bb_fape*tot_str).sum()

        tot_loss += 0.5*w_str*bb_l_fape
        for i in range(len(tot_str)):
            loss_dict[f'bb_fape_layer{i}'] = tot_str[i].detach()
        loss_dict['bb_fape_full'] = bb_l_fape.detach()

        tot_loss += w_pae*pae_loss + w_pde*pde_loss
        loss_dict['pae_loss'] = pae_loss.detach()
        loss_dict['pde_loss'] = pde_loss.detach()

        ## small-molecule ligands
        sm_res_mask = is_atom(label_aa_s[0,0])*res_mask[0] # (L,)

        ## AllAtom loss
        # get ground-truth torsion angles
        true_tors, true_tors_alt, tors_mask, tors_planar = trainer.xyz_converter.get_torsions(
            true, seq, mask_in=mask_crds)
        tors_mask *= mask_BB[...,None]

        # get alternative coordinates for ground-truth
        true_alt = torch.zeros_like(true)
        true_alt.scatter_(2, trainer.l2a[seq,:,None].repeat(1,1,1,3), true)
        natRs_all, _n0 = trainer.xyz_converter.compute_all_atom(seq, true[...,:3,:], true_tors)
        natRs_all_alt, _n1 = trainer.xyz_converter.compute_all_atom(seq, true_alt[...,:3,:], true_tors_alt)
        predTs = pred[-1,...]
        predRs_all, pred_all = trainer.xyz_converter.compute_all_atom(seq, predTs, pred_tors[-1]) 

        #  - resolve symmetry
        xs_mask = trainer.aamask[seq] # (B, L, 27)
        xs_mask[0,:,14:]=False # (ignore hydrogens except lj loss)
        xs_mask *= mask_crds # mask missing atoms & residues as well
        natRs_all_symm, nat_symm = resolve_symmetry(pred_allatom[-1], natRs_all[0], true[0], natRs_all_alt[0], true_alt[0], xs_mask[0])

        # torsion angle loss
        l_tors = torsionAngleLoss(
            pred_tors,
            true_tors,
            true_tors_alt,
            tors_mask,
            tors_planar,
            eps = 1e-10)
        tot_loss += w_str*l_tors
        loss_dict['torsion'] = l_tors.detach()

        ### FINETUNING LAYERS
        # lddts (CA)
        ca_lddt = calc_lddt(pred[:,:,:,1].detach(), true[:,:,1], mask_BB, mask_2d, same_chain, negative=negative, interface=interface)
        loss_dict['ca_lddt'] = ca_lddt[-1].detach()

        # lddts (allatom) + lddt loss
        lddt_loss, allatom_lddt = calc_allatom_lddt_loss(
            pred_allatom.detach(), nat_symm, pred_lddt, idx, mask_crds, mask_2d, same_chain, 
            negative=negative, interface=interface, N_stripe=10)
        tot_loss += w_lddt*lddt_loss
        loss_dict['lddt_loss'] = lddt_loss.detach()
        loss_dict['allatom_lddt'] = allatom_lddt[0].detach()

        # FAPE losses
        # allatom fape and torsion angle loss
        # frames, frame_mask = get_frames(
        #     pred_allatom[-1,None,...], mask_crds, seq, self.fi_dev, atom_frames)
        if task[0] in ['tf','neg_tf'] or res_mask.sum() == 0:
            l_fape = torch.zeros((pred.shape[0])).to(gpu)

        elif negative.item(): # inter-chain fapes should be ignored for negative cases
            l_fape, _, _ = compute_general_FAPE(
                pred_allatom[:,res_mask[0],:,:3],
                nat_symm[None,res_mask[0],:,:3],
                xs_mask[:,res_mask[0]],
                frames[:,res_mask[0]],
                frame_mask[:,res_mask[0]],
                frame_atom_mask_2d=frame_atom_mask_2d_intra_allatom[:, res_mask[0]][:, :, :, res_mask[0]]
            )

        else:
            l_fape, _, _ = compute_general_FAPE(
                pred_allatom[:,res_mask[0],:,:3],
                nat_symm[None,res_mask[0],:,:3],
                xs_mask[:,res_mask[0]],
                frames[:,res_mask[0]],
                frame_mask[:,res_mask[0]]
            )

        tot_loss += w_str*l_fape[0]
        loss_dict['allatom_fape'] = l_fape[0].detach()

        # rmsd loss (for logging only)
        if torch.any(mask_BB[0]):
            rmsd = calc_crd_rmsd(
                pred_allatom[:,mask_BB[0],:,:3],
                nat_symm[None,mask_BB[0],:,:3],
                xs_mask[:,mask_BB[0]]
                )
            loss_dict["rmsd"] = rmsd[0].detach()
        else:
            loss_dict["rmsd"] = torch.tensor(0, device=gpu)

        # create protein and not protein masks; not protein could include nucleic acids
        prot_mask_BB = is_protein(label_aa_s[0,0]) #*mask_BB[0] # (L,)
        not_prot_mask_BB  = ~prot_mask_BB.bool()
        xs_mask_prot, xs_mask_lig = xs_mask.clone(), xs_mask.clone()
        xs_mask_prot[:,~prot_mask_BB] = False
        xs_mask_lig[:,~not_prot_mask_BB] = False
        if torch.any(prot_mask_BB) and torch.any(mask_BB[0]):
            rmsd_prot_prot = calc_crd_rmsd(
                pred=pred_allatom[:,mask_BB[0],:,:3], true=nat_symm[None,mask_BB[0],:,:3],
                atom_mask=xs_mask_prot[:,mask_BB[0]], rmsd_mask=xs_mask_prot[:,mask_BB[0]]
            )
        else:
            rmsd_prot_prot = torch.tensor([0], device=pred.device)
        if torch.any(not_prot_mask_BB) and torch.any(mask_BB[0]):
            rmsd_lig_lig = calc_crd_rmsd(
                pred=pred_allatom[:,mask_BB[0],:,:3], true=nat_symm[None,mask_BB[0],:,:3],
                atom_mask=xs_mask_lig[:,mask_BB[0]], rmsd_mask=xs_mask_lig[:,mask_BB[0]]
            )
        else:
            rmsd_lig_lig = torch.tensor([0], device=pred.device)
        if torch.any(prot_mask_BB) and torch.any(not_prot_mask_BB) and torch.any(mask_BB[0]):
            rmsd_prot_lig = calc_crd_rmsd(
                pred=pred_allatom[:,mask_BB[0],:,:3], true=nat_symm[None,mask_BB[0],:,:3],
                atom_mask=xs_mask_prot[:,mask_BB[0]], rmsd_mask=xs_mask_lig[:,mask_BB[0]],
                alignment_radius=10.0
            )
        else:
            rmsd_prot_lig = torch.tensor([0], device=pred.device)
 
        loss_dict["rmsd_prot_prot"]= rmsd_prot_prot[0].detach()
        loss_dict["rmsd_lig_lig"]= rmsd_lig_lig[0].detach()
        loss_dict["rmsd_prot_lig"]= rmsd_prot_lig[0].detach()

        # cart bonded (bond geometry)
        bond_loss = calc_BB_bond_geom(seq[0], pred_allatom[0:1], idx)
        if w_bond > 0.0:
            tot_loss += w_bond*bond_loss
        loss_dict['bond_geom'] = bond_loss.detach()

        # if (pred_allatom.shape[0] > 1):
        #     bond_loss = calc_cart_bonded(seq, pred_allatom[1:], idx, self.cb_len, self.cb_ang, self.cb_tor)
        #     if w_bond > 0.0:
        #         tot_loss += w_bond*bond_loss.mean()
        #     loss_dict['clash_loss'] = ( bond_loss.detach() )
        # else:
        #     bond_loss = torch.tensor(0).to(gpu)
        # loss_dict['bond_loss'] = bond_loss.detach()

        # clash [use all atoms not just those in native]
        # clash_loss = calc_lj(
        #     seq[0], pred_allatom, 
        #     self.aamask, bond_feats, dist_matrix, self.ljlk_parameters, self.lj_correction_parameters, self.num_bonds,
        #     lj_lin=lj_lin
        # )
        clash_loss, num_violations = calc_l1_clash_loss(pred_allatom, seq[0],\
                                                        trainer.aamask, bond_feats, dist_matrix, trainer.ljlk_parameters, \
                                                        trainer.lj_correction_parameters, trainer.num_bonds)
        if w_clash > 0.0:
            tot_loss += w_clash*clash_loss.mean()
        loss_dict['clash_loss'] = clash_loss.detach()
        if torch.any(mask_BB[0]):
            atom_bond_loss, skip_bond_loss, rigid_loss = calc_atom_bond_loss(
                pred=pred_allatom[:,mask_BB[0]],
                true=nat_symm[None,mask_BB[0]],
                bond_feats=bond_feats[:,mask_BB[0]][:,:,mask_BB[0]],
                seq=seq[:,mask_BB[0]]
            )
        else:
            atom_bond_loss = torch.tensor(0, device=gpu)
            skip_bond_loss = torch.tensor(0, device=gpu)
            rigid_loss = torch.tensor(0, device=gpu)

        if w_atom_bond >= 0.0:
            tot_loss += w_atom_bond*atom_bond_loss
        loss_dict['atom_bond_loss'] = ( atom_bond_loss.detach() )

        if w_skip_bond >= 0.0:
            tot_loss += w_skip_bond*skip_bond_loss
        loss_dict['skip_bond_loss'] = ( skip_bond_loss.detach() )

        if w_rigid >= 0.0:
            tot_loss += w_rigid*rigid_loss
        loss_dict['rigid_loss'] = ( rigid_loss.detach() )
        chain_prot = same_chain.clone()
        protein_mask_2d = torch.einsum('l,r-> lr', prot_mask_BB, prot_mask_BB)

        _, allatom_lddt_prot_intra = calc_allatom_lddt_loss(
            pred_allatom.detach(), nat_symm, pred_lddt, idx, mask_crds, protein_mask_2d[None], 
            chain_prot, negative=True, N_stripe=10)
        loss_dict['allatom_lddt_prot_intra'] = allatom_lddt_prot_intra[0].detach()

        _, allatom_lddt_prot_inter = calc_allatom_lddt_loss(
            pred_allatom.detach(), nat_symm, pred_lddt, idx, mask_crds, protein_mask_2d[None], 
            chain_prot, interface=True, N_stripe=10)
        loss_dict['allatom_lddt_prot_inter'] = allatom_lddt_prot_inter[0].detach()
        
        chain_lig = same_chain.clone()
        not_protein_mask_2d = torch.einsum('l,r-> lr', not_prot_mask_BB, not_prot_mask_BB)
        _, allatom_lddt_lig_intra = calc_allatom_lddt_loss(
            pred_allatom.detach(), nat_symm, pred_lddt, idx, mask_crds, not_protein_mask_2d[None], 
            chain_lig, negative=True, bin_scaling=0.5, N_stripe=10)
        loss_dict['allatom_lddt_lig_intra'] = allatom_lddt_lig_intra[0].detach()
        
        _, allatom_lddt_lig_inter = calc_allatom_lddt_loss(
            pred_allatom.detach(), nat_symm, pred_lddt, idx, mask_crds, not_protein_mask_2d[None], 
            chain_lig, interface=True, bin_scaling=0.5, N_stripe=10)
        loss_dict['allatom_lddt_lig_inter'] = allatom_lddt_lig_inter[0].detach()

        chain_prot_lig_inter = torch.zeros_like(same_chain, dtype=bool)
        chain_prot_lig_inter += protein_mask_2d
        chain_prot_lig_inter += not_protein_mask_2d
        _, allatom_lddt_inter = calc_allatom_lddt_loss(
            pred_allatom.detach(), nat_symm, pred_lddt, idx, mask_crds, mask_2d, 
            chain_prot_lig_inter, interface=True, N_stripe=10)
        loss_dict['allatom_lddt_prot_lig_inter'] = allatom_lddt_inter[0].detach()
        loss_dict['total_loss'] = tot_loss.detach()

        return tot_loss, loss_dict


### this file will contain specific calls to the loss function 
class LossManager:
    """ this class computes the loss and holds useful primitives for loss calc """
    def __init__(self, config) -> None:
        self.loss_list = []
        self.loss_weights = []
        self.loss_dict = {}

    def compute_loss(self, rf_inputs, rf_outputs):
        for loss in self.loss_list:
            pass

    def get_frames(self): 
        if self.frames is not None and self.frame_mask is not None:
            return self.frames, self.frame_mask
        else:
            pass
    

loss_factory = {
    "c6d": None,
    "mlm": None,
    "lddt": None,
    "pae": None,
    "bb_fape": None,
    "allatom_fape": None,

}
def c6d_loss(loss_manager, trainer):
    pass
