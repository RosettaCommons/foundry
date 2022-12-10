import sys, os, json
import time
import numpy as np
import torch
import torch.nn as nn

from loss_halluc import calc_entropy_loss, calc_pae_loss
from optimization import run_gradient_descent, run_mcmc

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0,script_dir+'/models/fold_and_dock3/')
import parsers
from RoseTTAFoldModel import RoseTTAFoldModule
from data_loader import merge_a3m_hetero
import util
from kinematics import xyz_to_c6d, c6d_to_bins, xyz_to_t2d, get_chirals
from chemical import NTOTAL, NTOTALDOFS, NAATOKENS, INIT_CRDS
from model_params import MODEL_PARAM

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="RF2-allatom hallucination: protein scaffold design with explicit modeling of small-molecule ligands")
    parser.add_argument("-checkpoint",
        default="/databases/TrRosetta/RF2_allatom/checkpoints/rf2a_fd3_20221125_115.pt",
        help="Path to model weights")
    parser.add_argument("-pdb", help='PDB of motif')
    parser.add_argument("-parse_hetatm", action="store_true", default=False, help="parse ligand information from input pdb")
    parser.add_argument("-mol2", help='mol2 of small molecule')
    parser.add_argument("-num", type=int, default=1, help='number of designs')
    parser.add_argument("-start_num", type=int, default=0, help='start number of output designs')
    parser.add_argument("-grad_steps", type=int, required=True, help='number of gradient descent steps')
    parser.add_argument("-mcmc_steps", type=int, required=True, help='number of mcmc steps')
    parser.add_argument("-out", help='prefix of output files')
    parser.add_argument("-L", type=int, help='length of hallucinated protein')
    parser.add_argument("-T0", type=float, default=0.02, help='initial temperature for simulated annealing')
    parser.add_argument("-mcmc_halflife", default=100, help='half-life of simulated annealing')
    parser.add_argument("-cycles", type=str, default='10', help='number of recycles')
    parser.add_argument("-seq_prob_type", default='hard', help='soft or hard probabilities of sequence')
    parser.add_argument("-init_sd", type=float, default=1e-6, help='random initial logit standard deviation')
    parser.add_argument("-template_ligand", action='store_true', default=False, help='input template features for ligand structure')
    parser.add_argument("-init_ligand_xyz", action='store_true', default=False, help='input initial xyz coords for ligand structure')
    parser.add_argument("-learning_rate", type=float, default=0.05, help='gradient descent learning rate')
    parser.add_argument("-device", type=str, default='cuda:0', help='gpu to run on')
    parser.add_argument("-w_ent", type=float, default=1.0, help='weight of entropy loss')
    parser.add_argument("-w_pae", type=float, default=1.0, help='weight of pae loss')
    parser.add_argument("-w_ipae", type=float, default=1.0, help='weight of inter-pae loss')
    args = parser.parse_args()

    return args

# compute expected value from binned lddt
def lddt_unbin(pred_lddt):
    nbin = pred_lddt.shape[1]
    bin_step = 1.0 / nbin
    lddt_bins = torch.linspace(bin_step, 1.0, nbin, dtype=pred_lddt.dtype, device=pred_lddt.device)
    
    pred_lddt = nn.Softmax(dim=1)(pred_lddt)
    return torch.sum(lddt_bins[None,:,None]*pred_lddt, dim=1)

def load_model(args, MODEL_PARAM):
    device = args.device
    model = RoseTTAFoldModule(
        **MODEL_PARAM,
        aamask = util.allatom_mask.to(device),
        atom_type_index = util.atom_type_index.to(device),
        ljlk_parameters = util.ljlk_parameters.to(device),
        lj_correction_parameters = util.lj_correction_parameters.to(device),
        num_bonds = util.num_bonds.to(device),
        cb_len = util.cb_length_t.to(device),
        cb_ang = util.cb_angle_t.to(device),
        cb_tor = util.cb_torsion_t.to(device),
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def prepare_inputs(args, init_seq=None, random_noise=5):

    B = 1 # batch
    N = 1 # msa depth
    protein_L = args.L
    device = args.device

    if init_seq is None:
        msa_prot = torch.randint(20, (1,protein_L))
    else:
        msa_prot = init_seq
    ins_prot = torch.zeros((1,protein_L)).long()
    idx_prot = torch.arange(protein_L)

    if args.mol2 is not None:
        a3m_prot = {"msa": msa_prot, "ins": ins_prot}
        mol, msa_sm, ins_sm, xyz_sm, mask_sm = parsers.parse_mol(args.mol2)
        a3m_sm = {"msa": msa_sm.unsqueeze(0), "ins": ins_sm.unsqueeze(0)}
        G = util.get_nxgraph(mol)
        atom_frames = util.get_atom_frames(msa_sm, G)
        N_symmetry, sm_L, _ = xyz_sm.shape

        Ls = [protein_L, sm_L]
        a3m = merge_a3m_hetero(a3m_prot, a3m_sm, Ls)
        msa = a3m['msa'].long()
        ins = a3m['ins'].long()
        chirals = get_chirals(mol, xyz_sm[0])

    xyz = torch.full((N_symmetry, sum(Ls), NTOTAL, 3), np.nan).float()
    mask = torch.full(xyz.shape[:-1], False).bool()
    if args.pdb is not None:
        xyz[:, :Ls[0], :nprotatoms, :] = xyz_prot.expand(N_symmetry, Ls[0], nprotatoms, 3)
        mask[:, :protein_L, :nprotatoms] = mask_prot.expand(N_symmetry, Ls[0], nprotatoms)
    if args.mol2 is not None:
        xyz[:, Ls[0]:, 1, :] = xyz_sm
        mask[:, protein_L:, 1] = mask_sm
        
    idx_sm = torch.arange(max(idx_prot),max(idx_prot)+Ls[1])+200
    idx_pdb = torch.concat([idx_prot, idx_sm])

    chain_idx = torch.zeros((sum(Ls), sum(Ls))).long()
    chain_idx[:Ls[0], :Ls[0]] = 1
    chain_idx[Ls[0]:, Ls[0]:] = 1
    bond_feats = torch.zeros((sum(Ls), sum(Ls))).long()
    bond_feats[:Ls[0], :Ls[0]] = util.get_protein_bond_feats(Ls[0])
    if args.mol2 is not None:
        bond_feats[Ls[0]:, Ls[0]:] = util.get_bond_feats(mol)

    # blank template
    xyz_t = INIT_CRDS.reshape(1,1,NTOTAL,3).repeat(1,sum(Ls),1,1) \
        + torch.rand(1,sum(Ls),1,3)*random_noise - random_noise/2
    f1d_t = torch.nn.functional.one_hot(torch.full((1, sum(Ls)), 20).long(), num_classes=NAATOKENS-1).float() # all gaps
    conf = torch.zeros((1, sum(Ls), 1)).float()
    f1d_t = torch.cat((f1d_t, conf), -1)
    mask_t = torch.full((1,sum(Ls),NTOTAL), False)
    
    if args.template_ligand: # input true s.m. xyz as template
        xyz_t[0, Ls[0]:, 1] = xyz_sm[0] - xyz_sm[0].mean(-2) # centroid at origin
        f1d_t[0, Ls[0]:] = torch.cat((
            torch.nn.functional.one_hot(msa[0, Ls[0]:]-1, num_classes=NAATOKENS-1).float(),
            torch.ones((Ls[1], 1)).float()
        ), -1) # (1, L_sm, NAATOKENS)
        mask_t[0, Ls[0]:, 1] = mask_sm[0] # all symmetry variants have same mask

    xyz_t = torch.nan_to_num(xyz_t)

    # black-hole coordinates
    init = INIT_CRDS.reshape(1,NTOTAL,3).repeat(sum(Ls),1,1)
    xyz_prev = init + torch.rand(sum(Ls),1,3)*random_noise - random_noise/2
    mask_prev = torch.full(xyz_prev.shape[:-1], False).bool()

    if args.init_ligand_xyz:
        xyz_prev[Ls[0]:, 1] = xyz_sm[0] - xyz_sm[0].mean(-2) # centroid at origin
        mask_prev[Ls[0]:, 1] = mask_sm[0]
    
    # transfer inputs to device
    atom_frames = atom_frames[None].to(device, non_blocking=True)
    atom_mask = mask[None].to(device, non_blocking=True) # (B, L, 27)
    idx_pdb = idx_pdb[None].to(device, non_blocking=True) # (B, L)
    xyz_t = xyz_t[None].to(device, non_blocking=True)
    mask_t = mask_t[None].to(device, non_blocking=True)
    t1d = f1d_t[None].to(device, non_blocking=True)
    xyz_prev = xyz_prev[None].to(device, non_blocking=True)
    mask_prev = mask_prev[None].to(device, non_blocking=True)
    same_chain = chain_idx[None].to(device, non_blocking=True)
    bond_feats = bond_feats[None].to(device, non_blocking=True)
    chirals = chirals[None].to(device, non_blocking=True)
    
    # processing template features
    seq_unmasked = msa.clone() # (B, L)
    mask_t_2d = util.get_prot_sm_mask(mask_t, seq_unmasked[0]) # (B, T, L)
    mask_t_2d = mask_t_2d[:,:,None]*mask_t_2d[:,:,:,None] # (B, T, L, L)
    mask_t_2d = mask_t_2d.float() * same_chain.float()[:,None] # (ignore inter-chain region)
    mask_recycle = util.get_prot_sm_mask(mask_prev, seq_unmasked[0])
    mask_recycle = mask_recycle[:,:,None]*mask_recycle[:,None,:] # (B, L, L)
    mask_recycle = same_chain.float()*mask_recycle.float()

    xyz_t_frames = util.xyz_t_to_frame_xyz(xyz_t, seq_unmasked, atom_frames)
    t2d = xyz_to_t2d(xyz_t_frames, mask_t_2d)

    seq_tmp = t1d[...,:-1].argmax(dim=-1).reshape(-1,sum(Ls))

    alpha, _, alpha_mask, _ = util.get_torsions(
        xyz_t.reshape(-1,sum(Ls),NTOTAL,3),
        seq_tmp,
        util.torsion_indices.to(device),
        util.torsion_can_flip.to(device),
        util.reference_angles.to(device)
    )
    alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[...,0]))
    alpha[torch.isnan(alpha)] = 0.0
    alpha = alpha.reshape(1,-1,sum(Ls),NTOTALDOFS,2)
    alpha_mask = alpha_mask.reshape(1,-1,sum(Ls),NTOTALDOFS,1)
    alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(1, -1, sum(Ls), 3*NTOTALDOFS).to(device)

    return dict(
        Ls=Ls,
        msa=msa,
        ins=ins,
        xyz_prev=xyz_prev,
        idx_pdb=idx_pdb,
        bond_feats=bond_feats,
        chirals=chirals,
        atom_frames=atom_frames,
        t1d=t1d,
        t2d=t2d,
        xyz_t=xyz_t,
        alpha_t=alpha_t,
        mask_t=mask_t,
        mask_t_2d=mask_t_2d,
        same_chain=same_chain,
        mask_recycle=mask_recycle,
    )

def main():

    args = get_args()

    print('Loading network weights...')
    model = load_model(args, MODEL_PARAM)

    inputs = prepare_inputs(args)

    loss_funcs = [
        dict(w=args.w_ent, name='ent', func=calc_entropy_loss),
        dict(w=args.w_pae, name='pae', func=calc_pae_loss),
        dict(w=args.w_ipae, name='ipae', func=lambda out: calc_pae_loss(out, inter=True))
    ]

    for i_des in range(args.start_num, args.start_num+args.num):
        start_time = time.time()
        cycles = [int(i) for i in args.cycles.split(',')]

        if args.grad_steps > 0:
            out = run_gradient_descent(model, args, inputs, cycles[0], loss_funcs)
            inputs['msa'] = out['msa'][0]

        if args.mcmc_steps > 0:
            out = run_mcmc(model, args, inputs, cycles[min(len(cycles)-1,1)], loss_funcs)
            inputs['msa'] = out['msa'][0]

        out_prefix = args.out+f'_{i_des}'
        best_lddt = lddt_unbin(out['pred_lddt_binned'])
        util.writepdb(out_prefix+".pdb", out['pred_allatom'], out['msa'][0], bfacts=100*best_lddt[0].float(),
                      bond_feats=inputs['bond_feats'])


if __name__ == "__main__":
    main()

