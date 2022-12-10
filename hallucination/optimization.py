import sys, os, json
import time
import numpy as np
import torch
import torch.nn as nn

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0,script_dir+'/models/fold_and_dock2/')
import parsers
from RoseTTAFoldModel import RoseTTAFoldModule
from data_loader import merge_a3m_hetero
import util
from kinematics import xyz_to_c6d, c6d_to_bins, xyz_to_t2d, get_chirals
from chemical import NTOTAL, NTOTALDOFS, NAATOKENS, INIT_CRDS
from model_params import MODEL_PARAM

alphabet = list("ARNDCQEGHILKMFPSTWYV-")
aa_N_1 = dict(zip(range(len(alphabet)),alphabet))

def model_wrapper(model, inputs):
    logit_s, logit_aa_s, logit_pae, logit_pde, pred_crds, alpha_s, pred_allatom, pred_lddt_binned, \
        msa_prev, pair_prev, state_prev = model(**inputs)
    return dict(
        logit_s=logit_s,
        logit_aa_s=logit_aa_s,
        logit_pae=logit_pae,
        logit_pde=logit_pde,
        pred_crds=pred_crds,
        alpha_s=alpha_s,
        pred_allatom=pred_allatom,
        pred_lddt_binned=pred_lddt_binned,
        msa_prev=msa_prev,
        pair_prev=pair_prev, 
        state_prev=state_prev
    )

def run_gradient_descent(model, args, inputs, cycles, loss_funcs):
    B, N = 1,1
    device = args.device
    torch.set_grad_enabled(True)

    print()
    print('Starting gradient descent...')
    loss_header = ''.join([f'{loss["name"]:>12}' for loss in loss_funcs])
    print(f'  step{"best loss":>12}{"curr loss":>12}{loss_header} curr seq')

    Ls = inputs['Ls']
    msa = inputs['msa'][None].to(device) # (B, N, L)
    msa_one_hot_sm = nn.functional.one_hot(msa[:,:,Ls[0]:], num_classes=NAATOKENS).float()

    input_logits = args.init_sd*torch.randn([B,N,Ls[0],20]).to(device).float()
    input_logits = input_logits.requires_grad_(True)

    optimizer = NSGD([input_logits], lr=args.learning_rate*np.sqrt(Ls[0]), dim=[-1,-2])

    best_loss = torch.full((B,),1e4).to(device)

    for i_step in range(args.grad_steps):
        optimizer.zero_grad()

        # discretize protein tokens on protein residues
        msa_one_hot_prot = logits_to_probs(input_logits, output_type=args.seq_prob_type) 

        # pad with non-protein tokens on protein residues
        msa_one_hot = torch.concat([msa_one_hot_prot, torch.zeros((B,N,Ls[0],NAATOKENS-20)).to(device)], dim=-1)

        # pad with ligand residues
        msa_one_hot = torch.concat([msa_one_hot, msa_one_hot_sm], dim=-2)
        
        # predict structure
        xyz_prev = inputs['xyz_prev'].clone()
        msa_prev = None
        pair_prev = None
        alpha_prev = torch.zeros((1,sum(Ls),NTOTALDOFS,2), device=device)
        state_prev = None
        mask_recycle = inputs['mask_recycle'].clone()

        for i_cycle in range(cycles):
            out = model_wrapper(model, dict(
                msa_one_hot=msa_one_hot,
                seq_unmasked=msa_one_hot.argmax(-1)[:,0].detach(), # (B,L)
                xyz=inputs['xyz_prev'], 
                sctors=alpha_prev,
                idx=inputs['idx_pdb'],
                bond_feats=inputs['bond_feats'],
                chirals=inputs['chirals'],
                atom_frames=inputs['atom_frames'],
                t1d=inputs['t1d'], 
                t2d=inputs['t2d'],
                xyz_t=inputs['xyz_t'][...,1,:],
                alpha_t=inputs['alpha_t'],
                mask_t=inputs['mask_t_2d'],
                same_chain=inputs['same_chain'],
                msa_prev=msa_prev,
                pair_prev=pair_prev,
                state_prev=state_prev,
                mask_recycle=mask_recycle,
                use_checkpoint=True
            ))

            xyz_prev = out['pred_allatom'][-1].unsqueeze(0)
            msa_prev = out['msa_prev']
            pair_prev = out['pair_prev']
            alpha_prev = out['alpha_s'][-1]
            state_prev = out['state_prev']
            mask_recycle = None

        
        # calculate loss
        curr_msa = msa_one_hot.argmax(-1).detach().clone()  #(B=1,N=1,L)
        out['msa'] = curr_msa
        loss_s = [loss['func'](out) for loss in loss_funcs]
        curr_loss = torch.sum(torch.stack(loss_s), dim=0)
         
        # best design so far
        if curr_loss < best_loss:
            best_loss = curr_loss
            best_loss_s = loss_s
            best_out = out
            msa = curr_msa
            best_step = i_step

        # update sequence
        curr_loss.backward()
        optimizer.step()
        
        # print step info
        seq_str = ''.join([aa_N_1[int(a)] for a in msa_one_hot_prot.argmax(-1)[0,0]])
        loss_str = ''.join([f'{float(loss_val):>12.3f}' for loss_val in loss_s])
        print(f'{i_step:>6}{float(best_loss):>12.3f}{float(curr_loss):>12.3f}{loss_str} {seq_str}')

    seq_str = ''.join([aa_N_1[int(a)] for a in msa[0,0,:Ls[0]]])
    loss_str = ''.join([f'{float(loss_val):>12.3f}' for loss_val in best_loss_s])
    print(f' final{float(best_loss):>12.2f}{" "*(12)}{loss_str} {seq_str}')

    best_out.update(dict(loss = best_loss, msa = msa, best_step=best_step))
    torch.cuda.reset_peak_memory_stats()
    
    return best_out

def run_mcmc(model, args, inputs, cycles, loss_funcs):
    B, N = 1,1
    model.eval()
    torch.set_grad_enabled(False)

    print()
    print('Starting MCMC...')
    loss_header = ''.join([f'{loss["name"]:>12}' for loss in loss_funcs])
    print(f'  step{"best loss":>12}{"curr loss":>12}{"accept?":>8}{loss_header}   curr seq')

    Ls = inputs['Ls']
    device = args.device
    msa = inputs['msa'][None].to(device) # (B, N, L)
    
    best_loss = torch.full((B,),1e4).to(device)
    for i_step in range(args.mcmc_steps):

        # make mutation
        i_pos = np.random.randint(Ls[0])
        aa_new = np.random.randint(20)
        msa_new = msa.clone()
        msa_new[0,0,i_pos] = aa_new # assumes B=1

        # predict structure
        xyz_prev = inputs['xyz_prev'].clone()
        msa_prev = None
        pair_prev = None
        alpha_prev = torch.zeros((1,sum(Ls),NTOTALDOFS,2), device=msa_new.device)
        state_prev = None
        mask_recycle = inputs['mask_recycle'].clone()

        for i_cycle in range(cycles):
            out = model_wrapper(model, dict(
                msa_one_hot=nn.functional.one_hot(msa_new, num_classes=NAATOKENS).float(),
                seq_unmasked=msa_new[:,0], 
                xyz=inputs['xyz_prev'], 
                sctors=alpha_prev,
                idx=inputs['idx_pdb'],
                bond_feats=inputs['bond_feats'],
                chirals=inputs['chirals'],
                atom_frames=inputs['atom_frames'],
                t1d=inputs['t1d'], 
                t2d=inputs['t2d'],
                xyz_t=inputs['xyz_t'][...,1,:],
                alpha_t=inputs['alpha_t'],
                mask_t=inputs['mask_t_2d'],
                same_chain=inputs['same_chain'],
                msa_prev=msa_prev,
                pair_prev=pair_prev,
                state_prev=state_prev,
                mask_recycle=mask_recycle
            ))

            xyz_prev = out['pred_allatom'][-1].unsqueeze(0)
            msa_prev = out['msa_prev']
            pair_prev = out['pair_prev']
            alpha_prev = out['alpha_s'][-1]
            state_prev = out['state_prev']
            mask_recycle = None

        # calculate loss
        curr_msa = msa_new.detach().clone()
        out['msa'] = curr_msa
        loss_s = [loss['func'](out) for loss in loss_funcs]
        curr_loss = torch.sum(torch.stack(loss_s), dim=0)

        # batch-wise Metropolis update
        T = args.T0*(0.5)**(i_step/args.mcmc_halflife)
        p_accept = torch.clamp(torch.exp(-(curr_loss-best_loss)/T), min=0.0, max=1.0)
        accept = torch.rand(1).to(device) < p_accept
        if accept:
            best_loss = curr_loss
            best_loss_s = loss_s
            msa = curr_msa
            best_out = out

        seq_str = ''.join([aa_N_1[int(a)] for a in msa_new[0,~util.is_atom(msa_new)[0]]])
        loss_str = ''.join([f'{float(loss_val):>12.3f}' for loss_val in loss_s])
        print(f'{i_step:>6}{float(best_loss):>12.2f}{float(curr_loss):>12.2f}{int(accept):>8}{loss_str} {seq_str}')
        
    seq_str = ''.join([aa_N_1[int(a)] for a in msa[0,0,:Ls[0]]])
    loss_str = ''.join([f'{float(loss_val):>12.3f}' for loss_val in best_loss_s])
    print(f' final{float(best_loss):>12.2f}{" "*(12+8)}{loss_str} {seq_str}')
    best_out.update(dict(loss = best_loss, msa = msa, loss_terms = best_loss_s))
    torch.cuda.reset_peak_memory_stats()
    
    return best_out


def logits_to_probs(logits, output_type='hard', temp=1, add_gumbel_noise=False, eps=1e-8):
    device = logits.device
    B, N, L, A = logits.shape
    if add_gumbel_noise:
        U = torch.rand(logits.shape)
        noise = -torch.log(-torch.log(U + eps) + eps)
        noise = noise.to(device)
        logits = logits + noise

    y_soft = torch.nn.functional.softmax(logits/temp, -1)
    if output_type == 'soft':
        return y_soft
    elif output_type == 'hard':
        n_cat = y_soft.shape[-1]
        y_oh = torch.nn.functional.one_hot(y_soft.argmax(-1), n_cat)
        y_hard = (y_oh - y_soft).detach() + y_soft
        return y_hard
    else:
        raise NotImplementedError('Output type must be "soft" or "hard"')


class NSGD(torch.optim.Optimizer):
    def __init__(self, params, lr, dim):
        defaults = dict(lr=lr)
        super(NSGD, self).__init__(params, defaults)
        self.dim=dim
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                d_p = p.grad / (torch.norm(p.grad, dim=self.dim, keepdim=True) + 1e-8)
                p.add_(d_p, alpha=-group['lr'])
        return loss

