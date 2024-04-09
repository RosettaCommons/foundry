import torch
import torch.nn as nn

import numpy as np

from contextlib import ExitStack

from rf2aa.chemical import ChemicalData as ChemData

def recycle_step_legacy(ddp_model, input, n_cycle, use_amp, nograds=False, force_device=None):
    if force_device is not None:
        gpu = force_device
    else:
        gpu = ddp_model.device

    xyz_prev, alpha_prev, mask_recycle = \
        input["xyz_prev"], input["alpha_prev"], input["mask_recycle"]
    output_i = (None, None, xyz_prev, alpha_prev, mask_recycle)
    for i_cycle in range(n_cycle):
        with ExitStack() as stack:
            #stack.enter_context(torch.cuda.amp.autocast(enabled=use_amp))
            if i_cycle < n_cycle -1 or nograds is True:
                stack.enter_context(torch.no_grad())
                stack.enter_context(ddp_model.no_sync())
            return_raw = (i_cycle < n_cycle -1)
            use_checkpoint = not nograds and (i_cycle == n_cycle -1)

            input_i = add_recycle_inputs(input, output_i, i_cycle, gpu, return_raw=return_raw, use_checkpoint=use_checkpoint)
            output_i = ddp_model(**input_i)
    return output_i

def recycle_step_packed(ddp_model, input, n_cycle, use_amp, nograds=False, force_device=None):
    """ exactly same logic as legacy recycling, except inputs and outputs are dictionaries"""
    if force_device is not None:
        gpu = force_device
    else:
        gpu = ddp_model.device

    xyz_prev, alpha_prev, mask_recycle = \
        input["xyz_prev"], input["alpha_prev"], input["mask_recycle"]
    output_i = (None, None, xyz_prev, alpha_prev, mask_recycle)
    for i_cycle in range(n_cycle):
        with ExitStack() as stack:
            #stack.enter_context(torch.cuda.amp.autocast(enabled=use_amp))
            if i_cycle < n_cycle -1 or nograds is True:
                stack.enter_context(torch.no_grad())
                stack.enter_context(ddp_model.no_sync())
            return_raw = (i_cycle < n_cycle -1)
            use_checkpoint = not nograds and  (i_cycle == n_cycle -1)

            input_i = add_recycle_inputs(input, output_i, i_cycle, gpu, return_raw=return_raw, use_checkpoint=use_checkpoint)
            rf_outputs, rf_latents = ddp_model(input_i, use_checkpoint=use_checkpoint, use_amp=use_amp)
            output_i = unpack_outputs(rf_outputs, rf_latents, return_raw)

    return output_i

def run_model_forward(model, network_input, device="cpu"):
    """ run model forward pass, no recycling, no ddp (for tests) """
    gpu = device
    use_checkpoint = False
    use_amp = False
    xyz_prev, alpha_prev, mask_recycle = \
        network_input["xyz_prev"], network_input["alpha_prev"], network_input["mask_recycle"]
    output_i = (None, None, xyz_prev, alpha_prev, mask_recycle)
    input_i = add_recycle_inputs(network_input, output_i, 0, gpu, return_raw=False)
    input_i["seq_unmasked"] = input_i["seq_unmasked"].to(gpu)
    model.eval()
    with torch.no_grad():
        rf_outputs, rf_latents = model(input_i, use_checkpoint, use_amp)
    return rf_outputs, rf_latents

def run_model_forward_legacy(model, network_input, device="cpu"):
    """ run model forward pass, no recycling or ddp with legacy model (for tests)"""
    gpu = device
    xyz_prev, alpha_prev, mask_recycle = \
        network_input["xyz_prev"], network_input["alpha_prev"], network_input["mask_recycle"]
    output_i = (None, None, xyz_prev, alpha_prev, mask_recycle)
    input_i = add_recycle_inputs(network_input, output_i, 0, gpu, return_raw=False, use_checkpoint=False)
    input_i["seq_unmasked"] = input_i["seq_unmasked"].to(gpu)
    input_i["sctors"] = input_i["sctors"].to(gpu)
    model.eval()
    with torch.no_grad():
        output_i = model(**input_i)

    return output_i

def unpack_outputs(rf_outputs, rf_latents, return_raw):
    #HACK: this just unpacks the outputs into the way the previous RFAA loss function accepts it
    # in the future the loss function should accept rf_outputs and rf_latents
    msa, pair, state = rf_latents["msa"], rf_latents["pair"], rf_latents["state"]

    if return_raw:
        xyz_prev = rf_outputs["xyzs"][-1][None]
        alpha_prev = rf_outputs["alphas"][-1]
        return msa[:, 0], pair, xyz_prev, alpha_prev, None # mask_recycle is always None

    else:
        c6d_logits, mlm_logits, pae_logits, plddt_logits = rf_outputs["c6d"], rf_outputs["mlm"], \
            rf_outputs["pae"], rf_outputs["plddt"]
        pde_logits = None
        p_bind = None
        xyz, alphas = rf_outputs["xyzs"], rf_outputs["alphas"]
        if "xyz_intermediate" in rf_latents:
            intermediate_xyzs = torch.stack(rf_latents["xyz_intermediate"], dim=0)
            xyz = torch.cat((intermediate_xyzs, xyz), dim=0)

        if "alpha_intermediate" in rf_latents:
            alpha_intermediate = torch.stack(rf_latents["alpha_intermediate"], dim=0)
            alphas = torch.cat((alpha_intermediate, alphas), dim=0)

        xyz_allatom = None
        return (c6d_logits, mlm_logits, pae_logits, pde_logits, p_bind, 
                xyz, alphas, xyz_allatom, plddt_logits, msa[:, 0], pair, state)

def add_recycle_inputs(network_input, output_i, i_cycle, gpu, return_raw=False, use_checkpoint=False):
    input_i = {}
    for key in network_input:
        if key in ['msa_latent', 'msa_full', 'seq']:
            input_i[key] = network_input[key][:,i_cycle].to(gpu, non_blocking=True)
        else:
            input_i[key] = network_input[key]

    L = input_i["msa_latent"].shape[2]
    msa_prev, pair_prev, _, alpha, mask_recycle = output_i
    xyz_prev = ChemData().INIT_CRDS.reshape(1,1,ChemData().NTOTAL,3).repeat(1,L,1,1).to(gpu, non_blocking=True)

    input_i['msa_prev'] = msa_prev
    input_i['pair_prev'] = pair_prev
    input_i['xyz'] = xyz_prev
    input_i['mask_recycle'] = mask_recycle
    input_i['sctors'] = alpha
    input_i['return_raw'] = return_raw
    input_i['use_checkpoint'] = use_checkpoint

    input_i.pop('xyz_prev')
    input_i.pop('alpha_prev')
    return input_i


def get_recycle_schedule(max_cycle, n_epochs, n_train, world_size, **kwargs):
    '''
        get's the number of recycles per example.
    '''
    assert n_train % world_size == 0
    # need to sync different gpus
    recycle_schedules=[]
    # make deterministic
    np.random.seed(0)
    for i in range(n_epochs):
        recycle_schedule=[np.random.randint(1,max_cycle+1) for _ in range(n_train//world_size)]
        recycle_schedules.append(torch.tensor(recycle_schedule))
    return torch.stack(recycle_schedules, dim=0)

def get_recycle_schedule_opt(max_cycle, n_epochs, n_train, world_size, **kwargs):
    assert n_train % world_size == 0
    np.random.seed(0)
    recycle_schedule = np.random.randint(1, max_cycle+1, (n_epochs, n_train // world_size))
    return torch.tensor(recycle_schedule)

def get_random_recycle(max_cycle, **kwargs):
    N_cycle = np.random.randint(1, max_cycle+1)
    return N_cycle


recycle_sampling = {
    "random": get_random_recycle,
    "by_batch": get_recycle_schedule_opt
}