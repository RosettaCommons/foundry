# this will test if semantically meaningless inputs to the 
# network are actually semantically meaningless
import pytest
import torch
import copy
from functools import partial

from rf2aa.data.compose_dataset import compose_single_item_dataset
from rf2aa.data.dataloader_adaptor import prepare_input
from rf2aa.tests.test_conditions import setup_array, configs, make_deterministic, random_param_init
from rf2aa.tensor_util import assert_equal
from rf2aa.training.recycling import run_model_forward, add_recycle_inputs
from rf2aa.util import is_atom
from rf2aa.chemical import ChemicalData as ChemData
from rf2aa.util_module import XYZConverter
from rf2aa.chemical import initialize_chemdata

import rf2aa.cifutils as cifutils
assert "rf2aa" in cifutils.__name__

test_conditions = setup_array(["sm_compl"], ["rf2aa"])
gpu = "cuda:0" if torch.cuda.is_available() else "cpu"

@pytest.mark.parametrize("example,model", test_conditions)
def test_fake_NC_position_noise(example, model):
    """
    this tests whether the network is dependent on the semantically meaningless
    "N" and "C" coordinates of atom nodes
    """
    def run_model_forward_noise(model, network_input, device="cpu"):
        """ identical to run_model_forward but noises the N and C positions of atoms"""
        gpu = device
        use_checkpoint = False
        xyz_prev, alpha_prev, mask_recycle = \
            network_input["xyz_prev"], network_input["alpha_prev"], network_input["mask_recycle"]
        output_i = (None, None, xyz_prev, alpha_prev, mask_recycle)
        input_i = add_recycle_inputs(network_input, output_i, 0, gpu, return_raw=False)
        input_i["seq_unmasked"] = input_i["seq_unmasked"].to(gpu)
        # change N and C values of atom positions, this dummy function is necessary
        # because add_recycle_inputs resets the xyz parameter
        input_i["xyz"][is_atom(input_i["seq_unmasked"]), :, 0] += 1
        input_i["xyz"][is_atom(input_i["seq_unmasked"]), :, 2] += 1
        model.eval()
        with torch.no_grad():
            rf_outputs, rf_latents = model(input_i, use_checkpoint)
        return rf_outputs, rf_latents

    model_name, model, network_input = setup_test(example, model) 
    make_deterministic() 
    rf_outputs, rf_latents = run_model_forward(model, network_input, gpu)
    make_deterministic()
    rf_outputs_noise, rf_latents_noise = run_model_forward_noise(model, network_input, gpu)
    check_output(model_name, network_input, rf_outputs, rf_latents, rf_outputs_noise, rf_latents_noise)
    
@pytest.mark.parametrize("example,model", test_conditions)
def test_fake_sm_idx(example, model):
    """
    this tests if the idx feature which should be semantically meaningless 
    for small molecules affects the network
    """
    model_name, model, network_input = setup_test(example, model) 
    make_deterministic() 
    rf_outputs, rf_latents = run_model_forward(model, network_input, gpu)
    network_input_noise = copy.deepcopy(network_input)
    is_atom_node = is_atom(network_input_noise["seq_unmasked"])
    network_input_noise["idx"][is_atom_node] += 200
    assert torch.sum(~torch.eq(network_input["idx"], network_input_noise["idx"])) > 0, \
        "bug in test setup xyz_t should be different for base and noise case"

    make_deterministic()
    rf_outputs_noise, rf_latents_noise = run_model_forward(model, network_input_noise, gpu)
    check_output(model_name, network_input, rf_outputs, rf_latents, rf_outputs_noise, rf_latents_noise)

@pytest.mark.parametrize("example,model", test_conditions)
def test_mask_t(example, model):
    """
    this tests if random values in xyz_t, t1d or t2d affects the network
    if mask_t is all False
    """
    model_name, model, network_input = setup_test(example, model) 
    network_input["mask_t"] = torch.full_like(network_input["mask_t"], False)
    make_deterministic() 
    rf_outputs, rf_latents = run_model_forward(model, network_input, gpu)
    network_input_noise = copy.deepcopy(network_input)
    network_input_noise["xyz_t"] = torch.ones_like(network_input_noise["xyz_t"])
    assert torch.sum(~torch.eq(network_input["xyz_t"], network_input_noise["xyz_t"])) > 0, \
        "bug in test setup xyz_t should be different for base and noise case"
    make_deterministic()
    rf_outputs_noise, rf_latents_noise = run_model_forward(model, network_input_noise, gpu)
    check_output(model_name, network_input, rf_outputs, rf_latents, rf_outputs_noise, rf_latents_noise)

@pytest.mark.parametrize("example,model", test_conditions)
def test_same_chain(example, model):
    """
    same_chain should not affect the outputs of the network
    """
    model_name, model, network_input = setup_test(example, model) 
    make_deterministic() 
    rf_outputs, rf_latents = run_model_forward(model, network_input, gpu)
    network_input_noise = copy.deepcopy(network_input)
    network_input_noise["same_chain"] = torch.zeros_like(network_input_noise["same_chain"])
    assert torch.sum(~torch.eq(network_input["same_chain"], network_input_noise["same_chain"])) > 0, \
        "bug in test setup same_chain should be different for base and noise case"
    make_deterministic()
    rf_outputs_noise, rf_latents_noise = run_model_forward(model, network_input_noise, gpu)
    check_output(model_name, network_input, rf_outputs, rf_latents, rf_outputs_noise, rf_latents_noise)

def setup_test(example, model):
    model_name, model, config = model

    # initialize chemical database.  Force a reload
    ChemData.reset()
    init = partial(initialize_chemdata,config)
    init()
    
    model = random_param_init(model)
    model = model.to(gpu)
    xyz_converter = XYZConverter().to(gpu)
    _, item, loader_params, _, loader, loader_kwargs = example
    loader = compose_single_item_dataset(init, item, loader_params, loader, loader_kwargs)
    dataloader_inputs = next(iter(loader))
    task, item, network_input, true_crds, mask_crds, msa, mask_msa, unclamp, \
        negative, symmRs, Lasu, ch_label = prepare_input(dataloader_inputs,xyz_converter, gpu)
    return model_name, model, network_input

def check_output(model_name, network_input, rf_outputs, rf_latents, rf_outputs_noise, rf_latents_noise):
    """
    check that the outputs of the network are semantically the same
    """
    for name, output in rf_outputs.items():
        if torch.is_tensor(output):
            if name == "xyzs":
                # xyzs will be different because of the semantically meaningless coordinates
                # only measure similarity over meaningful outputs
                mask = ChemData().allatom_mask.to(gpu)[network_input["seq_unmasked"]][..., :3]
                output = output[:, mask[0]]
                assert torch.sum(mask) > 0 
                assert_equal(rf_outputs_noise[name][:, mask[0]], output)
                continue
            try:
                assert_equal(rf_outputs_noise[name], output)
            except Exception as e:
                raise ValueError(f"{name} not equivalent for {model_name} on {example[0]} dataset") \
                    from e
    
    for name, latent in rf_latents.items():
        if name == "xyz":
            continue # already tested this above
        if torch.is_tensor(latent):
            try:
                assert_equal(rf_latents_noise[name], latent)
            except Exception as e:
                raise ValueError(f"{name} not equivalent for {model_name} on {example[0]} dataset") from e 
