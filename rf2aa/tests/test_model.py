import os
import torch
import pytest
import warnings
warnings.filterwarnings("ignore")

from rf2aa.data.dataloader_adaptor import prepare_input
from rf2aa.debug import debug_device
from rf2aa.training.recycling import run_model_forward, run_model_forward_legacy
from rf2aa.tensor_util import assert_equal
from rf2aa.tests.test_conditions import setup_array,\
      configs, make_deterministic, dataset_pickle_path, model_pickle_path, random_param_init
from rf2aa.util import Ls_from_same_chain_2d, is_atom
from rf2aa.util_module import XYZConverter
from rf2aa.chemical import ChemicalData as ChemData


# goal is to test all the configs on a broad set of datasets

gpu = "cuda:0" if torch.cuda.is_available() else "cpu"

test_conditions = setup_array(["pdb", "na_compl", "rna", "sm_compl", "sm_compl_covale"], ["rf2aa", "rf_with_gradients", "untied_p2p", "rf2_deep_layerdropout"])
legacy_test_conditions = setup_array(["pdb", "na_compl", "rna", "sm_compl", "sm_compl_covale"], ["legacy_train"], device=gpu)

@pytest.mark.gpu
@pytest.mark.parametrize("example,model", test_conditions)
def test_regression(example, model):
    dataset_name, dataset_inputs, model_name, model = setup_test(example, model)
    make_deterministic()
    rf_outputs, rf_latents = run_model_forward(model, dataset_inputs, use_checkpoint=False, device=gpu)
    output_test = {
        "outputs": rf_outputs,
        "latents": rf_latents
    }
    model_pickle = model_pickle_path(dataset_name, model_name)
    match_values(output_test, model_pickle, model_name, dataset_name, output_test.keys())    

@pytest.mark.gpu
@pytest.mark.xfail
@pytest.mark.parametrize("example,model", test_conditions)
def test_regression_checkpoint(example, model):
    """
    run the model with checkpointing enabled
    """
    dataset_name, dataset_inputs, model_name, model = setup_test(example, model)
    make_deterministic()
    rf_outputs, rf_latents = run_model_forward(model, dataset_inputs, use_checkpoint=True, device=gpu)
    output_test = {
        "outputs": rf_outputs,
        "latents": rf_latents
    }
    model_pickle = model_pickle_path(dataset_name, model_name)
    match_values(output_test, model_pickle, model_name, dataset_name, output_test.keys())    
   

@pytest.mark.gpu
@pytest.mark.parametrize("example,model", legacy_test_conditions)
def test_regression_legacy(example, model):
    dataset_name, dataset_inputs, model_name, model = setup_test(example, model)
    make_deterministic()
    output_i = run_model_forward_legacy(model, dataset_inputs, use_checkpoint=False, device=gpu)
    model_pickle = model_pickle_path(dataset_name, model_name)
    output_names = ("logits_c6d", "logits_aa", "logits_pae", \
                        "logits_pde", "p_bind", "xyz", "alpha", "xyz_allatom", \
                        "lddt", "seq", "pair", "state")
    
    match_values_legacy(output_i, model_pickle, model_name, dataset_name, output_names)

@pytest.mark.gpu
@pytest.mark.parametrize("example,model", legacy_test_conditions)
def test_regression_legacy_checkpoint(example, model):
    dataset_name, dataset_inputs, model_name, model = setup_test(example, model)
    make_deterministic()
    output_i = run_model_forward_legacy(model, dataset_inputs, use_checkpoint=True, device=gpu)
    model_pickle = model_pickle_path(dataset_name, model_name)
    output_names = ("logits_c6d", "logits_aa", "logits_pae", \
                        "logits_pde", "p_bind", "xyz", "alpha", "xyz_allatom", \
                        "lddt", "seq", "pair", "state")
    
    match_values_legacy(output_i, model_pickle, model_name, dataset_name, output_names)

def setup_test(example, model):
    model_name, model, config = model

    # initialize chemical database
    ChemData.reset() # force reload chemical data
    ChemData(config.chem_params)

    model = random_param_init(model)
    model = model.to(gpu)
    dataset_name = example[0]
    dataloader_inputs = torch.load(dataset_pickle_path(dataset_name), map_location=gpu)
    xyz_converter = XYZConverter().to(gpu)
    task, item, network_input, true_crds, mask_crds, msa, mask_msa, unclamp, \
        negative, symmRs, Lasu, ch_label = prepare_input(dataloader_inputs,xyz_converter, gpu)
    network_input = grab_three_from_each_chain(network_input)
    return dataset_name, network_input, model_name, model

def grab_three_from_each_chain(network_input):
    """ for polymer chains, chose three residues, for atom chain choose all atoms (intended to make test pickles smaller)"""
    Ls = Ls_from_same_chain_2d(network_input["same_chain"])
    Ls.insert(0,0)

    first_in_chain = torch.cumsum(torch.tensor(Ls), dim=0)
    first_in_chain = first_in_chain[first_in_chain<sum(Ls)] # remove cases where the chain is only one node

    node_from_each_chain = torch.zeros(sum(Ls), device=network_input["seq_unmasked"].device)
    node_from_each_chain[first_in_chain] = 1
    second_in_chain = first_in_chain + 1
    second_in_chain = second_in_chain[second_in_chain<sum(Ls)] # remove cases where the chain is only one node
    node_from_each_chain[second_in_chain] = 1
    third_in_chain = second_in_chain + 1 # se3 transofmer would like more than 1 edge during fwd pass
    third_in_chain = third_in_chain[third_in_chain<sum(Ls)] # remove cases where the chain is only one node
    node_from_each_chain[third_in_chain] = 1
    is_atom_node = is_atom(network_input["seq_unmasked"])
    chosen_nodes = torch.logical_or(node_from_each_chain.bool(), is_atom_node).squeeze()
    return downsample_network_inputs(network_input, chosen_nodes)
    
def downsample_network_inputs(network_input, chosen_nodes):
    network_input['msa_latent'] = network_input['msa_latent'][..., chosen_nodes, :]
    network_input['msa_full'] = network_input["msa_full"][..., chosen_nodes, :]
    network_input['seq'] = network_input["seq"][..., chosen_nodes]
    network_input['seq_unmasked'] = network_input["seq_unmasked"][..., chosen_nodes]
    network_input['idx'] = network_input["idx"][..., chosen_nodes]
    network_input['t1d'] = network_input["t1d"][..., chosen_nodes, :]
    network_input['t2d']  = network_input["t2d"][:, :, chosen_nodes][:, :, :, chosen_nodes]
    network_input['xyz_t'] = network_input["xyz_t"][..., chosen_nodes, :]
    network_input['alpha_t'] = network_input["alpha_t"][..., chosen_nodes, :] 
    network_input['mask_t'] = network_input["mask_t"][..., chosen_nodes, :][..., chosen_nodes] # mask_t is t2d tensor
    network_input['same_chain'] = network_input["same_chain"][:, chosen_nodes][:, :, chosen_nodes]
    network_input['bond_feats'] = network_input["bond_feats"][:, chosen_nodes][:, :, chosen_nodes]
    network_input['dist_matrix'] = network_input["dist_matrix"][:, chosen_nodes][:, :, chosen_nodes]
    network_input['xyz_prev'] = network_input["xyz_prev"][..., chosen_nodes, :, :]
    network_input["alpha_prev"] = network_input['alpha_prev'].to(network_input["seq_unmasked"].device)
    network_input['alpha_prev'] = network_input["alpha_prev"][..., chosen_nodes, :, :]
    
    # need to decremetn chirals for all the deleted residues
    # assumes all ligand atoms come after protein atoms and no ligand atoms were removed
    num_deleted_residues = (~chosen_nodes).sum()
    network_input["chirals"][..., :-1] = network_input["chirals"][..., :-1] - num_deleted_residues

    return network_input


def match_values_legacy(output_i, model_pickle, model_name, dataset_name, output_names):
    if not os.path.exists(model_pickle):
        torch.save(output_i, model_pickle)
    else:
        output_regression = torch.load(model_pickle, map_location=gpu)
        for idx, output in enumerate(output_i):
            got = output
            want = output_regression[idx]
            if output_names[idx] == "logits_c6d":
                for i in range(len(want)):
                    
                    got_i = got[i]
                    want_i = want[i]
                    try:
                        torch.allclose(got_i, want_i, atol=1e-2)
                    except Exception as e:
                        raise ValueError(f"{output_names[idx]} not same for model: {model_name} on dataset: {dataset_name}") from e
            else:
                try:
                    assert torch.allclose(got, want, atol=1e-2)
                except Exception as e:
                    raise ValueError(f"{output_names[idx]} not same for model: {model_name} on dataset: {dataset_name}") from e


def match_values(output_test, model_pickle, model_name, dataset_name, output_names):
    if not os.path.exists(model_pickle):
        torch.save(output_test, model_pickle)
    else:
        output_regression = torch.load(model_pickle, map_location=gpu)
        for output_type in output_regression.keys():
            for output_name, output in output_regression[output_type].items():
                if torch.is_tensor(output):
                    got = output_test[output_type][output_name]
                    want = output
                    try:
                        assert torch.allclose(got, want, atol=1e-2)
                    except Exception as e:
                        raise ValueError(f"{output_name} does not match for model: {model_name} on dataset: {dataset_name}") from e
