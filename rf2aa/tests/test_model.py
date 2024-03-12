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
      configs, make_deterministic, dataset_pickle_path, model_pickle_path
from rf2aa.util_module import XYZConverter
from rf2aa.chemical import ChemicalData as ChemData


# goal is to test all the configs on a broad set of datasets

gpu = "cuda:0" if torch.cuda.is_available() else "cpu"

test_conditions = setup_array(["pdb", "na_compl", "rna", "sm_compl", "sm_compl_covale"], ["rf2aa"])
legacy_test_conditions = setup_array(["pdb", "na_compl", "rna", "sm_compl", "sm_compl_covale"], ["legacy_train"], device=gpu)


@pytest.mark.parametrize("example,model", test_conditions)
def test_regression(example, model):
    dataset_name, dataset_inputs, model_name, model = setup_test(example, model)
    make_deterministic()
    rf_outputs, rf_latents = run_model_forward(model, dataset_inputs, gpu)
    output_test = {
        "outputs": rf_outputs,
        "latents": rf_latents
    }
    model_pickle = model_pickle_path(dataset_name, model_name)
    if not os.path.exists(model_pickle):
        torch.save(output_test, model_pickle)
        print(f"Saved model pickle at {model_pickle}")
    else:
        output_regression = torch.load(model_pickle, map_location=gpu)
        for output_type in output_regression.keys():
            for output_name, output in output_regression[output_type].items():
                if torch.is_tensor(output):
                    got = output_test[output_type][output_name]
                    want = output
                    if output_name in ["alphas", "msa", "msa_full", "pair", "state"]:
                        try:
                            assert torch.allclose(got, want, atol=1e-4)
                        except Exception as e:
                            raise ValueError(f"{output_name} does not match for model: {model_name} on dataset: {dataset_name}") from e
                    else:
                        try:
                            assert_equal(got, want)
                        except Exception as e:
                            raise ValueError(f"{output_name} does not match for model: {model_name} on dataset: {dataset_name}") from e

@pytest.mark.parametrize("example,model", legacy_test_conditions)
def test_regression_legacy(example, model):
    dataset_name, dataset_inputs, model_name, model = setup_test(example, model)
    make_deterministic()
    output_i = run_model_forward_legacy(model, dataset_inputs, gpu)
    model_pickle = model_pickle_path(dataset_name, model_name)
    output_names = ("logits_c6d", "logits_aa", "logits_pae", \
                        "logits_pde", "p_bind", "xyz", "alpha", "xyz_allatom", \
                        "lddt", "seq", "pair", "state")
    
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
                        assert_equal(got_i, want_i)
                    except Exception as e:
                        raise ValueError(f"{output_names[idx]} not same for model: {model_name} on dataset: {dataset_name}") from e
            elif output_names[idx] in ["alpha", "xyz_allatom", "seq", "pair", "state"]:
                try:
                    assert torch.allclose(got, want, atol=1e-4)
                except Exception as e:
                    raise ValueError(f"{output_names[idx]} not same for model: {model_name} on dataset: {dataset_name}") from e
            else:
                try:
                    assert_equal(got, want)
                except Exception as e:
                    raise ValueError(f"{output_names[idx]} not same for model: {model_name} on dataset: {dataset_name}") from e

def setup_test(example, model):
    model_name, model, config = model

    # initialize chemical database
    ChemData.reset() # force reload chemical data
    ChemData(config.chem_params)

    model = model.to(gpu)
    dataset_name = example[0]
    dataloader_inputs = torch.load(dataset_pickle_path(dataset_name), map_location=gpu)
    xyz_converter = XYZConverter().to(gpu)
    task, item, network_input, true_crds, mask_crds, msa, mask_msa, unclamp, \
        negative, symmRs, Lasu, ch_label = prepare_input(dataloader_inputs,xyz_converter, gpu)
    return dataset_name, network_input, model_name, model

