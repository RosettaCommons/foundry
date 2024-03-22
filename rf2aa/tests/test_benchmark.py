import os
import torch
import pytest
import warnings
warnings.filterwarnings("ignore")

from rf2aa.data.dataloader_adaptor import prepare_input, get_loss_calc_items
from rf2aa.debug import debug_device
from rf2aa.training.recycling import run_model_forward, recycle_step_packed
from rf2aa.loss.loss_factory import get_loss_and_misc
from rf2aa.tensor_util import assert_equal
from rf2aa.tests.test_conditions import setup_benchmark_array,\
      make_deterministic, setup_data
from rf2aa.util_module import XYZConverter
from rf2aa.chemical import ChemicalData as ChemData, initialize_chemdata
from rf2aa.data.compose_dataset import compose_single_item_dataset
from rf2aa.set_seed import seed_all
from functools import partial


# goal is to test all the configs on a broad set of datasets
gpu = "cuda:0" if torch.cuda.is_available() else "cpu"

test_conditions, test_ids = setup_benchmark_array(["pdb256"], ["rf2aa","rf2_deep_layerdropout"])

def setup_test(example, trainer):
    model = trainer.model
    config = trainer.config.chem_params

    # initialize chemical database
    ChemData.reset() # force reload chemical data
    ChemData(config)

    # to GPU
    trainer.move_constants_to_device(gpu)
    model = model.to(gpu)

    dataset_name = example[0]
    item, loader_params, _, loader, loader_kwargs = example[1:]

    # read from disk, move to device
    dataloader = compose_single_item_dataset( None, item, loader_params, loader, loader_kwargs)
    dataloader_inputs = next(iter( dataloader ))
    dataloader_inputs = tuple(x.to(gpu) if type(x) is torch.Tensor else x for x in dataloader_inputs)

    xyz_converter = XYZConverter().to(gpu)
    inputs = prepare_input(dataloader_inputs, xyz_converter, gpu)
    return dataset_name, dataloader_inputs, inputs

@pytest.mark.gpu
@pytest.mark.benchmark(group="forward")
@pytest.mark.parametrize("example,trainer", test_conditions, ids=test_ids)
def test_benchmark_fw(benchmark, example, trainer):
    dataset_name, dataloader_inputs, inputs = setup_test(example, trainer)
    make_deterministic()
    task, item, network_input, true_crds, \
        atom_mask, msa, mask_msa, unclamp, negative, symmRs, Lasu, ch_label  = inputs

    def run():
        output_i = recycle_step_packed(trainer.model, network_input, 1, False, nograds=True, force_device=gpu) 
        torch.cuda.synchronize(gpu)
        return output_i

    result = benchmark(run)

@pytest.mark.gpu
@pytest.mark.benchmark(group="forward_backward")
@pytest.mark.parametrize("example,trainer", test_conditions, ids=test_ids)
def test_benchmark_fw_bw(benchmark, example, trainer):
    dataset_name, dataloader_inputs, inputs = setup_test(example, trainer)
    make_deterministic()
    task, item, network_input, true_crds, \
        atom_mask, msa, mask_msa, unclamp, negative, symmRs, Lasu, ch_label  = inputs
    msa = msa.to(gpu)
    mask_msa = mask_msa.to(gpu)

    def run():
        output_i = recycle_step_packed(trainer.model, network_input, 1, trainer.config.training_params.use_amp, nograds=False, force_device=gpu) 
        seq, same_chain, idx_pdb, bond_feats, dist_matrix, atom_frames = get_loss_calc_items(dataloader_inputs, device=gpu)

        loss, loss_dict = get_loss_and_misc(
            trainer,
            output_i, true_crds, atom_mask, same_chain,
            seq, msa[:, -1], mask_msa[:, -1], idx_pdb, bond_feats, dist_matrix, atom_frames, unclamp, negative, task, item, symmRs, Lasu, ch_label, 
            trainer.config.loss_param
        )

        loss.backward()
        torch.cuda.synchronize(gpu)
        return loss


    result = benchmark(run)

overrides = []
test_data = setup_data(overrides=overrides)
@pytest.mark.benchmark(group="forward_backward")
@pytest.mark.parametrize("name,item,loader_params,chem_params,loader,loader_kwargs", test_data.values())
def test_benchmark_dataloading(benchmark, name, item, loader_params, chem_params,loader, loader_kwargs):
    ChemData.reset()
    init = partial(initialize_chemdata,chem_params)
    init()

    seed_all()
    data_loader = compose_single_item_dataset(init, item, loader_params, loader, loader_kwargs)
    xyz_converter = XYZConverter().to(gpu)
    def run():
        dataloader_inputs = next(iter(data_loader))
        task, item, network_input, true_crds, mask_crds, msa, mask_msa, unclamp, \
            negative, symmRs, Lasu, ch_label = prepare_input(dataloader_inputs,xyz_converter, gpu)

    result = benchmark(run)
