import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

import pytest
from glob import glob
import warnings
warnings.filterwarnings("ignore")

from rf2aa.training.recycling import recycle_sampling
from rf2aa.tests.test_conditions import setup_benchmark_array
from rf2aa.chemical import ChemicalData as ChemData
from rf2aa.data.compose_dataset import compose_single_item_dataset


# goal is to test all the configs on a broad set of datasets
gpu = "cuda:0" if torch.cuda.is_available() else "cpu"

test_conditions, test_ids = setup_benchmark_array(["pdb256"], ["rf2aa"], device=gpu)

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
    #HACK: reduce crop size

    loader_params["CROP"] = 200
    loader_params["MAXCYCLE"] = 10
    # read from disk, move to device
    dataloader = compose_single_item_dataset( None, item, loader_params, loader, loader_kwargs)
    return dataloader

def test_minimize_example(example, trainer):
    rank, world_size = 0, 1
    if ('MASTER_ADDR' not in os.environ):
        os.environ['MASTER_ADDR'] = '127.0.0.1' # multinode requires this set in submit script

    gpu = trainer.init_process_group(rank, world_size) 

    dataloader = setup_test(example, trainer)
    trainer.train_loader = dataloader
    trainer.model = DDP(trainer.model, device_ids=[gpu], find_unused_parameters=False, broadcast_buffers=False)

    trainer.construct_optimizer()
    trainer.construct_scheduler()
    trainer.construct_scaler()
    trainer.recycle_schedule = recycle_sampling["by_batch"](trainer.config.loader_params.maxcycle, 
                                                            trainer.config.experiment.n_epoch,
                                                            trainer.config.dataset_params.n_train,
                                                            world_size)
    for epoch in range(trainer.config.experiment.n_epoch):
        #sampler.set_epoch(epoch) #TODO: need to make sure each gpu gets a different example
        trainer.train_epoch(epoch, rank, world_size)
        for file in glob("models/*"):
            os.remove(file)

if __name__ == "__main__":
    example, trainer = test_conditions[0]
    os.environ['MASTER_ADDR'] = '127.0.0.1' # multinode requires this set in submit script
    os.environ['MASTER_PORT'] = '%d'%trainer.config.ddp_params.port
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    test_minimize_example(example, trainer) 