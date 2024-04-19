import pytest
import numpy as np
import hydra
from functools import partial
from scipy.stats import pearsonr
from rf2aa.data.compose_dataset import compose_dataset
from rf2aa.chemical import initialize_chemdata


def get_sampler(config_name: str = "base", rank: int = 0, world_size: int = 1):
    overrides = [
        "dataset_params.n_train=64",
        "dataset_params.fraction_sm_compl=0.25",
        "dataset_params.fraction_pdb=0.25",
        "dataset_params.fraction_fb=0.25",
        "dataset_params.fraction_sm=0.25",
        "loader_params.crop=1024",
        "+loader_params.MSA_LIMIT=2",
    ]

    with hydra.initialize(version_base=None, config_path="../config/train"):
        config = hydra.compose(config_name=config_name, overrides=overrides)

    init_fn = partial(initialize_chemdata, config)
    init_fn()

    train_loader, train_sampler, _, _ = compose_dataset(
        init_fn,
        config.dataset_params,
        config.loader_params,
        rank=rank,
        world_size=world_size,
    )

    train_set = train_loader.dataset
    return train_set, train_sampler


@pytest.mark.parametrize("config_name", ["base"])
def test_sampler(config_name: str):
    train_set, train_sampler_0 = get_sampler(config_name, 0, 2)
    _, train_sampler_1 = get_sampler(config_name, 1, 2)

    assert len(train_sampler_0) == 32
    assert len(train_sampler_1) == 32

    indices_0 = list(iter(train_sampler_0))
    indices_1 = list(iter(train_sampler_1))

    assert len(indices_0) == 32
    assert len(indices_1) == 32

    lengths_0 = []
    lengths_1 = []
    tasks = ["monomer", "fb", "sm_compl", "sm"]
    for i in range(32):
        assert indices_0[i] != indices_1[i]
        inputs_0 = train_set[indices_0[i]]
        inputs_1 = train_set[indices_1[i]]

        task_0 = inputs_0[-2]
        task_1 = inputs_1[-2]
        assert task_0 == task_1
        assert task_0 in tasks
        assert task_1 in tasks

        length_0 = inputs_0[0].shape[1]
        length_1 = inputs_1[0].shape[1]
        lengths_0.append(length_0)
        lengths_1.append(length_1)

    result = pearsonr(lengths_0, lengths_1)
    p_value = result[1]
    assert p_value < 0.05
