import sys
import hydra
import torch

sys.path.append('/home/rohith/cifutils')
sys.path.append('/home/rohith/datahub')

from datahub.datasets.base import ConcatDatasetWithID
from cifutils import CIFParser
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler

from datahub.datasets.base import (
    FallbackDatasetWrapper,
)
from datahub.samplers import (
    DistributedMixedSampler,
    FallbackSamplerWrapper,
    LazyWeightedRandomSampler, 
    MixedSampler
)
from datahub.common import default
from rf2aa.trainer_new import ComposedTrainer, seed_all
from rf2aa.resolvers import resolve_import
from omegaconf import OmegaConf
import logging

logger = logging.getLogger("main")
OmegaConf.register_new_resolver("resolve_import", resolve_import)


def load_structural_datasets(cfg, name: str = "unknown", cif_parser: CIFParser | None= None, n_fallback_retries: int=0):
    datasets, weights = [], []
    for name, dataset_cfg in cfg.items():
        assert name not in datasets, f"Duplicate training dataset name: {name}"

        # ... get dataset
        kwargs = {}
        if "cif_parser" in dataset_cfg.dataset:
            kwargs["cif_parser"] = default(cif_parser, CIFParser())
        dataset = hydra.utils.instantiate(dataset_cfg.dataset, **kwargs)

        # ... get sampler
        if "weights" in dataset_cfg:
            dataset_weights = hydra.utils.instantiate(dataset_cfg.weights, dataset_df=dataset.data)
        else:
            dataset_weights = torch.ones(len(dataset))

        datasets.append(dataset)
        weights.append(dataset_weights)

    # Concatenate datasets
    _DUMMY_NUM_SAMPLES = 1  # We later override with proportional number of examples
    if len(datasets) > 1:
        datasets = ConcatDatasetWithID(datasets=datasets)
        weights = torch.cat(weights)  # NOTE: Order matters!
        _sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=_DUMMY_NUM_SAMPLES,  # We later override with proportional number of examples
            replacement=True,
            generator=None,
        )
        sampler = MixedSampler(
                datasets_info=[dict(name=name, dataset=datasets, sampler=_sampler, probability=1.0)],
                n_examples_per_epoch=_DUMMY_NUM_SAMPLES,  # We later override with proportional number of examples
                shuffle=True,
            )
    else:
        datasets = datasets[0]
        weights = weights[0]
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=_DUMMY_NUM_SAMPLES,  # We later override with proportional number of examples
            replacement=True,
            generator=None,
        )
    
    if n_fallback_retries > 0:
        fallback_dataset = datasets
        fallback_sampler = LazyWeightedRandomSampler(
            # NOTE: We instantiate a new sampler here to ensure different weights per
            weights=weights,
            num_samples=int(1e9),  # WARNING! Torch's WeightedRandomSampler scales as O(n_samples * n_weights). We use LazyWeightedRandomSampler to avoid this.
            replacement=True,
            generator=None,
            prefetch_buffer_size=4,
        )
        datasets = FallbackDatasetWrapper(datasets, fallback_dataset=fallback_dataset)
        sampler = FallbackSamplerWrapper(
            sampler,
            fallback_sampler=fallback_sampler,
            n_fallback_retries=n_fallback_retries,
        )

    return datasets, sampler

class NewDatapipeTrainer:
    
    def __call__(self, init_db, dataset_params, loader_params, rank: int, world_size: int):
        return self.construct_dataset(init_db, dataset_params, loader_params, rank, world_size)

    def construct_dataset(
        self, init_db, dataset_params, loader_params, rank: int, world_size: int
    ) -> tuple[DataLoader, Sampler, dict[str, DataLoader], dict[str, Sampler]]:
        # Shared parser instance
        cif_parser = CIFParser()

        # Training datasets
        datasets_info = []
        for train_name, train_cfg in dataset_params.train.items():
            logger.info(f"Loading dataset: {train_name}")
            dataset, sampler = load_structural_datasets(train_cfg.sub_datasets, train_name, cif_parser)
            datasets_info.append(
                dict(name=train_name, dataset=dataset, sampler=sampler, probability=train_cfg.probability)
            )

        # Check that the sum of probabilities is 1
        assert sum(dataset_info["probability"] for dataset_info in datasets_info) == 1.0, "Sum of probabilities must be 1.0"
        train_datasets = ConcatDatasetWithID(datasets=[dataset["dataset"] for dataset in datasets_info])

        # Concatenate datasets
        train_sampler = DistributedMixedSampler(
            datasets_info=datasets_info,
            num_replicas=world_size,
            rank=rank,
            n_examples_per_epoch=dataset_params.n_train,  # Number of examples per epoch (accross all GPUs)
            shuffle=True,
            drop_last=True,
        )

        # ... assemble final train loader
        train_loader = torch.utils.data.DataLoader(
            train_datasets,
            batch_size=loader_params.batch_size,
            sampler=train_sampler,
            **loader_params.dataloader_kwargs,

        )

        # Validation
        val_datasets, val_samplers, val_loaders = {}, {}, {}
        for val_name, val_cfg in dataset_params.get("val", {}).items():
            assert val_name not in val_datasets, f"Duplicate validation dataset name: {val_name}"

            val_datasets[val_name] = hydra.utils.instantiate(val_cfg)
            val_samplers[val_name] = torch.utils.data.SequentialSampler(val_datasets[val_name])
            val_loaders[val_name] = torch.utils.data.DataLoader(
                val_datasets[val_name],
                batch_size=loader_params.batch_size,
                sampler=val_samplers[val_name],
                **loader_params.dataloader_kwargs,
            )

        return train_loader, train_sampler, val_loaders, val_samplers