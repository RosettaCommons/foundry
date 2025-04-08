# TODO: Integrate cifparser & biotite into rf2aa .sif

import logging
import os

import certifi
import hydra
import torch  # import torch before adding to python path to ensure apptainer's torch is used
from datahub.datasets.datasets import (
    ConcatDatasetWithID,
    FallbackDatasetWrapper,
)
from datahub.samplers import (
    DistributedMixedSampler,
    FallbackSamplerWrapper,
    LazyWeightedRandomSampler,
)
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler

from modelhub.resolvers import resolve_import

logger = logging.getLogger("main")

# limit thread counts
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:512"
# Update environment variable with correct path (needed for W&B upload)
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
## To reproduce errors
torch.set_num_threads(4)
# torch.autograd.set_detect_anomaly(True)

# ...register custom resolvers
OmegaConf.register_new_resolver("resolve_import", resolve_import)


def load_structural_datasets(cfg, name: str = "unknown"):
    """
    Instantiate structural datasets for training or validation.

    Performs the following steps:
        1. Instantiates the datasets and their corresponding weights, as defined in the Hydra configuration.
        2. Combines the datasets into a single composed ConcatDatasetWithID (if multiple given), building the corresponding composed sampler as well.

    Args:
        cfg (dict): Configuration dictionary from Hydra defining datasets and their parameters. Datasets MAY NOT be nested for this function.
        name (str, optional): Name of the dataset. Defaults to "unknown".

    Returns:
        tuple: A tuple containing the composed dataset and the corresponding sampler.
            - composed_dataset (ConcatDatasetWithID or Dataset): The concatenated dataset if multiple datasets are provided,
              or the single dataset if only one is provided.
            - composed_sampler (Sampler): The sampler for the composed dataset, which can handle weighted sampling and fallback mechanisms.
    """
    datasets, weights = [], []

    # ...loop through datasets defined at this level of the configuration (note that these may be sub-datasets themselves)
    for name, dataset_cfg in cfg.items():
        assert name not in datasets, f"Duplicate training dataset name: {name}"

        # ...instantiate the dataset with the provided configuration
        kwargs = {}
        dataset = hydra.utils.instantiate(dataset_cfg.dataset, **kwargs)

        # ...get the sampler for the dataset
        # TODO: Allow passing a dataframe column as weights
        if "weights" in dataset_cfg:
            dataset_weights = hydra.utils.instantiate(
                dataset_cfg.weights, dataset_df=dataset.data
            )
        else:
            dataset_weights = torch.ones(len(dataset))

        datasets.append(dataset)
        weights.append(dataset_weights)

    # ...concatenate datasets
    _DUMMY_NUM_SAMPLES = 1  # (We later override with proportional number of examples)
    if len(datasets) > 1:
        # (We have multiple datasets; we must concatenate them)
        # ...compose the datasets into a single ConcatDatasetWithID
        composed_dataset = ConcatDatasetWithID(datasets=datasets)
        # ...do the same for the weights
        composed_weights = torch.cat(
            weights
        )  # NOTE: Order of the weights must match the order of the datasets!
        # ...define the composed sampler using the concatenated weights
        # NOTE: As written, we assume that the weights are normalized between the datasets
        # NOTE: If we wanted to use a MixedSampler to sample from sub-datasets with a given probability, we would need to generalize this code
        composed_sampler = WeightedRandomSampler(
            weights=composed_weights,
            num_samples=_DUMMY_NUM_SAMPLES,  # We later override with proportional number of examples
            replacement=True,
            generator=None,
        )
        assert len(composed_weights) == len(composed_dataset), (
            "Weights must match the number of examples in the dataset"
        )
    else:
        # (Only one dataset; just use it)
        composed_dataset = datasets[0]
        composed_weights = weights[0]
        composed_sampler = WeightedRandomSampler(
            weights=composed_weights,
            num_samples=_DUMMY_NUM_SAMPLES,  # We later override with proportional number of examples
            replacement=True,
            generator=None,
        )

    return composed_dataset, composed_sampler


class NewDatapipeTrainer:
    def __call__(self, init_db, dataset_params, loader_params, rank, world_size):
        return self.construct_dataset(
            init_db, dataset_params, loader_params, rank, world_size
        )

    def construct_dataset(
        self,
        _,
        dataset_params,
        loader_params,
        rank: int,
        world_size: int,
    ) -> tuple[DataLoader, Sampler, dict[str, DataLoader], dict[str, Sampler]]:
        # ...extract relevant parameters
        loader_cfg = loader_params.dataloader_kwargs  # (DataLoader configuration)

        # +-------------------------------------------------------------+
        # +--------------------- TRAINING DATASETS ---------------------+
        # +-------------------------------------------------------------+

        # ...we build a "dataset_info" dictionary, which contains the dataset object, sampler object, and probability for each training dataset
        # (See `MixedSampler` in `datahub/samplers/` for more information on "datasets_info")
        datasets_info = []
        # ...loop through "top-level" datasets (e.g., "distillation" and "pdb")
        for train_name, train_cfg in dataset_params.train.items():
            logger.info(f"Loading dataset: {train_name}")
            # ...load the dataset and sampler
            # (If the dataset contains multiple sub-datasets, they will be concatenated into a single dataset)
            # (This setup only supports a two-level hierarchy: top-level datasets and sub-datasets)
            dataset, sampler = load_structural_datasets(
                train_cfg.sub_datasets, train_name
            )
            datasets_info.append(
                dict(
                    name=train_name,
                    dataset=dataset,
                    sampler=sampler,
                    probability=train_cfg.probability,
                )
            )

        # ...check that the sum of probabilities of all datasets is 1
        assert (
            sum(dataset_info["probability"] for dataset_info in datasets_info) == 1.0
        ), "Sum of probabilities must be 1.0"

        # ...compose the list of training datasets into a single dataset
        composed_train_dataset = ConcatDatasetWithID(
            datasets=[dataset["dataset"] for dataset in datasets_info]
        )

        # ...compose the list of samplers, each with their corresponding probability and dataset, into a single sampler
        composed_train_sampler = DistributedMixedSampler(
            datasets_info=datasets_info,
            num_replicas=world_size,
            rank=rank,
            n_examples_per_epoch=dataset_params.n_train,  # Number of examples per epoch (accross all GPUs)
            shuffle=True,
            drop_last=True,
        )

        # ...wrap the composed dataset and sampler with a fallback mechanism, if needed
        if loader_params.n_fallback_retries > 0:
            # ...get the PDB dataset and sampler from the datasets_info
            # We fall back to the PDB dataset (rather than the composed dataset, which also includes distillation)
            pdb_dataset, pdb_sampler = next(
                (entry["dataset"], entry["sampler"])
                for entry in datasets_info
                if entry["name"] == "pdb"
            )

            # ...instantiate the fallback sampler (which must a different instance than the PDB sampler itself)
            fallback_sampler = LazyWeightedRandomSampler(
                # NOTE: We need a new sampler here (rather than using the `composed_sampler`) to avoid the O(n_samples * n_weights) scaling of WeightedRandomSampler
                weights=pdb_sampler.weights,  # Extract the weights from the PDB sampler (which is a WeightedRandomSampler)
                num_samples=int(1e9),
                replacement=True,
                generator=None,
                prefetch_buffer_size=4,
            )

            # ...wrap the composed dataset and sampler with the fallback mechanism
            composed_train_dataset = FallbackDatasetWrapper(
                composed_train_dataset, fallback_dataset=pdb_dataset
            )
            composed_train_sampler = FallbackSamplerWrapper(
                composed_train_sampler,
                fallback_sampler=fallback_sampler,
                n_fallback_retries=loader_params.n_fallback_retries,
            )

        # ...assemble the final train loader
        # assert loader_cfg.num_workers == 0, "num_workers must be 0 for distributed training"
        train_loader = torch.utils.data.DataLoader(
            composed_train_dataset,
            batch_size=1,  # cfg.ddp_params.batch_size,
            sampler=composed_train_sampler,
            collate_fn=lambda x: x,  # No collation
            **loader_cfg,
        )

        # Validation
        val_datasets, val_samplers, val_loaders = {}, {}, {}
        for val_name, val_cfg in dataset_params.get("val", {}).items():
            assert val_name not in val_datasets, (
                f"Duplicate validation dataset name: {val_name}"
            )

            val_datasets[val_name] = hydra.utils.instantiate(val_cfg)
            val_samplers[val_name] = torch.utils.data.distributed.DistributedSampler(
                val_datasets[val_name],
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
            fallback_sampler = LazyWeightedRandomSampler(
                # NOTE: We instantiate a new sampler here to ensure different weights per
                weights=torch.ones(len(val_datasets[val_name])),
                num_samples=int(
                    1e9
                ),  # WARNING! Torch's WeightedRandomSampler scales as O(n_samples * n_weights). We use LazyWeightedRandomSampler to avoid this.
                replacement=True,
                generator=None,
                prefetch_buffer_size=4,
            )

            val_datasets[val_name] = FallbackDatasetWrapper(
                val_datasets[val_name], fallback_dataset=val_datasets[val_name]
            )
            val_samplers[val_name] = FallbackSamplerWrapper(
                val_samplers[val_name],
                fallback_sampler=fallback_sampler,
                n_fallback_retries=loader_params.n_fallback_retries,
            )
            val_loaders[val_name] = torch.utils.data.DataLoader(
                val_datasets[val_name],
                batch_size=1,  # cfg.ddp_params.batch_size,
                sampler=val_samplers[val_name],
                pin_memory=loader_cfg.pin_memory,
                num_workers=loader_cfg.num_workers,
                collate_fn=lambda x: x,  # No collation
            )

        return train_loader, composed_train_sampler, val_loaders, val_samplers
