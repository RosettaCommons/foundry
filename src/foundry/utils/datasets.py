import hydra
import torch
from atomworks.ml.samplers import (
    DistributedMixedSampler,
    FallbackSamplerWrapper,
    LazyWeightedRandomSampler,
    LoadBalancedDistributedSampler,
    MixedSampler,
)
from beartype.typing import Any
from omegaconf import DictConfig, ListConfig
from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    Sampler,
    SequentialSampler,
    Subset,
    WeightedRandomSampler,
)
from torch.utils.data.distributed import DistributedSampler

from foundry.hydra.resolvers import register_resolvers
from foundry.utils.ddp import RankedLogger

ranked_logger = RankedLogger(__name__, rank_zero_only=True)
try:
    from atomworks.ml.datasets import (
        ConcatDatasetWithID,
        FallbackDatasetWrapper,
        get_row_and_index_by_example_id,
    )
except Exception as e:
    ranked_logger.warning(
        f"Failed to import atomworks.ml.datasets: {type(e).__name__}: {e}. "
        "If training networks, the PDB_MIRROR environment variable must be set."
    )


register_resolvers()


def wrap_dataset_and_sampler_with_fallbacks(
    dataset_to_be_wrapped: Dataset,
    sampler_to_be_wrapped: Sampler,
    dataset_to_fallback_to: Dataset,
    sampler_to_fallback_to: Sampler,
    n_fallback_retries: int,
) -> tuple[Dataset, Sampler]:
    """Wrap the specified dataset and sampler with fallback dataloading.

    If the provided fallback sampler does not have weights (e.g., a MixedSampler), we will use uniform weights.

    Args:
        dataset_to_be_wrapped (Dataset): The main dataset to be wrapped.
        sampler_to_be_wrapped (Sampler): The main sampler to be wrapped.
        dataset_to_fallback_to (Dataset): The fallback dataset. We will sample from this dataset if the main dataset fails.
        sampler_to_fallback_to (Sampler): The fallback sampler. We will sample from this sampler if the main sampler fails.
        n_fallback_retries (int): Number of retries for the fallback mechanism before raising an exception.

    Returns:
        tuple[Dataset, Sampler]: The wrapped dataset and sampler with fallbacks.
    """
    # Instantiate a new fallback sampler to avoid scaling issues
    fallback_sampler = LazyWeightedRandomSampler(
        weights=sampler_to_fallback_to.weights
        if "weights" in sampler_to_fallback_to
        else torch.ones(len(dataset_to_fallback_to)),
        num_samples=int(1e9),
        replacement=True,  # replacement for fallback dataloading, so we can draw a huge number of samples
        generator=None,
        prefetch_buffer_size=4,
    )

    # Wrap the dataset and sampler with fallback mechanisms
    wrapped_dataset = FallbackDatasetWrapper(
        dataset_to_be_wrapped, fallback_dataset=dataset_to_fallback_to
    )
    wrapped_sampler = FallbackSamplerWrapper(
        sampler_to_be_wrapped,
        fallback_sampler=fallback_sampler,
        n_fallback_retries=n_fallback_retries,
    )

    return wrapped_dataset, wrapped_sampler


def instantiate_single_dataset_and_sampler(cfg: DictConfig | dict) -> dict[str, Any]:
    """Instantiate a dataset and its corresponding sampler from a configuration dictionary.

    Args:
        cfg (DictConfig): Configuration dictionary defining the dataset and its parameters.

    Returns:
        dict[str, Any]: A dictionary containing the instantiated dataset and sampler.
    """
    # ... instantiate the dataset
    dataset = hydra.utils.instantiate(cfg.dataset)

    # Users may provide only weights, in which case we will use a WeightedRandomSampler,
    # or they may provide a sampler directly

    if "weights" in cfg and "sampler" not in cfg:
        # ... instantiate the weights and create a WeightedRandomSampler
        weights = hydra.utils.instantiate(cfg.weights, dataset_df=dataset.data)
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(dataset),
            replacement=True,
        )
    elif "sampler" in cfg and "weights" not in cfg:
        # ... instantiate the sampler with the number of samples
        sampler = hydra.utils.instantiate(cfg.sampler)
    else:
        dataset_name = getattr(getattr(cfg.dataset, "dataset", None), "name", None)
        ranked_logger.warning(
            f"No weights or sampler provided for dataset: {dataset_name}, using uniform weights with replacement."
        )
        sampler = WeightedRandomSampler(
            weights=torch.ones(len(dataset)),
            num_samples=len(dataset),
            replacement=True,
        )

    return {"dataset": dataset, "sampler": sampler}


def recursively_instantiate_datasets_and_samplers(
    cfg: DictConfig | dict, name: str | None = None
) -> dict[str, Any]:
    """Recursively instantiate datasets and samplers from a configuration dictionary.

    We must handle three cases:
        (1) A single "leaf" dataset (e.g., "distillation"), specified with the "dataset" key
        (2) Multiple sub-datasets that should be concatenated together with their weights (e.g., "interfaces" and "pn_units"),
            specified with the "sub_datasets" key
        (3) Multiple "leaf" datasets that should be sampled from with a certain probability (e.g., "distillation" and "pdb"),

    Args:
        cfg (DictConfig): Configuration dictionary defining datasets and their parameters.
        name (str, optional): The name of the dataset, used for reporting. Defaults to None.

    Returns:
        dict[str, Any]: A dictionary containing the instantiated dataset and sampler.
    """
    # ------- Base case (1): A single "leaf" dataset -------#
    if "dataset" in cfg:
        return {**instantiate_single_dataset_and_sampler(cfg), "name": name}

    # ------- Recursive case (2): Multiple sub-datasets that must be concatenated together -------#
    elif "sub_datasets" in cfg:
        # ... create a list of dictionaries for each sub-dataset
        datasets_info = []
        for sub_dataset_name, sub_dataset_cfg in cfg.sub_datasets.items():
            if sub_dataset_cfg is None:
                # (Skip any None sub-datasets; e.g., those overrode by the experiment config)
                continue

            datasets_info.append(
                recursively_instantiate_datasets_and_samplers(
                    sub_dataset_cfg, name=sub_dataset_name
                )
            )

        # ... concatenate sub-datasets and weights (e.g., "interfaces" and "pn_units" into one ConcatDataset)
        # NOTE: Order of the weights must match the order of the datasets!
        concatenated_dataset = ConcatDatasetWithID(
            datasets=[info["dataset"] for info in datasets_info]
        )
        concatenated_weights = torch.cat(
            [info["sampler"].weights for info in datasets_info]
        )
        sampler = WeightedRandomSampler(
            weights=concatenated_weights,
            num_samples=len(concatenated_dataset),
            replacement=True,
        )

        return {"dataset": concatenated_dataset, "sampler": sampler, "name": name}

    # ------- Recursive case (3): Multiple datasets that must be sampled from with specified probabilities -------#
    else:
        datasets_info = []
        for nested_dataset_name, nested_dataset_cfg in cfg.items():
            if nested_dataset_cfg is None:
                # (Skip any None training datasets; e.g., those overrode by the experiment config)
                continue

            # (To use a MixedSampler, we must provide a "probability" key for each dataset)
            assert (
                "probability" in nested_dataset_cfg
            ), "Expected 'probability' key in dataset configuration"
            datasets_info.append(
                {
                    **recursively_instantiate_datasets_and_samplers(
                        nested_dataset_cfg, name=nested_dataset_name
                    ),
                    "probability": nested_dataset_cfg["probability"],
                }
            )

        # ... check that the sum of probabilities of all datasets is 1
        assert (
            abs(1 - sum(dataset_info["probability"] for dataset_info in datasets_info))
            < 1e-5
        ), "Sum of probabilities must be 1.0"

        # ... compose the list of datasets into a single dataset
        composed_train_dataset = ConcatDatasetWithID(
            datasets=[dataset["dataset"] for dataset in datasets_info]
        )

        composed_train_sampler = MixedSampler(datasets_info=datasets_info, shuffle=True)

        return {
            "dataset": composed_train_dataset,
            "sampler": composed_train_sampler,
            "name": name,
        }


def assemble_distributed_loader(
    dataset: Dataset,
    sampler: Sampler | None = None,
    rank: int | None = None,
    world_size: int | None = None,
    n_examples_per_epoch: int | None = None,
    loader_cfg: DictConfig | dict | None = None,
    shuffle: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """Assembles a distributed DataLoader for training or validation.

    Performs the following steps:
        (1) If not already a distributed sampler, wraps the sampler with a DistributedSampler or DistributedMixedSampler
        (2) Wraps the dataset and sampler with a fallback mechanism, if needed
        (3) Assembles the final DataLoader

    Args:
        dataset (Dataset): The dataset to be used for training or validation.
        sampler (Sampler): The sampler to be used for training or validation. May already be distributed.
        rank (int): The rank of the current process in distributed training.
        world_size (int): The total number of processes participating in the distributed training.
        n_examples_per_epoch (int): The number of examples to sample per epoch, across all GPUs.
            For example, if we have 8 GPUs, with 2 gradient accumulation steps and 10 optimizer
            steps per epoch, we would sample 160 examples per epoch (8 * 2 * 10).
        loader_cfg (DictConfig or dict, optional): Additional configuration parameters for the
            DataLoader, such as `batch_size` and `num_workers`. Defaults to an empty dictionary.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.
        drop_last (bool, optional): Whether to drop the last incomplete batch if the dataset size
            is not divisible by the number of GPUs. Defaults to False.

    Returns:
        DataLoader: A PyTorch DataLoader configured for distributed training, with datasets
            concatenated and sampled according to their defined probabilities.
    """
    if not loader_cfg:
        loader_cfg = {}

    if isinstance(sampler, MixedSampler):
        # (If given a MixedSampler, we must convert to a DistributedMixedSampler)
        assert (
            rank is not None
            and world_size is not None
            and n_examples_per_epoch is not None
        ), "Rank, world_size, and n_examples_per_epoch must be provided for MixedSampler"
        sampler = DistributedMixedSampler(
            datasets_info=sampler.datasets_info,
            num_replicas=world_size,
            rank=rank,
            n_examples_per_epoch=n_examples_per_epoch,
            shuffle=shuffle,
            drop_last=drop_last,
        )
    elif isinstance(sampler, (RandomSampler, SequentialSampler)):
        # (If given a RandomSampler or SequentialSampler, we must convert to a DistributedSampler)
        assert (
            rank is not None and world_size is not None
        ), "Rank and world_size must be provided for RandomSampler or SequentialSampler"
        sampler = DistributedSampler(
            dataset=dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )
    elif sampler is None and isinstance(dataset, Subset):
        # We are subsetting the dataset to a specific set of example IDs
        ranked_logger.info(f"Subsetting dataset to {len(dataset)} examples!")
    else:
        # (We assume we are already given a DistributedSampler or DistributedMixedSampler)
        assert (
            rank is None and world_size is None
        ), "Rank and world_size will have no effect on the provided sampler and should be None"
        assert isinstance(
            sampler, (DistributedSampler, DistributedMixedSampler)
        ), "Invalid sampler type for distributed training."

    # ... wrap the composed dataset and sampler with a fallback mechanism, if needed
    if (
        "n_fallback_retries" in loader_cfg
        and loader_cfg.n_fallback_retries > 0
        and sampler is not None
    ):
        ranked_logger.info(
            f"Wrapping train dataset and sampler with {loader_cfg.n_fallback_retries} fallbacks..."
        )
        dataset, sampler = wrap_dataset_and_sampler_with_fallbacks(
            dataset_to_be_wrapped=dataset,
            sampler_to_be_wrapped=sampler,
            dataset_to_fallback_to=dataset,
            sampler_to_fallback_to=sampler,
            n_fallback_retries=loader_cfg.n_fallback_retries,
        )

    # ... assemble the final loader
    loader = DataLoader(
        dataset=dataset,
        sampler=sampler,
        collate_fn=lambda x: x,  # No collation
        **loader_cfg.dataloader_params if "dataloader_params" in loader_cfg else {},
    )

    return loader


def subset_dataset_to_example_ids(
    dataset: Dataset,
    example_ids: list[str] | ListConfig,
) -> Dataset:
    """Subset a dataset to a specific set of example IDs."""
    indices = []
    for example_id in example_ids:
        index = get_row_and_index_by_example_id(dataset, example_id)["index"]
        indices.append(index)

    return Subset(dataset, indices)


def assemble_val_loader_dict(
    cfg: DictConfig,
    rank: int = 0,
    world_size: int = 1,
    loader_cfg: DictConfig | dict | None = None,
) -> dict[str, DataLoader]:
    """Assemble a dictionary of validation loaders for multiple datasets.

    If a key is provided to balance the dataset, we will use a LoadBalancedDistributedSampler
    rather than a DistributedSampler to maintain a balanced example load across processes
    (i.e., avoid a situation where one GPU is allocated all small examples and another all large examples).

    Args:
        cfg (DictConfig): Configuration dictionary defining the validation datasets. Each key should correspond to a dataset name.
        rank (int, optional): The rank of the current process in distributed training. Defaults to 0.
        world_size (int, optional): The total number of processes participating in the distributed training. Defaults to 1.
        loader_cfg (DictConfig, optional): Additional configuration parameters for the DataLoader, such as `batch_size` and `num_workers`. Defaults to None.
    """
    # ... loop through the validation datasets and create a DataLoader for each, preserving the dataset name
    val_loaders = {}
    for val_dataset_name, val_dataset in cfg.items():
        if not val_dataset:
            # (Skip any None validation datasets; e.g., those overrode by the experiment config)
            continue

        assert (
            "dataset" in val_dataset
        ), f"Expected 'dataset' key in validation dataset config for {val_dataset_name}"
        dataset = hydra.utils.instantiate(
            val_dataset.dataset
        )  # directly instantiate the dataset

        if "key_to_balance" in val_dataset and val_dataset.key_to_balance:
            # (If a key is provided to balance the dataset, we will use a LoadBalancedDistributedSampler)
            key_to_balance = val_dataset.key_to_balance
            ranked_logger.info(f"Balancing dataset with key: {key_to_balance}")

            assert (
                key_to_balance in dataset.data.columns
            ), f"Key {key_to_balance} not found in dataset columns!"

            sampler = LoadBalancedDistributedSampler(
                dataset=dataset,
                num_replicas=world_size,
                rank=rank,
                key_to_balance=key_to_balance,
            )
        else:
            # (Otherwise, we will use a DistributedSampler, without regard to sample size)
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )

        val_loader = assemble_distributed_loader(
            dataset=dataset,
            sampler=sampler,
            loader_cfg=loader_cfg,
        )

        val_loaders[val_dataset_name] = val_loader

    return val_loaders
