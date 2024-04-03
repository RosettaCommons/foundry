import os
import warnings
import torch
import pandas as pd
import numpy as np
from torch.utils import data
from typing import Dict
from collections import OrderedDict


def get_id_lengths(
    df: pd.DataFrame, ids: np.ndarray, length_column: str = "LEN_EXIST"
) -> torch.Tensor:
    sub_df = df.groupby("CLUSTER")[[length_column]].mean().reset_index()
    length_dict = dict(zip(sub_df["CLUSTER"], sub_df[length_column]))
    lengths_list = [length_dict[id] for id in ids]
    lengths = torch.tensor(lengths_list, dtype=torch.float32)
    return lengths


def get_lengths_dict(dataset) -> Dict[str, torch.Tensor]:
    lengths_dict = {}
    for dataset_key in dataset.correct_dataset_ordering:
        length_column = "nheavy" if dataset_key == "sm" else "LEN_EXIST"
        lengths_dict[dataset_key] = get_id_lengths(
            dataset.dataset_dict[dataset_key],
            dataset.ID_dict[dataset_key],
            length_column,
        )
    return lengths_dict


def adjust_samples_for_num_replicas(
    num_per_epoch_dict: Dict[str, int], num_replicas: int
) -> Dict[str, int]:
    """
    Modifies the number of examples per epoch for each dataset to be divisible by num_replicas.
    Args:
        num_per_epoch_dict (dict): Mapping from dataset name to number of examples to sample from that dataset per epoch
        num_replicas (int): The number of nodes/GPUs in the distributed training setup.

    Returns:
        dict: The modified num_per_epoch_dict where the number of examples per epoch for each dataset is divisible by num_replicas.
    """
    adjusted_dict = {}
    for dataset, num_per_epoch in num_per_epoch_dict.items():
        # Round down to nearest multiple of num_replicas
        adjusted_num = (num_per_epoch // num_replicas) * num_replicas
        adjusted_dict[dataset] = adjusted_num
    return adjusted_dict


class DistributedWeightedSampler(data.Sampler):
    def __init__(
        self,
        dataset,
        weights_dict,
        num_example_per_epoch=25600,
        fractions=dict(
            pdb=1.0,
            fb=0,
            compl=0,
            neg_compl=0,
            na_compl=0,
            neg_na_compl=0,
            distil_tf=0,
            tf=0,
            neg_tf=0,
            rna=0,
            dna=0,
            sm_compl=0,
            metal_compl=0,
            sm_compl_multi=0,
            sm_compl_covale=0,
            sm_compl_asmb=0,
            sm=0,
            atomize_pdb=0,
            atomize_complex=0,
        ),
        num_replicas=None,
        rank=None,
        batch_by_dataset=False,
        batch_by_length=False,
        **unused,
    ):
        assert (
            num_replicas is not None
        ), "Please provide the number of replicas in DistributedWeightedSampler"
        assert (
            rank is not None
        ), "Please provide the rank of the current node in DistributedWeightedSampler"

        assert (
            num_example_per_epoch % num_replicas
        ) == 0, "Please ensure that the number of examples per epoch is evenly divisible by the number of nodes"
        assert np.allclose(
            sum([v for k, v in fractions.items()]), 1.0
        ), f"Fractions of datasets add up to {sum([v for k,v in fractions.items()])}, should add up to 1.0"

        if not batch_by_dataset:
            assert (
                not batch_by_length
            ), "Cannot batch by length without also batching by dataset."

        self.dataset = dataset
        self.lengths_dict = get_lengths_dict(dataset)
        self.weights_dict = weights_dict
        self.num_replicas = num_replicas
        self.batch_by_length = batch_by_length
        self.batch_by_dataset = batch_by_dataset
        self.rank = rank
        self.epoch = 0

        self.num_per_epoch_dict = {
            dataset_name: int(num_example_per_epoch * fractions[dataset_name])
            for dataset_name in self.dataset.correct_dataset_ordering
        }
        self.num_per_epoch_dict = adjust_samples_for_num_replicas(
            self.num_per_epoch_dict, num_replicas
        )
        self.total_size = sum(self.num_per_epoch_dict.values())
        self.num_samples = self.total_size // self.num_replicas

        if rank == 0:
            print(f"Total examples:")
            for k, v in self.dataset.ID_dict.items():
                print("  " + k, ":", len(v))
            print(f"Training examples per epoch ({self.total_size} total):")
            for k, v in self.num_per_epoch_dict.items():
                print("  " + k, ":", v)

    def _sample_from_dataset(self, dataset_name, g):
        """
        Samples a specified number of sequences from the given dataset.
        Samples with replacement based on the dataset type, forcing replacement if sampling more than dataset length.
        Parameters:
            dataset_name (str): The name of the dataset to sample from.
            g (torch.Generator): A pre-seeded generator to ensure consistency across nodes.

        Returns:
            Tensor: A tensor of sampled indices from the dataset.
        """
        # Throw warning if the number of sequences to be sampled is not more than the number of sequences in the dataset
        if self.num_per_epoch_dict[dataset_name] > len(
            self.dataset.ID_dict[dataset_name]
        ):
            warnings.warn(
                f"Number of sequences to be sampled in one epoch is greater than the number of "
                f"sequences in the dataset. Must sample with replacement. Ensure that this is the desired behavior. Dataset: {dataset_name}, "
                f"Dataset length: {len(self.dataset.ID_dict[dataset_name])}, "
                f"# to be sampled: {self.num_per_epoch_dict[dataset_name]}"
            )

        # Determine if sampling with replacement based on the dataset type, forcing replacement if sampling more than dataset length
        replacement = self.num_per_epoch_dict[dataset_name] > len(
            self.dataset.ID_dict[dataset_name]
        )

        # Sample indices from the dataset based on the weights (prefer longer sequences)
        return torch.multinomial(
            self.weights_dict[dataset_name],
            self.num_per_epoch_dict[dataset_name],
            generator=g,
            replacement=replacement,
        )

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        # get indices (all models)
        indices = torch.arange(len(self.dataset))

        # weighted subsampling
        # order of datasets in this loop should match order in DistilledDataset.__getitem__()
        offset = 0
        sel_indices = torch.tensor((), dtype=int)

        if self.batch_by_dataset:
            for dataset_name in self.dataset.correct_dataset_ordering:
                if self.num_per_epoch_dict[dataset_name] > 0:
                    # Sample and adjust for offset; _sample_from_dataset handles replacement
                    sampled_indices = self._sample_from_dataset(dataset_name, g)

                    # Divide sampled_idx into num_replicas chunks and assign each chunk to a node
                    split_sampled_indices = torch.split(
                        sampled_indices, len(sampled_indices) // self.num_replicas
                    )

                    assert all(
                        [
                            len(x) == len(split_sampled_indices[0])
                            for x in split_sampled_indices
                        ]
                    )
                    indices_for_rank = split_sampled_indices[self.rank]

                    # If also batching by sequence length, sort the indices by length
                    if self.batch_by_length:
                        lengths_of_dataset = self.lengths_dict[dataset_name]
                        lengths_per_rank = lengths_of_dataset[indices_for_rank]
                        sorted_length_order = torch.argsort(lengths_per_rank)
                        indices_for_rank = indices_for_rank[sorted_length_order]

                    # Add the global dataset offset to the indices
                    indices_for_rank += offset

                    # Add the sampled indices to the running tensor based on the node rank
                    sel_indices = torch.cat((sel_indices, indices_for_rank))
                offset += len(self.dataset.ID_dict[dataset_name])

            # For each node, the indices are shuffled with the same seed, and so will draw from the same datasets in the same order
            indices = sel_indices[torch.randperm(len(sel_indices), generator=g)]
        else:
            # Standard implementation of WeightedDistributedSampler without batching by dataset or length
            for dataset_name in self.dataset.correct_dataset_ordering:
                if self.num_per_epoch_dict[dataset_name] > 0:
                    sampled_idx = self._sample_from_dataset(dataset_name, g)
                    sel_indices = torch.cat(
                        (sel_indices, indices[sampled_idx + offset])
                    )
            offset += len(self.dataset.ID_dict[dataset_name])

            # shuffle indices
            indices = sel_indices[torch.randperm(len(sel_indices), generator=g)]

            # per each gpu
            indices = indices[self.rank : self.total_size : self.num_replicas]

        assert len(indices) == self.num_samples
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class DistributedWeightedSamplerOld(data.Sampler):
    def __init__(
        self,
        dataset,
        weights_dict,
        num_example_per_epoch=25600,
        fractions=OrderedDict(
            pdb=1.0,
            fb=0,
            compl=0,
            neg_compl=0,
            na_compl=0,
            neg_na_compl=0,
            distil_tf=0,
            tf=0,
            neg_tf=0,
            rna=0,
            dna=0,
            sm_compl=0,
            metal_compl=0,
            sm_compl_multi=0,
            sm_compl_covale=0,
            sm_compl_asmb=0,
            sm=0,
            atomize_pdb=0,
            atomize_complex=0,
        ),
        num_replicas=None,
        rank=None,
        datasets_with_replacement=[
            "pdb",
            "fb",
            "compl",
            "neg_compl",
            "na_compl",
            "neg_na_compl",
            "distil_tf",
            "tf",
            "neg_tf",
            "rna",
            "dna",
        ],
        lengths=None,
        batch_by_dataset=False,
        batch_by_length=False,
        **unused,
    ):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()

        assert (
            num_example_per_epoch % num_replicas
        ) == 0, "Please ensure that the number of examples per epoch is evenly divisible by the number of nodes"
        assert np.allclose(
            sum([v for k, v in fractions.items()]), 1.0
        ), f"Fractions of datasets add up to {sum([v for k,v in fractions.items()])}, should add up to 1.0"

        # Load lengths into a tensor, if file exists
        if lengths is not None and os.path.isfile(lengths):
            lengths = torch.load(lengths)
        else:
            if lengths is not None:
                warnings.warn(
                    f"Lengths file {lengths} does not exist. Ignoring lengths file."
                )
            lengths = None

        if batch_by_length:
            assert (
                lengths is not None
            ), "If batching by length, must pass a valid lengths tensor."

        if not batch_by_dataset:
            assert (
                not batch_by_length
            ), "Cannot batch by length without also batching by dataset."

        self.dataset = dataset
        self.weights_dict = weights_dict
        self.num_replicas = num_replicas
        self.lengths = lengths
        self.batch_by_length = batch_by_length
        self.batch_by_dataset = batch_by_dataset

        if batch_by_dataset:
            # Ensure that all GPU's can process an example from the same dataset at once
            self.num_per_epoch_dict = adjust_samples_for_num_replicas(
                OrderedDict(
                    [
                        (
                            dataset_name,
                            int(round(num_example_per_epoch * fractions[dataset_name])),
                        )
                        for dataset_name in self.dataset.dataset_dict.keys()
                    ]
                ),
                num_replicas,
            )
        else:
            self.num_per_epoch_dict = OrderedDict(
                [
                    (
                        dataset_name,
                        int(round(num_example_per_epoch * fractions[dataset_name])),
                    )
                    for dataset_name in self.dataset.dataset_dict.keys()
                ]
            )

        # Account for rounding error
        dataset_names = list(self.dataset.dataset_dict.keys())
        nonzero_dataset_names = [
            name for name in dataset_names if self.num_per_epoch_dict[name] > 0
        ]

        # Calculate the actual number of examples that will be sampled (will be a multiple of num_replicas)
        num_per_epoch_actual = sum(
            [self.num_per_epoch_dict[name] for name in nonzero_dataset_names]
        )

        # Handle remainders by rounding down to the nearest multiple of num_replicas and sampling from `pdb`
        remainder = num_example_per_epoch - num_per_epoch_actual
        remainder = remainder - (remainder % num_replicas)
        self.num_per_epoch_dict[
            nonzero_dataset_names[0]
        ] += remainder  # The first dataset is the pdb

        self.total_size = num_per_epoch_actual + remainder
        self.num_samples = self.total_size // self.num_replicas
        self.rank = rank
        self.epoch = 0
        # Sample the protein datasets with replacement to account for length weighting
        # Other datasets (e.g., small molecule datasets) will be sampled WITHOUT replacement (since LEN_EXIST is not the appropriate weighting)
        self.datasets_with_replacement = datasets_with_replacement
        if rank == 0:
            print(f"Total examples:")
            for k, v in self.dataset.ID_dict.items():
                print("  " + k, ":", len(v))
            print(f"Training examples per epoch ({self.total_size} total):")
            for k, v in self.num_per_epoch_dict.items():
                print("  " + k, ":", v)

    def _sample_from_dataset(self, dataset_name, g):
        """
        Samples a specified number of sequences from the given dataset.
        Samples with replacement based on the dataset type, forcing replacement if sampling more than dataset length.
        Parameters:
            dataset_name (str): The name of the dataset to sample from.
            g (torch.Generator): A pre-seeded generator to ensure consistency across nodes.

        Returns:
            Tensor: A tensor of sampled indices from the dataset.
        """
        # Throw warning if the number of sequences to be sampled is not more than the number of sequences in the dataset
        if self.num_per_epoch_dict[dataset_name] > len(
            self.dataset.ID_dict[dataset_name]
        ):
            warnings.warn(
                f"Number of sequences to be sampled in one epoch is greater than the number of "
                f"sequences in the dataset. Must sample with replacement. Ensure that this is the desired behavior. Dataset: {dataset_name}, "
                f"Dataset length: {len(self.dataset.ID_dict[dataset_name])}, "
                f"# to be sampled: {self.num_per_epoch_dict[dataset_name]}"
            )

        # Determine if sampling with replacement based on the dataset type, forcing replacement if sampling more than dataset length
        replacement = (
            dataset_name in self.datasets_with_replacement
            or self.num_per_epoch_dict[dataset_name]
            > len(self.dataset.ID_dict[dataset_name])
        )

        # Sample indices from the dataset based on the weights (prefer longer sequences)
        return torch.multinomial(
            self.weights_dict[dataset_name],
            self.num_per_epoch_dict[dataset_name],
            generator=g,
            replacement=replacement,
        )

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        # get indices (all models)
        indices = torch.arange(len(self.dataset))

        # weighted subsampling
        # order of datasets in this loop should match order in DistilledDataset.__getitem__()
        offset = 0
        sel_indices = torch.tensor((), dtype=int)

        if self.batch_by_dataset:
            for dataset_name in self.dataset.dataset_dict.keys():
                if self.num_per_epoch_dict[dataset_name] > 0:
                    # Sample and adjust for offset; _sample_from_dataset handles replacement
                    sampled_idx = self._sample_from_dataset(dataset_name, g) + offset

                    # Divide sampled_idx into num_replicas chunks and assign each chunk to a node
                    sampled_idx_split = torch.split(
                        sampled_idx, len(sampled_idx) // self.num_replicas
                    )
                    assert all(
                        [len(x) == len(sampled_idx_split[0]) for x in sampled_idx_split]
                    )

                    # If also batching by sequence length, sort the indices by length
                    if self.batch_by_length and self.lengths is not None:
                        sampled_idx_split = [
                            x[torch.argsort(self.lengths[x])] for x in sampled_idx_split
                        ]

                    # Add the sampled indices to the running tensor based on the node rank
                    sel_indices = torch.cat(
                        (sel_indices, indices[sampled_idx_split[self.rank]])
                    )
                offset += len(self.dataset.ID_dict[dataset_name])

            # For each node, the indices are shuffled with the same seed, and so will draw from the same datasets in the same order
            indices = sel_indices[torch.randperm(len(sel_indices), generator=g)]
        else:
            # Standard implementation of WeightedDistributedSampler without batching by dataset or length
            for dataset_name in self.dataset.dataset_dict.keys():
                if self.num_per_epoch_dict[dataset_name] > 0:
                    sampled_idx = self._sample_from_dataset(dataset_name, g)
                    sel_indices = torch.cat(
                        (sel_indices, indices[sampled_idx + offset])
                    )
            offset += len(self.dataset.ID_dict[dataset_name])

            # shuffle indices
            indices = sel_indices[torch.randperm(len(sel_indices), generator=g)]

            # per each gpu
            indices = indices[self.rank : self.total_size : self.num_replicas]

        # print('rank',self.rank,': expecting',self.num_samples,'examples, drew',len(indices),'examples')
        assert (
            len(indices) == self.num_samples
        )  # more stringent, switch with line above during debugging
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


sampler_factory = {
    "DistributedWeightedSampler": DistributedWeightedSampler,
    "DistributedWeightedSamplerOld": DistributedWeightedSamplerOld,
}
