"""Ordered CPU preprocessing, optionally overlapping inference in spawned workers."""

import copy
import random

import numpy as np
import torch
from rfd3.model.batched_sampler import stable_seed
from torch.utils.data import DataLoader, Dataset


def _identity(example):
    return example


class _SeededExamples(Dataset):
    def __init__(self, dataset, indices, seed, transform_config):
        self.dataset = dataset
        self.indices = indices
        self.seed = seed
        self.transform_config = transform_config
        self._needs_transform = False

    def __len__(self):
        return len(self.indices)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["dataset"] = copy.copy(self.dataset)
        # Transform routes contain lambdas. Construct them in the CPU worker.
        state["dataset"].transform = None
        state["_needs_transform"] = True
        return state

    def __getitem__(self, index):
        if self._needs_transform:
            import hydra
            from omegaconf import OmegaConf

            self.dataset.transform = hydra.utils.instantiate(
                OmegaConf.create(self.transform_config)
            )
            self._needs_transform = False
        index = self.indices[index]
        if self.seed is None:
            return self.dataset[index]
        py_state, np_state = random.getstate(), np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[]):
                seed = stable_seed(
                    self.seed, self.dataset.idx_to_id(index), "preprocess"
                )
                random.seed(seed)
                np.random.seed(seed % (2**32))
                torch.random.default_generator.manual_seed(seed)
                return self.dataset[index]
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)


def iter_inference_examples(
    dataset,
    *,
    seed,
    rank=0,
    world_size=1,
    num_workers=0,
    transform_config=None,
    batch_size=1,
    pin_memory=False,
):
    """Workers live for this input stream; batching and all CUDA work stay upstream."""
    if num_workers < 0:
        raise ValueError("inference_num_workers must be nonnegative")
    dataset = _SeededExamples(
        dataset, range(rank, len(dataset), world_size), seed, transform_config
    )
    if not num_workers:
        yield from dataset
        return
    if transform_config is None:
        raise ValueError("Worker preprocessing requires a pipeline configuration")
    loader = DataLoader(
        dataset,
        batch_size=None,
        collate_fn=_identity,
        num_workers=num_workers,
        multiprocessing_context="spawn",
        # CUDA's pinning thread also receives/deserializes worker results while
        # the parent is occupied with sampling, instead of doing IPC on demand.
        pin_memory=pin_memory,
        prefetch_factor=max(2, (batch_size + num_workers - 1) // num_workers),
        # Do not consume the parent's model RNG when creating the iterator.
        generator=torch.Generator().manual_seed(seed if seed is not None else 0),
    )
    yield from loader
