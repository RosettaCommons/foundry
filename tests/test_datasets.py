"""Unit tests for foundry.utils.datasets.

These helpers build the training/validation dataloading stack from hydra configs:
selecting a sampler from config keys, wrapping a dataset+sampler with fallback
dataloading, and converting a non-distributed sampler into a distributed one
before assembling the final ``DataLoader``. The contracts worth pinning are the
control flow and the input-validation guards, not the heavy atomworks dataset
machinery, so the fixtures are tiny map-style datasets and stock torch samplers.
"""

import pandas as pd
import pytest
import torch
from atomworks.ml.datasets import FallbackDatasetWrapper
from atomworks.ml.samplers import (
    DistributedMixedSampler,
    FallbackSamplerWrapper,
    MixedSampler,
)
from omegaconf import OmegaConf
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

from foundry.utils.datasets import (
    assemble_distributed_loader,
    instantiate_single_dataset_and_sampler,
    wrap_dataset_and_sampler_with_fallbacks,
)


class _Tiny(Dataset):
    """Minimal map-style dataset: indexable, with a ``__len__`` and a ``.data`` frame."""

    def __init__(self, n: int = 4):
        self._n = n
        self.data = pd.DataFrame({"x": range(n)})

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i: int) -> int:
        return i


class _StubSampler(Sampler):
    """A sampler with no required init args, to verify the ``sampler`` config branch."""

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return 0


def _weights_from_df(dataset_df: pd.DataFrame) -> torch.Tensor:
    """Hydra target for the ``weights`` config key; receives ``dataset_df`` like the real ones."""
    return torch.tensor([0.5, 0.3, 0.2])


# fully-qualified targets for hydra to import (module is already in sys.modules under __name__)
_DATASET_TARGET = f"{__name__}._Tiny"
_WEIGHTS_TARGET = f"{__name__}._weights_from_df"
_SAMPLER_TARGET = f"{__name__}._StubSampler"


# --------------------------------------------------------------------------------------
# wrap_dataset_and_sampler_with_fallbacks
# --------------------------------------------------------------------------------------


def test_wrap_uses_fallback_sampler_weights_when_present():
    """A weighted fallback sampler's own weights are reused (the `hasattr` fix).

    Regression for the latent bug where `"weights" in sampler` iterated the sampler's
    integer indices and never matched the string, so a weighted sampler silently fell
    back to uniform weights.
    """
    dataset, fallback = _Tiny(3), _Tiny(3)
    weighted = WeightedRandomSampler(
        weights=torch.tensor([0.1, 0.2, 0.7]), num_samples=3, replacement=True
    )

    _, wrapped_sampler = wrap_dataset_and_sampler_with_fallbacks(
        dataset, SequentialSampler(dataset), fallback, weighted, n_fallback_retries=2
    )

    assert wrapped_sampler.fallback_sampler.weights.tolist() == pytest.approx(
        [0.1, 0.2, 0.7]
    )


def test_wrap_uses_uniform_weights_when_sampler_has_no_weights():
    """A sampler without `.weights` yields uniform weights sized to the fallback dataset."""
    dataset, fallback = _Tiny(3), _Tiny(5)

    _, wrapped_sampler = wrap_dataset_and_sampler_with_fallbacks(
        dataset,
        SequentialSampler(dataset),
        fallback,
        SequentialSampler(fallback),
        n_fallback_retries=2,
    )

    assert wrapped_sampler.fallback_sampler.weights.tolist() == [1.0] * 5


def test_wrap_returns_fallback_wrapper_types():
    dataset, fallback = _Tiny(3), _Tiny(3)

    wrapped_dataset, wrapped_sampler = wrap_dataset_and_sampler_with_fallbacks(
        dataset,
        SequentialSampler(dataset),
        fallback,
        SequentialSampler(fallback),
        n_fallback_retries=3,
    )

    assert isinstance(wrapped_dataset, FallbackDatasetWrapper)
    assert isinstance(wrapped_sampler, FallbackSamplerWrapper)
    assert wrapped_sampler.n_fallback_retries == 3


# --------------------------------------------------------------------------------------
# instantiate_single_dataset_and_sampler
# --------------------------------------------------------------------------------------


def test_instantiate_weights_only_builds_weighted_sampler():
    """`weights` without `sampler` -> WeightedRandomSampler from the instantiated weights."""
    cfg = OmegaConf.create(
        {
            "dataset": {"_target_": _DATASET_TARGET, "n": 3},
            "weights": {"_target_": _WEIGHTS_TARGET},
        }
    )
    result = instantiate_single_dataset_and_sampler(cfg)

    assert isinstance(result["sampler"], WeightedRandomSampler)
    assert result["sampler"].num_samples == 3
    assert result["sampler"].weights.tolist() == pytest.approx([0.5, 0.3, 0.2])


def test_instantiate_sampler_only_uses_provided_sampler():
    """`sampler` without `weights` -> that sampler is instantiated verbatim."""
    cfg = OmegaConf.create(
        {
            "dataset": {"_target_": _DATASET_TARGET, "n": 3},
            "sampler": {"_target_": _SAMPLER_TARGET},
        }
    )
    result = instantiate_single_dataset_and_sampler(cfg)

    assert isinstance(result["sampler"], _StubSampler)


def test_instantiate_neither_key_falls_back_to_uniform():
    """Neither `weights` nor `sampler` -> uniform WeightedRandomSampler over the dataset."""
    cfg = OmegaConf.create({"dataset": {"_target_": _DATASET_TARGET, "n": 4}})
    result = instantiate_single_dataset_and_sampler(cfg)

    assert isinstance(result["sampler"], WeightedRandomSampler)
    assert result["sampler"].num_samples == 4
    assert result["sampler"].weights.tolist() == [1.0] * 4


def test_instantiate_both_keys_falls_back_to_uniform():
    """Providing BOTH `weights` and `sampler` falls through to uniform weights (not either one)."""
    cfg = OmegaConf.create(
        {
            "dataset": {"_target_": _DATASET_TARGET, "n": 4},
            "weights": {"_target_": _WEIGHTS_TARGET},
            "sampler": {"_target_": _SAMPLER_TARGET},
        }
    )
    result = instantiate_single_dataset_and_sampler(cfg)

    assert isinstance(result["sampler"], WeightedRandomSampler)
    assert result["sampler"].weights.tolist() == [1.0] * 4


# --------------------------------------------------------------------------------------
# assemble_distributed_loader
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("sampler_cls", [RandomSampler, SequentialSampler])
def test_assemble_random_sequential_requires_rank_world_size(sampler_cls):
    dataset = _Tiny(4)
    with pytest.raises(AssertionError, match="Rank and world_size must be provided"):
        assemble_distributed_loader(dataset, sampler=sampler_cls(dataset))


def test_assemble_converts_sequential_to_distributed_sampler():
    dataset = _Tiny(4)
    loader = assemble_distributed_loader(
        dataset, sampler=SequentialSampler(dataset), rank=0, world_size=1
    )

    assert isinstance(loader, DataLoader)
    assert isinstance(loader.sampler, DistributedSampler)


def test_assemble_mixed_sampler_requires_distributed_args():
    dataset = _Tiny(4)
    mixed = MixedSampler(
        datasets_info=[
            {
                "sampler": SequentialSampler(dataset),
                "dataset": dataset,
                "probability": 1.0,
            }
        ],
        n_examples_per_epoch=None,
    )
    with pytest.raises(AssertionError, match="must be provided for MixedSampler"):
        assemble_distributed_loader(dataset, sampler=mixed)


def test_assemble_converts_mixed_to_distributed_mixed_sampler():
    dataset = _Tiny(4)
    mixed = MixedSampler(
        datasets_info=[
            {
                "sampler": SequentialSampler(dataset),
                "dataset": dataset,
                "probability": 1.0,
            }
        ],
        n_examples_per_epoch=None,
    )
    loader = assemble_distributed_loader(
        dataset, sampler=mixed, rank=0, world_size=1, n_examples_per_epoch=4
    )

    assert isinstance(loader.sampler, DistributedMixedSampler)


def test_assemble_rejects_unknown_sampler_type():
    """A non-distributed sampler that isn't Mixed/Random/Sequential is rejected."""
    dataset = _Tiny(3)
    bare = WeightedRandomSampler(weights=torch.ones(3), num_samples=3, replacement=True)
    with pytest.raises(AssertionError, match="Invalid sampler type"):
        assemble_distributed_loader(dataset, sampler=bare)


def test_assemble_rejects_rank_with_already_distributed_sampler():
    dataset = _Tiny(4)
    dist = DistributedSampler(dataset, num_replicas=1, rank=0)
    with pytest.raises(AssertionError, match="will have no effect"):
        assemble_distributed_loader(dataset, sampler=dist, rank=0, world_size=1)


def test_assemble_passes_through_distributed_sampler():
    dataset = _Tiny(4)
    dist = DistributedSampler(dataset, num_replicas=1, rank=0)
    loader = assemble_distributed_loader(dataset, sampler=dist)

    assert loader.sampler is dist
    assert loader.dataset is dataset


def test_assemble_subset_with_no_sampler():
    """A pre-subset dataset with sampler=None is loaded as-is (no distributed sampler)."""
    subset = Subset(_Tiny(4), [0, 1])
    loader = assemble_distributed_loader(subset, sampler=None)

    assert loader.dataset is subset


def test_assemble_wraps_with_fallbacks_when_configured():
    dataset = _Tiny(4)
    dist = DistributedSampler(dataset, num_replicas=1, rank=0)
    loader = assemble_distributed_loader(
        dataset, sampler=dist, loader_cfg={"n_fallback_retries": 2}
    )

    assert isinstance(loader.sampler, FallbackSamplerWrapper)
    assert isinstance(loader.dataset, FallbackDatasetWrapper)


def test_assemble_forwards_dataloader_params():
    dataset = _Tiny(4)
    dist = DistributedSampler(dataset, num_replicas=1, rank=0)
    loader = assemble_distributed_loader(
        dataset, sampler=dist, loader_cfg={"dataloader_params": {"batch_size": 2}}
    )

    assert loader.batch_size == 2


if __name__ == "__main__":
    pytest.main(["-v", __file__])
