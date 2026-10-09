"""Spawned preprocessing preserves ordering, sharding and per-example RNG."""

import random

import numpy as np
import pytest
import torch
from rfd3.inference.loading import iter_inference_examples


class RandomTransform:
    def __init__(self):
        self.route = lambda index: dict(
            index=index,
            python=random.random(),
            numpy=float(np.random.rand()),
            torch=float(torch.rand(())),
        )

    def __call__(self, index):
        return self.route(index)


class Examples:
    def __init__(self):
        self.transform = RandomTransform()

    def __len__(self):
        return 7

    def idx_to_id(self, index):
        return f"example-{index}"

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError(index)
        return self.transform(index)


class TensorExamples(Examples):
    def __getitem__(self, index):
        row = super().__getitem__(index)
        row["tensor"] = torch.full((8,), row["torch"])
        return row


def collect(workers, rank=0, world_size=1):
    return list(
        iter_inference_examples(
            Examples(),
            seed=123,
            num_workers=workers,
            batch_size=32,
            rank=rank,
            world_size=world_size,
            transform_config={"_target_": f"{__name__}.RandomTransform"},
        )
    )


def test_spawn_preserves_features_order_and_parent_rng():
    expected = collect(0)
    py_state, np_state, torch_state = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )
    assert collect(2) == expected
    assert random.getstate() == py_state
    actual_np = np.random.get_state()
    assert actual_np[0] == np_state[0]
    np.testing.assert_array_equal(actual_np[1], np_state[1])
    assert actual_np[2:] == np_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_rank_shards_are_disjoint_and_reproducible():
    expected = collect(0)
    shards = [collect(1, rank, 2) for rank in range(2)]
    assert sorted(shards[0] + shards[1], key=lambda row: row["index"]) == expected
    assert not ({r["index"] for r in shards[0]} & {r["index"] for r in shards[1]})


def test_worker_config_required():
    with pytest.raises(ValueError, match="pipeline configuration"):
        list(iter_inference_examples(Examples(), seed=1, num_workers=1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA pinning required")
def test_pinned_prefetch_preserves_values_order_and_cpu_ownership():
    def stream(workers, pin):
        return list(iter_inference_examples(
            TensorExamples(), seed=123, num_workers=workers, pin_memory=pin,
            transform_config={"_target_": f"{__name__}.RandomTransform"},
        ))

    expected = stream(0, False)
    actual = stream(2, True)
    assert len(actual) == len(expected)
    for got, ref in zip(actual, expected):
        tensor = got.pop("tensor")
        assert tensor.device.type == "cpu" and tensor.is_pinned()
        torch.testing.assert_close(tensor, ref.pop("tensor"), rtol=0, atol=0)
        assert got == ref
