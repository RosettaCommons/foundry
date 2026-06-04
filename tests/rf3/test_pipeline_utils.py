"""Unit tests for rf3.data.pipeline_utils.build_ground_truth_distogram_transform.

The builder assembles the per-token-group noise samplers for the noised
ground-truth distogram conditioning. The contracts pinned here:

- For inference the sampler for each token group is a *constant* function
  returning ``noise_scale``-filled tensors; the atomized and not-atomized
  groups must capture their own scale (the closure-capture correctness that the
  ``is not None`` narrowing fix depends on).
- A ``None`` noise scale for a group skips that group entirely (no sampler is
  registered for it).
- ``allowed_chain_types_for_conditioning`` is threaded onto the transform
  verbatim, including the ``None`` ("disable conditioning") default.
"""

import torch
from atomworks.enums import ChainType
from rf3.data.ground_truth_template import TokenGroupNoiseScaleSampler
from rf3.data.pipeline_utils import build_ground_truth_distogram_transform

_BOTH = {"atomized": 2.0, "not_atomized": 3.0}


def _inference_samplers(scales):
    transform = build_ground_truth_distogram_transform(
        template_noise_scales=scales, is_inference=True
    )
    sampler = transform.noise_scale_distribution
    assert isinstance(sampler, TokenGroupNoiseScaleSampler)
    return [fn for _, fn in sampler.mask_and_sampling_fns]


def test_inference_samplers_are_constant_per_group():
    atomized_fn, not_atomized_fn = _inference_samplers(_BOTH)
    assert torch.equal(atomized_fn((4,)), torch.full((4,), 2.0))
    assert torch.equal(not_atomized_fn((2,)), torch.full((2,), 3.0))


def test_inference_skips_none_atomized():
    (only_fn,) = _inference_samplers({"atomized": None, "not_atomized": 3.0})
    # The surviving sampler must be the not-atomized one (3.0), not a stale capture.
    assert torch.equal(only_fn((3,)), torch.full((3,), 3.0))


def test_inference_skips_none_not_atomized():
    (only_fn,) = _inference_samplers({"atomized": 2.0, "not_atomized": None})
    assert torch.equal(only_fn((3,)), torch.full((3,), 2.0))


def test_inference_both_none_yields_no_samplers():
    assert _inference_samplers({"atomized": None, "not_atomized": None}) == []


def test_training_branch_registers_one_sampler_per_non_none_group():
    transform = build_ground_truth_distogram_transform(
        template_noise_scales={"atomized": 2.0, "not_atomized": None},
        is_inference=False,
    )
    sampler = transform.noise_scale_distribution
    assert isinstance(sampler, TokenGroupNoiseScaleSampler)
    assert len(sampler.mask_and_sampling_fns) == 1


def test_allowed_chain_types_defaults_to_none():
    transform = build_ground_truth_distogram_transform(
        template_noise_scales=_BOTH, is_inference=True
    )
    assert transform.allowed_chain_types is None


def test_allowed_chain_types_is_threaded_through():
    allowed = ChainType.get_all_types()[:1]
    transform = build_ground_truth_distogram_transform(
        template_noise_scales=_BOTH,
        allowed_chain_types_for_conditioning=allowed,
        is_inference=True,
    )
    assert transform.allowed_chain_types == allowed
