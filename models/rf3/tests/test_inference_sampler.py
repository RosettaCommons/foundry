"""Unit tests for rf3.diffusion_samplers.inference_sampler pure helpers.

``SampleDiffusion`` runs the AF3 diffusion roll-out; the pure, network-free pieces
pinned here are the noise schedule and the initial-point-cloud sampler:

- ``_construct_inference_noise_schedule`` builds the AF3 inference schedule
  ``t_hat = sigma_data * (s_max**(1/p) + t*(s_min**(1/p) - s_max**(1/p)))**p`` over
  ``num_timesteps`` points of ``t`` linearly spaced in ``[min_t, max_t]``. At ``t=0`` it
  is ``sigma_data*s_max`` and at ``t=1`` it is ``sigma_data*s_min``, decreasing in between
  (AF3 Supplement §3.7.1).
- ``SamplePartialDiffusion`` overrides the schedule to start the roll-out part-way
  through, returning the full schedule's tail from index ``partial_t``.
- ``_get_initial_structure`` returns ``c0 * N(0,1) + coords`` — Gaussian noise scaled by
  ``c0`` (derived from ``noise_schedule[0]``) added to the coords to be noised.
"""

import pytest
import torch
from rf3.diffusion_samplers.inference_sampler import (
    SampleDiffusion,
    SamplePartialDiffusion,
)

# AF3 defaults (configs are the source of truth — no defaults in the constructor),
# with a short schedule so the tests stay tiny.
_KW = dict(
    num_timesteps=8,
    min_t=0,
    max_t=1,
    sigma_data=16,
    s_min=4e-4,
    s_max=160,
    p=7,
    gamma_0=0.8,
    gamma_min=1.0,
    noise_scale=1.003,
    step_scale=1.5,
    solver="af3",
)

_CPU = torch.device("cpu")


# --- _construct_inference_noise_schedule ------------------------------------


def test_noise_schedule_length_and_endpoints():
    sched = SampleDiffusion(**_KW)._construct_inference_noise_schedule(_CPU)
    assert sched.shape == (8,)
    # t=0 -> sigma_data*s_max; t=1 -> sigma_data*s_min (the (1/p)/**p cancel at the ends).
    assert sched[0].item() == pytest.approx(16 * 160, rel=1e-4)
    assert sched[-1].item() == pytest.approx(16 * 4e-4, rel=1e-4)


def test_noise_schedule_monotonically_decreasing():
    sched = SampleDiffusion(**_KW)._construct_inference_noise_schedule(_CPU)
    assert bool((sched[1:] < sched[:-1]).all())


def test_noise_schedule_matches_af3_formula():
    sched = SampleDiffusion(**_KW)._construct_inference_noise_schedule(_CPU)
    t = torch.linspace(0, 1, 8)
    expected = 16 * (160 ** (1 / 7) + t * (4e-4 ** (1 / 7) - 160 ** (1 / 7))) ** 7
    assert torch.allclose(sched, expected, atol=1e-6)


def test_noise_schedule_is_flat_when_s_min_equals_s_max():
    # With s_min == s_max the t-dependent term vanishes, so every step is sigma_data*s.
    sched = SampleDiffusion(
        **{**_KW, "s_min": 5.0, "s_max": 5.0}
    )._construct_inference_noise_schedule(_CPU)
    assert torch.allclose(sched, torch.full((8,), 16 * 5.0), atol=1e-4)


# --- SamplePartialDiffusion._construct_inference_noise_schedule --------------


def test_partial_schedule_is_tail_of_full():
    full = SampleDiffusion(**_KW)._construct_inference_noise_schedule(_CPU)
    partial = SamplePartialDiffusion(
        partial_t=3, **_KW
    )._construct_inference_noise_schedule(_CPU)
    assert partial.shape == (8 - 3,)
    assert torch.equal(partial, full[3:])


def test_partial_schedule_rejects_t_at_or_above_num_timesteps():
    sampler = SamplePartialDiffusion(partial_t=8, **_KW)
    with pytest.raises(AssertionError, match="must be less than num_timesteps"):
        sampler._construct_inference_noise_schedule(_CPU)


# --- _get_initial_structure -------------------------------------------------


def test_initial_structure_shape_and_zero_scale():
    coords = torch.randn(4, 6, 3)
    sampler = SampleDiffusion(**_KW)
    out = sampler._get_initial_structure(
        torch.tensor(2.0), D=4, L=6, coord_atom_lvl_to_be_noised=coords
    )
    assert out.shape == (4, 6, 3)
    # c0 == 0 zeroes the noise term, so the result is exactly the coords to be noised.
    zero_scale = sampler._get_initial_structure(
        torch.tensor(0.0), D=4, L=6, coord_atom_lvl_to_be_noised=coords
    )
    assert torch.equal(zero_scale, coords)
