from unittest.mock import patch

import torch
from rfd3.inference.batching import collate_examples
from rfd3.model.batched_sampler import apply_symmetry, sample
from rfd3.model.inference_sampler import SampleDiffusionWithSymmetry
from rfd3_batch_fixtures import example
from test_rfd3_batch_sampler import StubDenoiser


def symmetric(seed, n):
    ex = example((1,) * (2 * n + 1), seed=seed)
    f = ex["feats"]
    f.update(
        sym_entity_id=torch.tensor([0] * (2 * n) + [-1]),
        sym_transform_id=torch.tensor([0] * n + [1] * n + [-1]),
        is_sym_asu=torch.tensor([True] * n + [False] * (n + 1)),
        sym_transform={
            "0": (torch.eye(3), torch.zeros(3)),
            "1": (
                torch.diag(torch.tensor([-1.0, -1.0, 1.0])),
                torch.tensor([float(seed), 0.0, 0.0]),
            ),
        },
    )
    return ex


def test_per_example_symmetry_maps_match_legacy_with_different_asus_and_frames():
    exs = [symmetric(7, 2), symmetric(8, 3)]
    batch = collate_examples(exs, atom_capacity=10, token_capacity=10, batch_capacity=3)
    sampler = SampleDiffusionWithSymmetry(gamma_0=0.6)
    actual = apply_symmetry(batch["coords"], batch["f"], allow_realignment=False)
    for b, ex in enumerate(exs):
        expected = sampler.apply_symmetry_to_X_L(
            ex["coord_atom_lvl_to_be_noised"].clone(), ex["feats"]
        )
        n = expected.shape[-2]
        torch.testing.assert_close(actual[b, :, :n], expected)
        assert (actual[b, :, n:] == 0).all()
    assert (actual[2] == 0).all()


def test_symmetric_rollout_matches_legacy_with_injected_noise():
    exs = [symmetric(7, 2), symmetric(8, 3)]
    batch = collate_examples(exs, atom_capacity=10, token_capacity=10)
    noise = [torch.randn_like(batch["coords"]) for _ in range(4)]

    def source(tag, template, atomwise=True):
        return noise[0 if tag == "initial" else int(tag.split("_")[1]) + 1]

    sampler = SampleDiffusionWithSymmetry(num_timesteps=4, gamma_0=0.6)
    with torch.no_grad():
        actual = sample(
            sampler,
            StubDenoiser(),
            batch["coords"],
            batch["f"],
            {},
            noise_source=source,
        )
        for b, ex in enumerate(exs):
            n = ex["coord_atom_lvl_to_be_noised"].shape[-2]
            with patch("torch.normal", side_effect=[x[b, :, :n] for x in noise]):
                expected = sampler.sample_diffusion_like_af3(
                    f=ex["feats"],
                    diffusion_module=StubDenoiser(),
                    diffusion_batch_size=3,
                    coord_atom_lvl_to_be_noised=ex["coord_atom_lvl_to_be_noised"],
                    initializer_outputs={},
                    ref_initializer_outputs=None,
                    f_ref=None,
                )
            # The sigma=160 rollout amplifies fp32 reduction-order differences.
            torch.testing.assert_close(
                actual["X_L"][b, :, :n], expected["X_L"], atol=1e-4, rtol=1e-5
            )
