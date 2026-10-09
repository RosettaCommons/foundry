from unittest.mock import patch

import torch
from rfd3.inference.batching import collate_examples
from rfd3.model.batched_sampler import ExampleRandomness, augment, masked_align, sample
from rfd3.model.inference_sampler import SampleDiffusionWithMotif
from rfd3_batch_fixtures import example

from foundry.utils.alignment import weighted_rigid_align


class StubDenoiser:
    def __call__(self, X_noisy_L, t, f, **kwargs):
        return {"X_L": X_noisy_L / 2}

    forward_batched = __call__


def test_rollout_matches_legacy_with_identical_noise():
    examples = [example(), example((1, 2), seed=8)]
    batch = collate_examples(examples, atom_capacity=9, token_capacity=4)
    sampler = SampleDiffusionWithMotif(num_timesteps=4, allow_realignment=False)
    noise = [torch.randn_like(batch["coords"]) for _ in range(4)]

    def source(tag, template, atomwise=True):
        return noise[0 if tag == "initial" else int(tag.split("_")[1]) + 1]

    with torch.no_grad():
        actual = sample(
            sampler,
            StubDenoiser(),
            batch["coords"],
            batch["f"],
            {},
            noise_source=source,
        )
        for b, ex in enumerate(examples):
            n = len(ex["feats"]["atom_to_token_map"])
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
            torch.testing.assert_close(actual["X_L"][b, :, :n], expected["X_L"])
            for got, ref in zip(actual["X_noisy_L_traj"], expected["X_noisy_L_traj"]):
                torch.testing.assert_close(got[b, :, :n], ref)
        for key in ("X_L",):
            assert (actual[key][0, :, 6:] == 0).all()
            assert (actual[key][1, :, 3:] == 0).all()


def test_alignment_matches_weighted_kabsch_and_empty_rows_are_finite():
    x, y = torch.randn(3, 2, 7, 3), torch.randn(3, 2, 7, 3)
    mask = torch.tensor(
        [[1, 1, 1, 1, 0, 0, 0], [0] * 7, [1, 0, 0, 0, 0, 0, 0]], dtype=torch.bool
    )
    out = masked_align(x, y, mask)
    torch.testing.assert_close(
        out[0], weighted_rigid_align(x[0], y[0], mask[0]), atol=2e-6, rtol=2e-5
    )
    torch.testing.assert_close(out[1], y[1])
    torch.testing.assert_close(out[2, :, 0], x[2, :, 0])
    assert torch.isfinite(out).all()


def test_augmentation_centers_each_example_without_padding():
    batch = collate_examples([example(), example((1, 2), seed=8)], atom_capacity=9)
    x, f = batch["coords"], batch["f"]
    rotation = torch.eye(3).expand(2, 3, 3, 3)
    out = augment(x, x, f, rotation, torch.zeros(2, 3, 1, 3), center_option="all")
    for b, n in enumerate((6, 3)):
        torch.testing.assert_close(
            out[b, :, :n], x[b, :, :n] - x[b, :, :n].mean(-2, keepdim=True)
        )
        assert (out[b, :, n:] == 0).all()


def test_randomness_is_independent_of_peers_padding_and_order():
    ex = [example(), example((1, 2), seed=8)]
    batch = collate_examples(ex, atom_capacity=9)
    rng = ExampleRandomness(ex, seed=91)
    out = rng("step_2", batch["coords"])
    rev = collate_examples(list(reversed(ex)), atom_capacity=12)
    rev_out = ExampleRandomness(list(reversed(ex)), seed=91)("step_2", rev["coords"])
    torch.testing.assert_close(out[0, :, :6], rev_out[1, :, :6])
    torch.testing.assert_close(out[1, :, :3], rev_out[0, :, :3])
    assert not torch.equal(out[0, 0], out[0, 1])
