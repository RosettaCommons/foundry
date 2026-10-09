"""Inference scheduling and asynchronous noise preserve sampling semantics."""

import pytest
import torch
from rfd3.inference.batching import collate_examples
from rfd3.model.batched_sampler import ExampleRandomness, sample
from rfd3.model.inference_sampler import SampleDiffusionWithMotif
from rfd3_batch_fixtures import example


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_schedule_thresholds_and_entropy_trajectory(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    batch = collate_examples([example()], atom_capacity=9, token_capacity=4)
    x = batch["coords"].to(device)
    f = {k: v.to(device) for k, v in batch["f"].items()}
    # Include exact equality at both cutoffs; decisions must retain strict >.
    schedule = torch.tensor([10.0, 5.0, 2.0, 0.0], device=device)
    sampler = SampleDiffusionWithMotif(
        num_timesteps=4, allow_realignment=False, gamma_min=2.0,
        cfg_t_max=5.0, use_classifier_free_guidance=True, cfg_scale=2.0,
    )
    sampler._construct_inference_noise_schedule = lambda *args: schedule
    times = []
    logits = torch.arange(32, device=device, dtype=x.dtype).expand(1, 3, 4, 32) / 10

    class Denoiser:
        def forward_batched(self, x, t, f, **kwargs):
            times.append(t.clone())
            return {"X_L": x / 2, "sequence_logits_I": logits}

    out = sample(
        sampler, Denoiser(), x, f, {}, f_ref=f, ref_initializer={},
        noise_source=lambda tag, template, atomwise=True: torch.zeros_like(template),
    )
    # No CFG call at t=5 (equal to cutoff), or at either lower timestep.
    assert len(times) == 3
    expected = [10 * (1 + sampler.gamma_0), 5.0, 2.0]
    for got, value in zip(times, expected):
        torch.testing.assert_close(got, torch.full_like(got, value))
    p = logits.softmax(-1)
    entropy = torch.where(
        f["token_valid"][:, None], -(p * (p + 1e-10).log()).sum(-1), 0
    ).cpu()
    assert len(out["sequence_entropy_traj"]) == 3
    for got in out["sequence_entropy_traj"]:
        assert got.device.type == "cpu"
        torch.testing.assert_close(got, entropy, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_async_noise_matches_cpu_rng_after_allocator_reuse():
    examples = [example(), example((1, 2), seed=8)]
    batch = collate_examples(examples, atom_capacity=9)
    rng = ExampleRandomness(examples, seed=91)
    template = batch["coords"].cuda()
    # Enqueue many calls without host waits, allowing pinned allocations to be
    # released/reused while earlier copies may still be pending.
    with torch.cuda.stream(torch.cuda.Stream()):
        actual = [rng(f"step_{i}", template) for i in range(32)]
        done = torch.cuda.Event()
        done.record()
    done.synchronize()
    for i, got in enumerate(actual):
        expected = rng(f"step_{i}", batch["coords"])
        torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=0)
