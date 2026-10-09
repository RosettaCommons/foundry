import torch
from rfd3.inference.batching import (
    batch_signature,
    cfg_reference_examples,
    collate_examples,
    iter_compatible_batches,
)
from rfd3.model.batched_sampler import sample
from rfd3.model.inference_sampler import SampleDiffusionWithMotif
from rfd3_batch_fixtures import example, tiny_model


def with_crop(ex, atom_start, token_start):
    ex["feats"]["is_motif_atom_unindexed"][atom_start:] = True
    ex["feats"]["is_motif_token_unindexed"][token_start:] = True
    return ex


def test_schema_cfg_handles_channel_size_equal_to_sequence_length():
    ex = with_crop(example((1, 1, 1)), 2, 2)
    ref = cfg_reference_examples([ex], ["active_donor"])[0]
    assert ref["feats"]["ref_pos"].shape == (2, 3)
    assert ref["feats"]["token_bonds"].shape == (2, 2)
    assert ref["feats"]["ref_atom_name_chars"].shape == (2, 4, 64)
    assert ex["feats"]["ref_pos"].shape == (3, 3)


def test_cfg_different_crop_lengths_have_separate_padding_masks():
    exs = [with_crop(example(), 3, 2), example((1, 2), seed=8)]
    refs = cfg_reference_examples(exs, ["active_donor"])
    batch = collate_examples(refs, atom_capacity=8, token_capacity=4)
    assert batch["f"]["atom_valid"].sum(-1).tolist() == [3, 3]
    assert batch["f"]["token_valid"].sum(-1).tolist() == [2, 2]


def test_cfg_delta_scatter_and_conditional_initializers_are_independent():
    from rfd3.model.layers.batched import initialize
    from test_rfd3_batch_model import prepared

    model = tiny_model()
    examples = [with_crop(example(), 3, 2), example((1, 2), seed=8)]
    with torch.no_grad():
        f, x = prepared(examples, model)
        ref, _ = prepared(
            cfg_reference_examples(examples, ["ref_atomwise_rasa"]), model
        )
        init = initialize(model.token_initializer, f)
        saved = {k: v.clone() for k, v in init.items()}
        ref_init = initialize(model.token_initializer, ref)
        for k in init:
            torch.testing.assert_close(init[k], saved[k])
        sampler = SampleDiffusionWithMotif(
            num_timesteps=2, use_classifier_free_guidance=True, cfg_scale=2.0
        )

        class Stub:
            def forward_batched(self, X_noisy_L, t, f, **kwargs):
                return {"X_L": torch.zeros_like(X_noisy_L)}

        actual = sample(
            sampler,
            Stub(),
            x,
            f,
            init,
            f_ref=ref,
            ref_initializer=ref_init,
            noise_source=lambda tag, template, atomwise=True: torch.zeros_like(
                template
            ),
        )
        schedule = sampler._construct_inference_noise_schedule(x.device)
        t_hat = schedule[0] * (
            1 + (sampler.gamma_0 if schedule[1] > sampler.gamma_min else 0)
        )
        delta = x / t_hat
        # Indexed prefix gets ordinary delta; cropped tail gets cfg_scale * delta.
        delta[0, :, 3:6] *= 2
        expected = x + sampler.step_scale * (schedule[1] - t_hat) * delta
        torch.testing.assert_close(actual["X_L"], expected)


def test_partial_schedules_and_shapes_partition_before_collation():
    a, b, c = example(seed=1), example(seed=2), example(seed=3)
    b["feats"]["partial_t"] = torch.ones(6) * 10
    groups = list(iter_compatible_batches([a, b, c], 2))
    assert [len(g) for g in groups] == [1, 1, 1]
    assert batch_signature(a, 8, 4)[:2] == (8, 4)
