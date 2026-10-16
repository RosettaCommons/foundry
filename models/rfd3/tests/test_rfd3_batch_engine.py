from pathlib import Path
from types import SimpleNamespace

import torch
from rfd3.engine import RFD3InferenceEngine
from rfd3_batch_fixtures import example, tiny_model


def engine(model):
    obj = object.__new__(RFD3InferenceEngine)
    obj.compile_cuda_graphs = True
    obj.compile_shape_policy = "batch_max"
    obj.inference_batch_size = 3
    obj.atom_padding_multiple = 1
    obj.token_padding_multiple = 1
    obj.compile_model = False
    obj.seed = 17
    obj.ckpt_path = Path("/tmp/test-checkpoint.pt")
    obj.dump_trajectories = False
    obj.dump_prediction_metadata_json = True
    obj.align_trajectory_structures = False
    obj.out_dir = None
    obj.trainer = SimpleNamespace(
        state={"model": model}, fabric=SimpleNamespace(to_device=lambda x: x)
    )

    def build(output, ex):
        assert output["X_L"].shape == ex["coord_atom_lvl_to_be_noised"].shape
        assert output["sequence_logits_I"].shape[-2] == len(ex["feats"]["restype"])
        return list(output["X_L"]), {i: {} for i in range(len(output["X_L"]))}

    obj.trainer._build_predicted_atom_array_stack = build
    return obj


def test_engine_outputs_every_real_example_and_sample_once():
    obj = engine(tiny_model())
    exs = [example(), example((1, 2), seed=8)]
    result = obj._model_forward_batch(exs)
    assert set(result) == {ex["example_id"] for ex in exs}
    for ex in exs:
        outputs = result[ex["example_id"]]
        assert len(outputs) == 3
        assert [o.example_id for o in outputs] == [
            f"{ex['example_id']}_model_{d}" for d in range(3)
        ]
        assert all(
            o.atom_array.shape == ex["coord_atom_lvl_to_be_noised"][0].shape
            for o in outputs
        )


def test_engine_cfg_uses_different_reference_lengths():
    model = tiny_model()
    model.use_classifier_free_guidance = True
    model.cfg_features = ["active_donor"]
    model.inference_sampler.sampler.use_classifier_free_guidance = True
    model.inference_sampler.sampler.cfg_scale = 1.5
    obj = engine(model)
    exs = [example(), example((1, 2), seed=8)]
    exs[0]["feats"]["is_motif_atom_unindexed"][3:] = True
    exs[0]["feats"]["is_motif_token_unindexed"][2:] = True
    result = obj._model_forward_batch(exs)
    assert len(result) == 2
    assert all(
        torch.isfinite(o.atom_array).all()
        for outputs in result.values()
        for o in outputs
    )


def test_trajectory_axes_and_noisy_denoised_labels(monkeypatch):
    obj = engine(tiny_model())
    obj.dump_trajectories = True
    ex = example()
    ex["atom_array"] = object()
    noisy = [torch.full((3, 6, 3), float(step)) for step in range(2)]
    denoised = [x + 10 for x in noisy]
    monkeypatch.setattr("rfd3.engine._reshape_trajectory", lambda x, align: x)
    monkeypatch.setattr(
        "rfd3.engine.build_stack_from_atom_array_and_batched_coords", lambda x, aa: x
    )
    output = dict(
        network_output=dict(X_noisy_L_traj=noisy, X_denoised_L_traj=denoised),
        predicted_atom_array_stack=[None] * 3,
        prediction_metadata={i: {} for i in range(3)},
    )
    result = obj._format_model_output(ex, output)
    for d, out in enumerate(result):
        torch.testing.assert_close(out.noisy_trajectory_stack, torch.stack(noisy)[:, d])
        torch.testing.assert_close(
            out.denoised_trajectory_stack, torch.stack(denoised)[:, d]
        )


def test_distributed_partition_does_not_duplicate_tail(monkeypatch):
    all_examples = [example(seed=i) for i in range(5)]

    class Dataset:
        def __init__(self, **kwargs):
            pass

        def __len__(self):
            return len(all_examples)

        def idx_to_id(self, idx):
            return all_examples[idx]["example_id"]

        def __getitem__(self, idx):
            return all_examples[idx]

    monkeypatch.setattr("rfd3.inference.datasets.ContigJsonDataset", Dataset)
    names = []
    for rank in range(2):
        obj = engine(tiny_model())
        obj.pipeline = None
        obj.trainer.fabric.global_rank = rank
        obj.trainer.fabric.world_size = 2
        obj._model_forward_batch = lambda xs: {x["example_id"]: [] for x in xs}
        result = obj._run_multi_batched({})
        names.extend(result)
    assert sorted(names) == sorted(x["example_id"] for x in all_examples)


def test_multichain_legacy_fallback_is_explicit():
    obj = engine(tiny_model())
    ex = example((1, 1, 1, 1))
    ex["feats"]["asym_id"] = torch.arange(4)
    assert "more than three chains" in obj._batch_fallback_reason(ex)
