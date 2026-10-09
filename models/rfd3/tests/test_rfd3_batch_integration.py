"""Opt-in local checkpoint smoke test; never downloads model weights."""

import copy
import json
import os

import numpy as np
import pytest
import torch
from biotite.structure import get_residue_count
from rfd3.engine import RFD3InferenceConfig, RFD3InferenceEngine


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RFD3_RUN_BATCH_INTEGRATION") != "1",
    reason="Set RFD3_RUN_BATCH_INTEGRATION=1 with a locally installed checkpoint",
)
def test_checkpoint_mixed_length_batch_writes_valid_structures(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps({"short": {"length": 16}, "long": {"length": 18}}))
    config = RFD3InferenceConfig(
        inference_batch_size=2,
        diffusion_batch_size=1,
        atom_padding_multiple=256,
        token_padding_multiple=32,
        seed=42,
        skip_existing=False,
        inference_sampler={"num_timesteps": 3, "n_recycle": 1},
        prevalidate_inputs=False,
    )
    engine = RFD3InferenceEngine(**config)
    groups = []
    original = engine._model_forward_batch

    def record_batch(examples):
        groups.append(len(examples))
        return original(examples)

    monkeypatch.setattr(engine, "_model_forward_batch", record_batch)
    outputs = engine.run(inputs=str(inputs), n_batches=1, out_dir=None)
    assert groups == [2]  # genuine mixed-example execution, not two serial calls
    assert len(outputs) == 2
    for name, designs in outputs.items():
        assert len(designs) == 1
        design = designs[0]
        assert np.isfinite(design.atom_array.coord).all()
        assert get_residue_count(design.atom_array) == (16 if "short" in name else 18)
        design.dump(tmp_path)
        assert (tmp_path / f"{design.example_id}.cif.gz").is_file()

    from rfd3.inference.batching import collate_examples, prepare_attention
    from rfd3.inference.datasets import ContigJsonDataset
    from rfd3.model.layers.batched import initialize

    dataset = ContigJsonDataset(
        data={"a": {"length": 16}, "b": {"length": 18}},
        transform=engine.pipeline,
        name="parity",
        cif_parser_args=None,
        subset_to_keys=None,
        eval_every_n=1,
    )
    examples = [dataset[0], dataset[1]]
    model = engine._rfd3_net()
    dm = model.diffusion_module
    with torch.no_grad():
        for ex in examples:
            ex["coord_atom_lvl_to_be_noised"] = (
                torch.randn_like(ex["coord_atom_lvl_to_be_noised"]) * 3
            )
        batch = collate_examples(examples, atom_capacity=256, token_capacity=32)
        f = prepare_attention(
            batch["f"],
            atom_keys=dm.n_attn_keys,
            atom_neighbors=dm.n_attn_seq_neighbours,
            token_keys=dm.diffusion_transformer.n_keys,
            token_neighbors=dm.diffusion_transformer.n_local_tokens,
        )
        init = initialize(model.token_initializer, f)
        actual = dm.forward_batched(
            batch["coords"], torch.ones(2, 1) * 2, f, **init, n_recycle=1
        )
        for b, ex in enumerate(examples):
            lf = copy.deepcopy(ex["feats"])
            expected = dm(
                ex["coord_atom_lvl_to_be_noised"],
                torch.ones(1) * 2,
                lf,
                **model.token_initializer(lf),
                n_recycle=1,
            )
            for key, length in [
                ("X_L", len(lf["atom_to_token_map"])),
                ("sequence_logits_I", len(lf["restype"])),
            ]:
                got = actual[key][b, :, :length]
                print(
                    "CHECKPOINT_PARITY",
                    b,
                    key,
                    "max_abs_error",
                    (got - expected[key]).abs().max().item(),
                )
                torch.testing.assert_close(got, expected[key], atol=0.025, rtol=0.025)
