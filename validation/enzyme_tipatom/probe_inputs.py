"""Archive CPU preprocessed enzyme features with the public padded loader seed rules."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from rfd3.engine import RFD3InferenceConfig, RFD3InferenceEngine
from rfd3.inference.datasets import ContigJsonDataset
from rfd3.inference.loading import iter_inference_examples

p = argparse.ArgumentParser()
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
root = Path(__file__).resolve().parent
ckpt = (
    "/mnt/home/jiechen7/other/foundry-pr397-precision/audit_artifacts/rfd3_latest.ckpt"
)
e = RFD3InferenceEngine(
    **RFD3InferenceConfig(
        ckpt_path=ckpt,
        diffusion_batch_size=2,
        compile_model=True,
        seed=11,
        prevalidate_inputs=True,
    )
)
cfg = e._override_checkpoint_config(
    torch.load(ckpt, map_location="cpu", mmap=True, weights_only=False)["train_cfg"]
)
e._construct_pipeline(cfg)
e._set_out_dir(None)
a.out.mkdir(parents=True, exist_ok=True)
for seed in [11, 22, 33, 44]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    e.seed = seed
    inputs = e._canonicalize_inputs(str(root / "input.json"))
    specs = e._multiply_specifications(inputs=inputs, n_batches=1)
    d = ContigJsonDataset(
        data=specs,
        transform=e.pipeline,
        name="inference-dataset",
        cif_parser_args=None,
        subset_to_keys=None,
        eval_every_n=1,
    )
    x = next(
        iter(
            iter_inference_examples(
                d,
                seed=seed,
                rank=0,
                world_size=1,
                num_workers=0,
                batch_size=1,
                pin_memory=False,
            )
        )
    )
    atom = x["atom_array"]
    torch.save(
        {
            "feats": x["feats"],
            "coords": x["coord_atom_lvl_to_be_noised"],
            "atom_coords": atom.coord,
            "bonds": atom.bonds.as_array() if atom.bonds else None,
            "annotations": {
                k: atom.get_annotation(k) for k in atom.get_annotation_categories()
            },
            "example_id": x["example_id"],
        },
        a.out / f"s{seed}.pt",
    )
    print(seed, x["example_id"], len(atom), flush=True)
