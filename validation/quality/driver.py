"""Paired quality pilot; invoke each stage with its explicitly selected source/env."""
import argparse
import copy
import importlib.util
import json
import random
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def install_pooling(path):
    spec = importlib.util.spec_from_file_location("quality_pooling", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import foundry.utils.torch as utils
    utils.scatter_mean = module.scatter_mean


def host(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {key: host(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [host(value) for value in obj]
    return obj


def cases(plan):
    return [(f"L{length}_s{seed}", length, seed) for length in plan["lengths"] for seed in plan["seeds"]]


def generate(args, plan):
    from rfd3.engine import RFD3InferenceConfig, RFD3InferenceEngine
    from rfd3.inference.datasets import ContigJsonDataset
    from biotite.structure.io.pdb import PDBFile

    engine = RFD3InferenceEngine(**RFD3InferenceConfig(
        ckpt_path=args.checkpoint, diffusion_batch_size=1, seed=42,
        compile_model=False, prevalidate_inputs=False,
        inference_sampler={"num_timesteps": plan["rfd3_steps"], "allow_realignment": False},
    ))
    engine.initialize()
    net = engine.trainer.state["model"]
    for _ in range(8):
        if hasattr(net, "diffusion_module"):
            break
        for attr in ("_forward_module", "module", "shadow", "model"):
            if hasattr(net, attr):
                net = getattr(net, attr)
                break
    net.eval()
    dm = net.diffusion_module
    sampler = net.inference_sampler.sampler
    if sampler.n_recycle is None:
        sampler.n_recycle = dm.n_recycle
    for module in net.modules():
        module.inference_kernel_backend = "torch"
    padded = args.arm.startswith("padded")
    if padded:
        engine._compile_diffusion_submodules()
        from rfd3.inference.batching import collate_examples, prepare_attention
        from rfd3.model.batched_sampler import sample
        from rfd3.model.layers.batched import initialize
    out = args.root / args.arm
    out.mkdir(exist_ok=True)
    inputs_dir = args.root / "matched_inputs"
    inputs_dir.mkdir(exist_ok=True)
    for case, length, seed in cases(plan)[:args.limit]:
        if (out / f"{case}.json").exists():
            continue
        seed_all(seed)
        saved = inputs_dir / f"{case}.pt"
        if args.arm == "native_aw2" or args.arm == "padded_aw3":
            dataset = ContigJsonDataset(data={case: {"length": str(length)}}, transform=engine.pipeline,
                name="quality", cif_parser_args=None, subset_to_keys=None, eval_every_n=1)
            example = dataset[0]
            if args.arm == "native_aw2":
                gen = torch.Generator().manual_seed(1000 + seed)
                noise = [torch.randn(example["coord_atom_lvl_to_be_noised"].shape, generator=gen) for _ in range(plan["rfd3_steps"])]
                torch.save({"example": host(example), "noise": noise}, saved)
            else:
                reference = torch.load(saved, weights_only=False)
                noise = reference["noise"]
                differences = []
                for key, value in example["feats"].items():
                    other = reference["example"]["feats"][key]
                    if isinstance(value, torch.Tensor):
                        equal = torch.equal(value.cpu(), other)
                    else:
                        equal = np.array_equal(value, other)
                    if not equal:
                        differences.append(key)
                (out / f"{case}_feature_changes.json").write_text(json.dumps(differences))
        else:
            payload = torch.load(saved, weights_only=False)
            example, noise = payload["example"], payload["noise"]
        original_atoms = example["atom_array"].copy()
        example = engine.trainer.fabric.to_device(example)
        noise = [value.cuda() for value in noise]
        f, coords = example["feats"], example["coord_atom_lvl_to_be_noised"]
        seed_all(seed + 9000)
        started = time.monotonic()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            if not padded:
                init = net.token_initializer(copy.deepcopy(f))
                with patch("torch.normal", side_effect=noise):
                    result = sampler.sample_diffusion_like_af3(f=copy.deepcopy(f), diffusion_module=dm,
                        diffusion_batch_size=1, coord_atom_lvl_to_be_noised=coords,
                        initializer_outputs=init, ref_initializer_outputs=None, f_ref=None)
                xyz = result["X_L"][0].float().cpu().numpy()
            else:
                batch = collate_examples([example], atom_capacity=2304, token_capacity=192, slot_capacity=14)
                bf = prepare_attention(batch["f"], atom_keys=dm.n_attn_keys,
                    atom_neighbors=dm.n_attn_seq_neighbours, token_keys=dm.diffusion_transformer.n_keys,
                    token_neighbors=dm.diffusion_transformer.n_local_tokens)
                init = initialize(net.token_initializer, bf)
                def noise_source(tag, template, atomwise=True):
                    value = noise[0 if tag == "initial" else int(tag.split("_")[1]) + 1]
                    dst = torch.zeros_like(template)
                    dst[0, :, :value.shape[-2]] = value
                    return dst
                result = sample(sampler, dm, batch["coords"], bf, init, noise_source=noise_source)
                xyz = result["X_L"][0, 0, :len(original_atoms)].float().cpu().numpy()
        assert np.isfinite(xyz).all()
        original_atoms.coord = xyz
        backbone = original_atoms[np.isin(original_atoms.atom_name, ["N", "CA", "C", "O"])]
        assert len(backbone) == length * 4
        backbone.res_name[:] = "ALA"
        backbone.set_annotation("occupancy", np.ones(len(backbone)))
        backbone.set_annotation("b_factor", np.zeros(len(backbone)))
        writer = PDBFile()
        writer.set_structure(backbone)
        writer.write(out / f"{case}.pdb")
        np.save(out / f"{case}_ca.npy", backbone.coord[backbone.atom_name == "CA"])
        distances = np.linalg.norm(np.diff(backbone.coord[backbone.atom_name == "CA"], axis=0), axis=1)
        record = {"case": case, "arm": args.arm, "length": length, "seed": seed,
            "seconds": time.monotonic() - started, "ca_neighbor_min": float(distances.min()),
            "ca_neighbor_max": float(distances.max()), "finite": True}
        (out / f"{case}.json").write_text(json.dumps(record, indent=2))
        print(json.dumps(record), flush=True)
        del result


def design_sequences(args, plan):
    from mpnn.inference_engines.mpnn import MPNNInferenceEngine
    engine = MPNNInferenceEngine(model_type="protein_mpnn", checkpoint_path=args.checkpoint,
        is_legacy_weights=True, out_directory=None, write_fasta=False, write_structures=False)
    outputs = []
    for arm in ([args.arm] if args.arm else plan["arms"]):
        for case, length, seed in cases(plan)[:args.limit]:
            structure = args.root / arm / f"{case}.pdb"
            results = engine.run(input_dicts=[{"structure_path": str(structure), "batch_size": 2,
                "seed": 5000 + seed, "temperature": 0.1}])
            assert len(results) == 2
            for index, result in enumerate(results):
                sequence = result.output_dict["designed_sequence"]
                assert len(sequence) == length and set(sequence) <= set("ACDEFGHIKLMNPQRSTVWY")
                outputs.append({"name": f"{arm}_{case}_d{index}", "arm": arm, "case": case,
                    "seed": seed, "sequence": sequence,
                    "components": [{"chain_id": "A", "seq": sequence, "chain_type": "polypeptide(l)"}]})
            print("MPNN", arm, case, flush=True)
    (args.root / "sequences.json").write_text(json.dumps(outputs, indent=2))


def fold(args, plan):
    from rf3.inference_engines.rf3 import RF3InferenceEngine
    from rf3.utils.inference import InferenceInput
    engine = RF3InferenceEngine(ckpt_path=args.checkpoint, diffusion_batch_size=1, n_recycles=10,
        num_steps=50, early_stopping_plddt_threshold=0, seed=42)
    engine.initialize()
    outputs = json.loads((args.root / "sequences.json").read_text())
    for item in outputs:
        if (args.root / "refolds" / item["name"] / (item["name"] + "_model.cif")).exists():
            continue
        engine.seed = 6000 + item["seed"] * 10 + int(item["name"].rsplit("_d", 1)[1])
        seed_all(engine.seed)
        inp = InferenceInput.from_json_dict({"name": item["name"], "components": item["components"]})
        engine.run(inputs=[inp], out_dir=args.root / "refolds")
        print("RF3", item["name"], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["generate", "mpnn", "fold"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--arm")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--pooling-source", type=Path)
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.root.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(4)
    if args.pooling_source:
        install_pooling(args.pooling_source)
    plan = json.loads((Path(__file__).parent / "plan.json").read_text())
    {"generate": generate, "mpnn": design_sequences, "fold": fold}[args.stage](args, plan)


if __name__ == "__main__":
    main()
