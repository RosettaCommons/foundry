"""Paired real-checkpoint quality audit; outputs are ordinary JSON/PT/CIF artifacts.

Run with either AW environment and that arm's source first on PYTHONPATH.
Both source trees must carry the identical deterministic pooling implementation.
"""

import argparse
import copy
import hashlib
import json
import os
import random
import sys
from pathlib import Path

os.environ["DISABLE_CUEQUIVARIANCE"] = "1"

import atomworks
import biotite
import numpy as np
import torch
from atomworks.io.utils.io_utils import to_cif_file

BASE = Path(__file__).resolve().parents[1]
CHECKPOINTS = BASE / ".validation/checkpoints"
SEEDS = (11, 29, 47)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all(),
    )


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu(v) for v in value)
    return copy.deepcopy(value)


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def atom_metrics(atoms, template=None):
    xyz = atoms.coord
    bonds = atoms.bonds.as_array()
    # Covalent geometry only: exclude metal coordination and unspecified types.
    bonds = bonds[np.isin(bonds[:, 2], [1, 2, 3, 5, 6, 7])]
    d = np.linalg.norm(xyz[bonds[:, 0]] - xyz[bonds[:, 1]], axis=-1)
    from scipy.spatial.distance import pdist, squareform

    dist = squareform(pdist(xyz))
    np.fill_diagonal(dist, np.inf)
    bonded = atoms.bonds.adjacency_matrix()
    # Exclude bonds and 1-3 neighbors from severe nonbonded clash count.
    excluded = bonded | ((bonded.astype(int) @ bonded.astype(int)) > 0)
    dist[excluded] = np.inf
    result = dict(
        n_atoms=len(atoms),
        finite=bool(np.isfinite(xyz).all()),
        n_chains=len(set(atoms.chain_id)),
        covalent_bonds=len(d),
        covalent_outlier_fraction=float(np.mean((d < 0.9) | (d > 2.0)))
        if len(d)
        else 0,
        covalent_min=float(d.min()) if len(d) else None,
        covalent_max=float(d.max()) if len(d) else None,
        severe_nonbonded_clashes=int(np.triu(dist < 0.8, 1).sum()),
    )
    if template is not None:
        from biotite.structure import rmsd, superimpose

        a = xyz[atoms.atom_name == "CA"]
        b = template.coord[template.atom_name == "CA"]
        if a.shape == b.shape:
            aligned, _ = superimpose(b, a)
            result["template_ca_rmsd"] = float(rmsd(b, aligned))
    return result


CASES = (
    "protein32",
    "protein_dna",
    "glycan",
    "cap",
    "ligands",
    "template32",
    "protein32_dna8",
)


def make_rf3_input(name):
    from rf3.utils.inference import InferenceInput

    design = json.loads(
        (BASE / "tests/data/atomworks_quality/designed32.json").read_text()
    )[0]
    design["name"] = "protein32"
    if name == "protein32":
        return InferenceInput.from_json_dict(design)
    if name == "protein32_dna8":
        design["name"] = name
        design["components"].append(
            dict(seq="ACGTACGT", chain_id="B", chain_type="polydeoxyribonucleotide")
        )
        return InferenceInput.from_json_dict(design)
    components = [dict(seq="GNGG", chain_id="A", chain_type="polypeptide(l)")]
    for case_name, component, bond in [
        (
            "protein_dna",
            dict(seq="ACGT", chain_id="B", chain_type="polydeoxyribonucleotide"),
            None,
        ),
        ("glycan", dict(ccd_code="NAG", chain_id="B"), ["A/ASN/2/ND2", "B/NAG/1/C1"]),
        ("cap", dict(ccd_code="ACE", chain_id="B"), ["A/GLY/1/N", "B/ACE/1/C"]),
    ]:
        if name != case_name:
            continue
        return InferenceInput.from_json_dict(
            dict(
                name=name,
                components=components + [component],
                bonds=[bond] if bond else [],
            )
        )
    lig = json.loads(
        (BASE / "models/rf3/tests/data/glke_with_ligands.json").read_text()
    )[0]
    for component in lig["components"]:
        if "path" in component:
            component["path"] = str(BASE / component["path"])
    lig["name"] = "ligands"
    if name == "ligands":
        return InferenceInput.from_json_dict(lig)
    path = BASE / "tests/data/atomworks_quality/template32.cif"
    return InferenceInput.from_cif_path(
        path, example_id="template32", template_selection="A"
    )


def run_rf3(args):
    from rf3.inference_engines.rf3 import RF3InferenceEngine
    from rf3.model.layers import af3_diffusion_transformer, pairformer_layers

    from foundry.utils.torch import scatter_mean

    assert af3_diffusion_transformer.scatter_mean is scatter_mean
    assert pairformer_layers.scatter_mean is scatter_mean
    engine = RF3InferenceEngine(
        ckpt_path=CHECKPOINTS / "rf3_foundry_01_24_latest_remapped.ckpt",
        n_recycles=10,
        num_steps=50,
        diffusion_batch_size=1,
        seed=11,
        early_stopping_plddt_threshold=0,
        triangle_attention_backend="vanilla",
        dense_attention_backend="vanilla",
    )
    engine.initialize()
    pipeline = engine.pipeline
    original_validation = engine.trainer.validation_step
    context = {}

    def capture_pipeline(data):
        path = args.out / context["name"] / f"seed{context['seed']}"
        path.mkdir(parents=True, exist_ok=True)
        if args.replay:
            stored = torch.load(
                args.replay / context["name"] / f"seed{context['seed']}" / "input.pt",
                weights_only=False,
                map_location="cpu",
            )
            restore_rng(stored["rng"])
            output = stored["input"]
        else:
            output = pipeline(data)
            stored = dict(input=cpu(output), rng=rng_state())
            torch.save(stored, path / "input.pt")
        return output

    def capture_validation(*a, **kw):
        path = args.out / context["name"] / f"seed{context['seed']}"
        torch.save(
            dict(input=cpu(kw["batch"]), rng=rng_state()), path / "forward_input.pt"
        )
        result = original_validation(*a, **kw)
        torch.save(cpu(result), path / "network.pt")
        return result

    engine.pipeline = capture_pipeline
    engine.trainer.validation_step = capture_validation
    for name in CASES:
        if args.cases and name not in args.cases:
            continue
        for seed in args.seeds or SEEDS:
            context.update(name=name, seed=seed)
            seed_all(seed)
            item = make_rf3_input(name)
            engine.seed = seed
            result = engine.run(item, out_dir=None, dump_predictions=False)
            sample = result[name][0]
            path = args.out / name / f"seed{seed}"
            atoms = sample.atom_array
            to_cif_file(atoms, path / "output.cif")
            torch.save(atoms, path / "atoms.pt")
            report = dict(
                summary=jsonable(sample.summary_confidences),
                geometry=atom_metrics(
                    atoms, item.atom_array if name == "template32" else None
                ),
            )
            (path / "metrics.json").write_text(json.dumps(report, indent=2))
            print("COMPLETE", args.arm, "rf3", name, seed, report, flush=True)


def run_mpnn(args):
    from mpnn.inference_engines.mpnn import MPNNInferenceEngine

    for model, ckpt in [
        ("protein_mpnn", "proteinmpnn_v_48_020.pt"),
        ("ligand_mpnn", "ligandmpnn_v_32_010_25.pt"),
    ]:
        engine = MPNNInferenceEngine(
            model_type=model,
            checkpoint_path=str(CHECKPOINTS / ckpt),
            is_legacy_weights=True,
            write_fasta=False,
            write_structures=False,
            device="cuda",
        )
        original = engine.model.forward
        context = {}

        def capture_forward(network_input):
            path = context["path"]
            path.mkdir(parents=True, exist_ok=True)
            stored = dict(input=cpu(network_input), rng=rng_state())
            torch.save(stored, path / "input.pt")
            result = original(network_input)
            torch.save(cpu(result), path / "network.pt")
            return result

        engine.model.forward = capture_forward
        for case in (
            "protein32",
            "protein_dna",
            "glycan",
            "cap",
            "ligands",
            "template32",
        ):
            if args.cases and case not in args.cases:
                continue
            # Both AW versions consume identical AW2 RF3 coordinates.
            structure = args.structures / case / "seed11/output.cif"
            for seed in args.seeds or SEEDS:
                context["path"] = args.out / model / case / f"seed{seed}"
                outputs = engine.run(
                    input_dicts=[
                        dict(
                            structure_path=str(structure),
                            seed=seed,
                            batch_size=2,
                            fixed_residues=["A4"],
                            temperature=0.1,
                            features_to_return=None,
                        )
                    ]
                )
                report = [jsonable(o.output_dict) for o in outputs]
                (context["path"] / "metrics.json").write_text(
                    json.dumps(report, indent=2)
                )
                print("COMPLETE", args.arm, model, case, seed, report, flush=True)


def main():
    global BASE, CHECKPOINTS
    p = argparse.ArgumentParser()
    p.add_argument("--fixture-root", type=Path, default=BASE)
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--model", choices=["rf3", "mpnn"], required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--replay", type=Path)
    p.add_argument("--structures", type=Path)
    p.add_argument("--cases", nargs="*")
    p.add_argument("--seeds", type=int, nargs="*")
    args = p.parse_args()
    BASE = args.fixture_root.resolve()
    CHECKPOINTS = args.checkpoint_dir.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import mpnn
    import rdkit
    import rf3

    from foundry.utils import torch as foundry_torch

    provenance = dict(
        arm=args.arm,
        argv=sys.argv,
        atomworks=atomworks.__version__,
        biotite=biotite.__version__,
        torch=torch.__version__,
        rdkit=rdkit.__version__,
        cuequivariance_enabled=__import__("foundry").SHOULD_USE_CUEQUIVARIANCE,
        pooling_sha256=hashlib.sha256(
            Path(foundry_torch.__file__).read_bytes()
        ).hexdigest(),
        pooling_path=foundry_torch.__file__,
        rf3_path=rf3.__file__,
        mpnn_path=mpnn.__file__,
        gpu=torch.cuda.get_device_name(),
        job=os.environ.get("SLURM_JOB_ID"),
    )
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(provenance, flush=True)
    (run_rf3 if args.model == "rf3" else run_mpnn)(args)


if __name__ == "__main__":
    main()
