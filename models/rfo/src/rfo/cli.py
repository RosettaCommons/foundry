#!/usr/bin/env python
"""RFOptimization command-line interface.

    rfo --config config.yaml

Runs the optimization: each cycle routes to RF3 backprop or Boltz with
probability ``backprop_fraction`` (default 0.5), then applies one MPNN-guided
amino-acid substitution and feeds the result into the next cycle. A YAML config sets
the inputs and output directory; any field can be overridden on the command
line. The driver launches separate processes for Boltz, RF3 and external
MPNN, optionally through Apptainer.
"""

import argparse
import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent  # src/rfo

# config key -> environment variable (resolves the asset store / containers)
ASSET_ENV = {
    "boltz_executable": "BOLTZ_EXECUTABLE",
    "boltz_sif": "BOLTZ_SIF",
    "boltz_ckpt": "BOLTZ_CKPT",
    "boltz_cache": "BOLTZ_CACHE",
    "boltz_bind_paths": "BOLTZ_BIND_PATHS",
    "rf3_python": "RF3_PYTHON",
    "rf3_sif": "RF3_SIF",
    "rf3_ckpt": "RF3_CKPT",
    "rf3_bind_paths": "RF3_BIND_PATHS",
    "mpnn_repo": "MPNN_REPO",
    "mpnn_python": "MPNN_PYTHON",
    "mpnn_sif": "MPNN_SIF",
    "mpnn_bind_paths": "MPNN_BIND_PATHS",
    "mpnn_ckpt": "MPNN_CKPT",
    "mpnn_model_type": "MPNN_MODEL_TYPE",
}


DEFAULTS = dict(
    input_structure=None,
    af3_json_template="",
    out_dir=None,
    backprop_fraction=0.5,
    total_cycles=10,
    # optional
    loss=None,  # backprop objective (loss_type); default from rf3_config
    msa_paths=None,  # {chain_id: path.a3m}; applied to both branches
    templates=None,  # {chain_id: structure}; fixes that chain in the RF3 branch
    rf3_config=None,
    num_samples=1,
    template_plddt_threshold=80,
    mpnn_num_seqs=1,
    mpnn_temperature=0.1,
    cyclic=False,
    seed=42,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rfo",
        argument_default=argparse.SUPPRESS,
        description="RFOptimization: RF3-backprop / Boltz cycling with MPNN redesign.",
    )
    p.add_argument("--config", help="YAML run config (see configs/example.yaml).")
    p.add_argument(
        "--input",
        dest="input_structure",
        help="Initial complex (PDB/CIF); binder = chain A, target = chain B.",
    )
    p.add_argument(
        "--af3-json",
        dest="af3_json_template",
        help="Optional AF3-style JSON for the Boltz branch; by default "
        "the complex is read from --input. Use this only to pin "
        "SMILES ligands or embedded MSAs.",
    )
    p.add_argument("--out-dir", dest="out_dir", help="Output directory.")
    p.add_argument(
        "--backprop-fraction",
        dest="backprop_fraction",
        type=float,
        help="Per-cycle probability of RF3 backprop vs Boltz (default 0.5).",
    )
    p.add_argument("--total-cycles", type=int, help="Number of cycles (default 10).")
    p.add_argument(
        "--loss",
        help="Backprop objective, e.g. pae_interface_mean, iptm "
        "(default: the rf3_config's loss_type).",
    )
    p.add_argument(
        "--rf3-config",
        dest="rf3_config",
        help="Backprop optimizer config (advanced; defaults to cycle_ppi).",
    )
    p.add_argument(
        "--num-samples", type=int, help="Boltz diffusion samples per prediction."
    )
    p.add_argument(
        "--mpnn-num-seqs",
        type=int,
        help="MPNN decoding orders to average for single-mutation scoring.",
    )
    p.add_argument(
        "--mpnn-temperature",
        type=float,
        help="Compatibility option; unused by maximum-likelihood mutation selection.",
    )
    p.add_argument("--template-plddt-threshold", type=int)
    p.add_argument(
        "--cyclic", action="store_true", help="Designed chain is a cyclic peptide."
    )
    p.add_argument("--seed", type=int)
    for key in ASSET_ENV:
        p.add_argument(
            "--" + key.replace("_", "-"),
            dest=key,
            help=f"Override {ASSET_ENV[key]} (also accepted under assets:).",
        )
    return p


def main() -> None:
    if len(sys.argv) == 1:
        build_parser().print_help()
        sys.exit(1)
    cli = vars(build_parser().parse_args())

    config = {}
    cfg_path = cli.pop("config", None)
    if cfg_path:
        with open(cfg_path) as f:
            config = yaml.safe_load(f) or {}

    if not isinstance(config, dict):
        sys.exit("rfo: config must be a YAML mapping.")
    unknown = set(config) - set(DEFAULTS) - {"assets"}
    if unknown:
        sys.exit(f"rfo: unknown config fields: {', '.join(sorted(unknown))}")
    assets = config.pop("assets", None) or {}
    if not isinstance(assets, dict) or set(assets) - set(ASSET_ENV):
        sys.exit(
            "rfo: assets must contain supported keys; see --help and docs/installation.md."
        )
    for key in ASSET_ENV:
        if key in cli:
            assets[key] = cli.pop(key)
    # Explicit config overrides environment; command-line flags override config.
    for key, value in assets.items():
        if value is not None:
            os.environ[ASSET_ENV[key]] = os.path.expanduser(str(value))

    # Merge: defaults < config < command-line flags.
    params = dict(DEFAULTS)
    params.update({k: v for k, v in config.items() if k in DEFAULTS})
    params.update(cli)

    if not params.get("rf3_config"):
        params["rf3_config"] = str(HERE / "backprop" / "configs" / "cycle_ppi.yaml")

    if not params["input_structure"] or not params["out_dir"]:
        sys.exit(
            "rfo: 'input_structure' and 'out_dir' are required "
            "(set them in the config or pass --input / --out-dir)."
        )

    if not 0 <= params["backprop_fraction"] <= 1:
        sys.exit("rfo: backprop_fraction must be between 0 and 1.")
    if any(params[k] < 1 for k in ("total_cycles", "num_samples", "mpnn_num_seqs")):
        sys.exit("rfo: total_cycles, num_samples and mpnn_num_seqs must be positive.")
    if params["mpnn_temperature"] <= 0:
        sys.exit("rfo: mpnn_temperature must be positive.")
    for key in ("input_structure", "out_dir", "rf3_config", "af3_json_template"):
        if params[key]:
            params[key] = str(Path(params[key]).expanduser().resolve())
    for key in ("msa_paths", "templates"):
        if params[key]:
            params[key] = {
                chain: str(Path(path).expanduser().resolve())
                for chain, path in params[key].items()
            }
    if not Path(params["input_structure"]).is_file():
        sys.exit(f"rfo: input structure does not exist: {params['input_structure']}")

    from rfo.cycling import cycle as cyc

    try:
        cyc.run_loop(argparse.Namespace(**params))
    except (OSError, ValueError, RuntimeError) as exc:
        sys.exit(f"rfo: {exc}")


if __name__ == "__main__":
    main()
