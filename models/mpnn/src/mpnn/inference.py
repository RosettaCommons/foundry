#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../../../../.ipd/shebang/mpnn_exec.sh" "$0" "$@"'

from mpnn.inference_engines.mpnn import MPNNInferenceEngine
from mpnn.utils.inference import (
    MPNN_GLOBAL_INFERENCE_DEFAULTS,
    build_arg_parser,
    cli_to_json,
)


def main() -> None:
    """Top-level CLI entry point for MPNN inference.

    This script wires together:
      - CLI / arg parsing
      - JSON config building (or loading an existing JSON)
      - Execution of the MPNNInferenceEngine
    """
    # CLI
    parser = build_arg_parser()
    args = parser.parse_args()

    # JSON config building/loading
    config = cli_to_json(args)

    # Split global vs per-input config
    engine = MPNNInferenceEngine(
        model_type=config.get(
            "model_type", MPNN_GLOBAL_INFERENCE_DEFAULTS["model_type"]
        ),
        checkpoint_path=config.get(
            "checkpoint_path", MPNN_GLOBAL_INFERENCE_DEFAULTS["checkpoint_path"]
        ),
        is_legacy_weights=config.get(
            "is_legacy_weights", MPNN_GLOBAL_INFERENCE_DEFAULTS["is_legacy_weights"]
        ),
        out_directory=config.get(
            "out_directory", MPNN_GLOBAL_INFERENCE_DEFAULTS["out_directory"]
        ),
        write_fasta=config.get(
            "write_fasta", MPNN_GLOBAL_INFERENCE_DEFAULTS["write_fasta"]
        ),
        write_structures=config.get(
            "write_structures", MPNN_GLOBAL_INFERENCE_DEFAULTS["write_structures"]
        ),
    )

    # In this case, structures are resolved from inputs; no explicit atom arrays
    _ = engine.run(input_dicts=config["inputs"], atom_arrays=None)


if __name__ == "__main__":
    main()
