"""Generate one small structure through Pipelines using only the baked image code."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from blocks import Blocks
    from pipelines import ExecutionEngine, init_cli

    init_cli(["foundry-fast-smoke", "--local", "--use_local_gpus"])
    block = Blocks.get("rfd3")(
        cloud=False,
        command_template=[
            "apptainer", "exec", "--nv", "--cleanenv", "--pwd", "/opt/foundry",
            "--bind", "/mnt/home:/mnt/home", "--bind", "/projects:/projects",
            "--env", "OMP_NUM_THREADS=4,MKL_NUM_THREADS=4,TORCHINDUCTOR_COMPILE_THREADS=4",
            str(args.image.resolve()), "rfd3", "design",
        ],
        inputs={"smoke": {"length": 32}},
        batch_size=1,
        n_batches=1,
        inference_sampler={"num_timesteps": 7, "use_classifier_free_guidance": False},
    )
    engine = ExecutionEngine(rundir=str(args.output.resolve()))
    handler = engine.run([block])
    assert len(handler) == 1, f"Expected one sampled structure, got {len(handler)}"
    table = handler.unpack()
    print(table.to_string())
    print(engine)
    handler.write_parquet(str(args.output.resolve() / "smoke.parquet"))
    (args.output / "validation.json").write_text(json.dumps({
        "image": str(args.image.resolve()), "structures": len(handler),
        "length": 32, "schedule_points": 7,
        "compile_model": True, "low_memory_mode": False,
        "purpose": "Image/CLI/Pipelines integration smoke test, not biological validation",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
