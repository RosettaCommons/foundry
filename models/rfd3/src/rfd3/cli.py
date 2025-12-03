from pathlib import Path

import typer
from hydra import compose, initialize_config_dir

app = typer.Typer()


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def design(ctx: typer.Context):
    """Run design using hydra config overrides and input files."""
    # Find the RFD3 configs directory relative to this file
    # This file is at: models/rfd3/src/rfd3/cli.py
    # Configs are at: models/rfd3/configs/
    rfd3_package_dir = Path(__file__).parent.parent.parent  # Go up to models/rfd3/
    config_path = str(rfd3_package_dir / "configs")

    # Get all arguments
    args = ctx.params.get("args", []) + ctx.args
    args = [a for a in args if a not in ["design", "fold"]]

    # Ensure we have at least a default inference_engine if not specified
    has_inference_engine = any(arg.startswith("inference_engine=") for arg in args)
    if not has_inference_engine:
        args.append("inference_engine=rfdiffusion3")

    with initialize_config_dir(config_dir=config_path, version_base="1.3"):
        cfg = compose(config_name="inference", overrides=args)

        # Lazy import to avoid loading heavy dependencies at CLI startup
        from foundry.utils.logging import suppress_warnings
        from rfd3.run_inference import run_inference

        with suppress_warnings(is_inference=True):
            run_inference(cfg)


if __name__ == "__main__":
    app()
