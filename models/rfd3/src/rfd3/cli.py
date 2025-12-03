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
    # Development: models/rfd3/src/rfd3/cli.py -> models/rfd3/configs/
    # Installed: site-packages/rfd3/cli.py -> site-packages/rfd3/configs/

    # Try development location first
    dev_config_path = Path(__file__).parent.parent.parent / "configs"
    if dev_config_path.exists():
        config_path = str(dev_config_path)
    else:
        # Fall back to installed package location
        config_path = str(Path(__file__).parent / "configs")

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
