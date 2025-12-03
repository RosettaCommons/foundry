from pathlib import Path

import typer
from hydra import compose, initialize_config_dir

app = typer.Typer(pretty_exceptions_enable=False)


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def fold(
    ctx: typer.Context,
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed logging output"
    ),
):
    """Run structure prediction using hydra config overrides or simple input file."""
    # Configure logging BEFORE any heavy imports
    if not verbose:
        from foundry.utils.logging import configure_minimal_inference_logging

        configure_minimal_inference_logging()

    # Find the RF3 configs directory relative to this file
    # This file is at: models/rf3/src/rf3/cli.py
    # Configs are at: models/rf3/configs/
    rf3_package_dir = Path(__file__).parent.parent.parent  # Go up to models/rf3/
    config_path = str(rf3_package_dir / "configs")

    # Get all arguments
    args = ctx.params.get("args", []) + ctx.args

    # Parse arguments
    hydra_overrides = []

    if len(args) == 1 and "=" not in args[0]:
        # Old style: single positional argument assumed to be inputs
        hydra_overrides.append(f"inputs={args[0]}")
    else:
        # New style: all arguments are hydra overrides
        hydra_overrides.extend(args)

    # Ensure we have at least a default inference_engine if not specified
    has_inference_engine = any(
        arg.startswith("inference_engine=") for arg in hydra_overrides
    )
    if not has_inference_engine:
        hydra_overrides.append("inference_engine=rf3")

    # Handle verbose flag
    if verbose:
        hydra_overrides.append("verbose=true")

    with initialize_config_dir(config_dir=config_path, version_base="1.3"):
        cfg = compose(config_name="inference", overrides=hydra_overrides)
        # Lazy import to avoid loading heavy dependencies at CLI startup
        from rf3.inference import run_inference

        run_inference(cfg)


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def predict(
    ctx: typer.Context,
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed logging output"
    ),
):
    """Alias for fold command."""
    fold(ctx, verbose=verbose)


if __name__ == "__main__":
    app()
