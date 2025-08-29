import os

import typer
from hydra import compose, initialize_config_dir

from modelhub.inference import run_inference

app = typer.Typer()


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def fold(ctx: typer.Context):
    """Run structure prediction using hydra config overrides or simple input file."""
    config_path = os.path.join(
        os.environ.get("PROJECT_PATH", os.environ["PROJECT_ROOT"]), "configs"
    )

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

    with initialize_config_dir(config_dir=config_path, version_base="1.3"):
        cfg = compose(config_name="inference", overrides=hydra_overrides)
        run_inference(cfg)


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def predict(ctx: typer.Context):
    """Alias for fold command."""
    fold(ctx)


if __name__ == "__main__":
    app()
