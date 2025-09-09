import os

import typer
from hydra import compose, initialize_config_dir

app = typer.Typer(pretty_exceptions_show_locals=False, pretty_exceptions_short=True)

DOCS_URL = "https://github.com/RosettaCommons/modelforge/blob/production/src/modelhub/inference_engines/README.md"


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def fold(ctx: typer.Context):
    """Run structure prediction using hydra config overrides or simple input file."""
    # Get all arguments
    args = ctx.params.get("args", []) + ctx.args

    # Parse arguments
    hydra_overrides = []

    # Basic error handling & user conviencence
    if any(problematic_args := [arg for arg in args if arg.startswith("--")]):
        raise ValueError(
            "NOTE: The RF3 CLI does not support the --arg=value syntax "
            "because we use Hydra's override grammar. This means we use "
            "arg=value instead without the '--'. For more info please "
            f"refer to {DOCS_URL}. "
            f"Problematic arguments: {problematic_args}"
        )

    if any(arg.startswith("inputs=") for arg in args):
        if "=" not in args[0]:
            raise ValueError(
                "Cannot simultaneously specify inputs via positional argument and the `inputs=...` syntax. "
                "Please either use `rf3 fold path/to/inputs.json` or `rf3 fold inputs=path/to/inputs.json`. "
                "You may use .cif / .pdb / .json files as inputs. For more information please refer to: "
                f"{DOCS_URL}"
            )
    elif "=" not in args[0]:
        args[0] = f"inputs={args[0]}"
    else:
        raise ValueError(
            "Missing the `inputs=...` argument. Please use either `rf3 fold path/to/inputs.json` or "
            "`rf3 fold inputs=path/to/inputs.cif`."
        )

    hydra_overrides.extend(args)

    # Ensure we have at least a default inference_engine if not specified
    has_inference_engine = any(
        arg.startswith("inference_engine=") for arg in hydra_overrides
    )
    if not has_inference_engine:
        hydra_overrides.append("inference_engine=rf3")

    # ... lazy import to speed up CLI
    from modelhub.inference import run_inference

    config_path = os.path.join(
        os.environ.get("PROJECT_PATH", os.environ["PROJECT_ROOT"]), "configs"
    )

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
