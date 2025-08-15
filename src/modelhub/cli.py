import os
import typer
from hydra import compose, initialize_config_dir
from modelhub.inference import run_inference

app = typer.Typer()


@app.command()
def fold(inputs: str):
    """Run structure prediction using the given input file."""
    config_path = os.path.join(
        os.environ.get("PROJECT_PATH", os.environ["PROJECT_ROOT"]), "configs"
    )
    with initialize_config_dir(config_dir=config_path, version_base="1.3"):
        overrides = [f"inputs={inputs}"]
        cfg = compose(config_name="inference", overrides=overrides)
        run_inference(cfg)

@app.command()
def predict(inputs: str):
    """Alias for fold command."""
    fold(inputs=inputs)


if __name__ == "__main__":
    app()
