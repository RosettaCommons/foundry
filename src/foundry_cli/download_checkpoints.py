"""CLI for foundry model checkpoint installation and management."""

import hashlib
import os
from pathlib import Path
from typing import Optional
from urllib.request import urlopen

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

app = typer.Typer(help="Foundry model checkpoint installation utilities")
console = Console()

# Checkpoint URLs and metadata
# TODO: Replace these with your actual checkpoint URLs
CHECKPOINTS = {
    "rfd3": {
        "url": "https://files.ipd.uw.edu/pub/rfd3/rfd3_foundry_2025_12_01_remapped.ckpt",
        "filename": "rfd3_latest.ckpt",
        "sha256": None,  # Optional: add checksum for verification
        "description": "RFdiffusion3 checkpoint",
    },
    "rf3_preprint_921": {
        "url": "https://files.ipd.uw.edu/pub/rf3/rf3_foundry_09_21_preprint_remapped.ckpt",
        "filename": "rf3_foundry_09_21_preprint_remapped.ckpt",
        "sha256": None,
        "description": "RF3 preprint checkpoint trained with data until 9/2021",
    },
    "rf3_preprint_124": {
        "url": "https://files.ipd.uw.edu/pub/rf3/rf3_foundry_01_24_preprint_remapped.ckpt",
        "filename": "rf3_foundry_01_24_preprint_remapped.ckpt",
        "sha256": None,
        "description": "RF3 preprint checkpoint trained with data until 1/2024",
    },
    "rf3": {
        "url": "https://files.ipd.uw.edu/pub/rf3/rf3_foundry_01_24_latest_remapped.ckpt",
        "filename": "rf3_foundry_01_24_latest_remapped.ckpt",
        "sha256": None,
        "description": "latest RF3 checkpoint trained with data until 1/2024 (expect best performance)",
    },
    "proteinmpnn": {
        "url": "https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_020.pt",
        "filename": "proteinmpnn_v_48_020.pt",
        "sha256": None,
        "description": "ProteinMPNN checkpoint",
    },
    "ligandmpnn": {
        "url": "https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_010_25.pt",
        "filename": "ligandmpnn_v_32_010_25.pt",
        "sha256": None,
        "description": "LigandMPNN checkpoint",
    },
    "solublempnn": {
        "url": "https://files.ipd.uw.edu/pub/ligandmpnn/solublempnn_v_48_020.pt",
        "filename": "solublempnn_v_48_020.pt",
        "sha256": None,
        "description": "SolubleMPNN checkpoint",    
    }
}


def get_default_checkpoint_dir() -> Path:
    """Get the default checkpoint directory.

    Priority:
    1. FOUNDRY_CHECKPOINTS_DIR environment variable
    2. ~/.foundry/checkpoints
    """
    if "FOUNDRY_CHECKPOINTS_DIR" in os.environ:
        return Path(os.environ["FOUNDRY_CHECKPOINTS_DIR"])
    return Path.home() / ".foundry" / "checkpoints"


def download_file(url: str, dest: Path, verify_hash: Optional[str] = None) -> None:
    """Download a file with progress bar and optional hash verification.

    Args:
        url: URL to download from
        dest: Destination file path
        verify_hash: Optional SHA256 hash to verify against

    Raises:
        ValueError: If hash verification fails
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        # Get file size
        with urlopen(url) as response:
            file_size = int(response.headers.get("Content-Length", 0))

            task = progress.add_task(
                f"Downloading {dest.name}", total=file_size, start=True
            )

            # Download with progress
            hasher = hashlib.sha256() if verify_hash else None
            with open(dest, "wb") as f:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    if hasher:
                        hasher.update(chunk)
                    progress.update(task, advance=len(chunk))

    # Verify hash if provided
    if verify_hash:
        computed_hash = hasher.hexdigest()
        if computed_hash != verify_hash:
            dest.unlink()  # Remove corrupted file
            raise ValueError(
                f"Hash mismatch! Expected {verify_hash}, got {computed_hash}"
            )
        console.print("[green]✓[/green] Hash verification passed")


def install_model(
    model_name: str, checkpoint_dir: Path, force: bool = False
) -> None:
    """Install a single model checkpoint.

    Args:
        model_name: Name of the model (rfd3, rf3, mpnn)
        checkpoint_dir: Directory to save checkpoints
        force: Overwrite existing checkpoint if it exists
    """
    if model_name not in CHECKPOINTS:
        console.print(f"[red]Error:[/red] Unknown model '{model_name}'")
        console.print(f"Available models: {', '.join(CHECKPOINTS.keys())}")
        raise typer.Exit(1)

    checkpoint_info = CHECKPOINTS[model_name]
    dest_path = checkpoint_dir / checkpoint_info["filename"]

    # Check if already exists
    if dest_path.exists() and not force:
        console.print(
            f"[yellow]⚠[/yellow] {model_name} checkpoint already exists at {dest_path}"
        )
        console.print("Use --force to overwrite")
        return

    console.print(
        f"[cyan]Installing {model_name}:[/cyan] {checkpoint_info['description']}"
    )

    try:
        download_file(
            checkpoint_info["url"], dest_path, checkpoint_info.get("sha256")
        )
        console.print(
            f"[green]✓[/green] Successfully installed {model_name} to {dest_path}"
        )
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to install {model_name}: {e}")
        raise typer.Exit(1)


@app.command()
def install(
    models: list[str] = typer.Argument(
        ...,
        help="Models to install: 'all', 'rfd3', 'rf3', 'mpnn', or combination",
    ),
    checkpoint_dir: Optional[Path] = typer.Option(
        None,
        "--checkpoint-dir",
        "-d",
        help="Directory to save checkpoints (default: $FOUNDRY_CHECKPOINTS_DIR or ~/.foundry/checkpoints)",
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Overwrite existing checkpoints"
    ),
):
    """Install model checkpoints for foundry.

    Examples:

        foundry install all

        foundry install rfd3 rf3

        foundry install proteinmpnn --checkpoint-dir ./checkpoints
    """
    # Determine checkpoint directory
    if checkpoint_dir is None:
        checkpoint_dir = get_default_checkpoint_dir()

    console.print(f"[bold]Checkpoint directory:[/bold] {checkpoint_dir}")
    console.print()

    # Expand 'all' to all available models
    if "all" in models:
        models_to_install = list(CHECKPOINTS.keys())
    else:
        models_to_install = models

    # Install each model
    for model_name in models_to_install:
        install_model(model_name, checkpoint_dir, force)
        console.print()

    console.print("[bold green]Installation complete![/bold green]")


@app.command(name="list")
def list_models():
    """List available model checkpoints."""
    console.print("[bold]Available models:[/bold]\n")
    for name, info in CHECKPOINTS.items():
        console.print(f"  [cyan]{name:8}[/cyan] - {info['description']}")


@app.command()
def show(
    checkpoint_dir: Optional[Path] = typer.Option(
        None,
        "--checkpoint-dir",
        "-d",
        help="Checkpoint directory to show",
    ),
):
    """Show installed checkpoints."""
    if checkpoint_dir is None:
        checkpoint_dir = get_default_checkpoint_dir()

    if not checkpoint_dir.exists():
        console.print(
            f"[yellow]No checkpoints directory found at {checkpoint_dir}[/yellow]"
        )
        raise typer.Exit(0)

    checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
    if not checkpoint_files:
        console.print(
            f"[yellow]No checkpoint files found in {checkpoint_dir}[/yellow]"
        )
        raise typer.Exit(0)

    console.print(f"[bold]Installed checkpoints in {checkpoint_dir}:[/bold]\n")
    total_size = 0
    for ckpt in sorted(checkpoint_files):
        size = ckpt.stat().st_size / (1024**3)  # GB
        total_size += size
        console.print(f"  {ckpt.name:30} {size:8.2f} GB")

    console.print(f"\n[bold]Total:[/bold] {total_size:.2f} GB")


@app.command()
def clean(
    checkpoint_dir: Optional[Path] = typer.Option(
        None,
        "--checkpoint-dir",
        "-d",
        help="Checkpoint directory to clean",
    ),
    confirm: bool = typer.Option(
        True, "--confirm/--no-confirm", help="Ask for confirmation before deleting"
    ),
):
    """Remove all downloaded checkpoints."""
    if checkpoint_dir is None:
        checkpoint_dir = get_default_checkpoint_dir()

    if not checkpoint_dir.exists():
        console.print(f"[yellow]No checkpoints found at {checkpoint_dir}[/yellow]")
        raise typer.Exit(0)

    # List files to delete
    checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
    if not checkpoint_files:
        console.print(
            f"[yellow]No checkpoint files found in {checkpoint_dir}[/yellow]"
        )
        raise typer.Exit(0)

    console.print("[bold]Files to delete:[/bold]")
    total_size = 0
    for ckpt in checkpoint_files:
        size = ckpt.stat().st_size / (1024**3)  # GB
        total_size += size
        console.print(f"  {ckpt.name} ({size:.2f} GB)")

    console.print(f"\n[bold]Total:[/bold] {total_size:.2f} GB")

    # Confirm deletion
    if confirm:
        should_delete = typer.confirm("\nDelete these files?")
        if not should_delete:
            console.print("[yellow]Cancelled[/yellow]")
            raise typer.Exit(0)

    # Delete files
    for ckpt in checkpoint_files:
        ckpt.unlink()
        console.print(f"[red]✗[/red] Deleted {ckpt.name}")

    console.print("[green]✓[/green] Cleanup complete")


if __name__ == "__main__":
    app()
