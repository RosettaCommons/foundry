"""CLI for foundry model checkpoint installation and management."""

import hashlib
from pathlib import Path
from typing import Optional
from urllib.request import urlopen

import typer
from dotenv import load_dotenv
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

from foundry.inference_engines.checkpoint_registry import (
    REGISTERED_CHECKPOINTS,
    append_checkpoint_to_env,
    get_default_checkpoint_dirs,
)

load_dotenv(override=True)

app = typer.Typer(help="Foundry model checkpoint installation utilities")
console = Console()


def _resolve_checkpoint_dirs(checkpoint_dir: Optional[Path]) -> list[Path]:
    """Return checkpoint search path with defaults first."""
    checkpoint_dirs = get_default_checkpoint_dirs()
    if checkpoint_dir is not None:
        resolved = checkpoint_dir.expanduser().absolute()
        if resolved not in checkpoint_dirs:
            checkpoint_dirs.insert(0, resolved)
        else:
            # Move to front
            checkpoint_dirs.remove(resolved)
            checkpoint_dirs.insert(0, resolved)

        # Try to persist checkpoint dir to .env (optional, may not exist in Colab etc.)
        if append_checkpoint_to_env(checkpoint_dirs):
            console.print(
                f"Tracked checkpoint directories: {':'.join(str(path) for path in checkpoint_dirs)}"
            )

    return checkpoint_dirs


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


def install_model(model_name: str, checkpoint_dir: Path, force: bool = False) -> None:
    """Install a single model checkpoint.

    Args:
        model_name: Name of the model (rfd3, rf3, mpnn)
        checkpoint_dir: Directory to save checkpoints
        force: Overwrite existing checkpoint if it exists
    """
    if model_name not in REGISTERED_CHECKPOINTS:
        console.print(f"[red]Error:[/red] Unknown model '{model_name}'")
        console.print(f"Available models: {', '.join(REGISTERED_CHECKPOINTS.keys())}")
        raise typer.Exit(1)

    checkpoint_info = REGISTERED_CHECKPOINTS[model_name]
    dest_path = checkpoint_dir / checkpoint_info.filename

    # Check if already exists
    if dest_path.exists() and not force:
        console.print(
            f"[yellow]⚠[/yellow] {model_name} checkpoint already exists at {dest_path}"
        )
        console.print("Use --force to overwrite")
        return

    console.print(
        f"[cyan]Installing {model_name}:[/cyan] {checkpoint_info.description}"
    )

    try:
        download_file(checkpoint_info.url, dest_path, checkpoint_info.sha256)
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
        help="Models to install: 'all', 'rfd3', 'rf3', 'mpnn', or a combination thereof",
    ),
    checkpoint_dir: Optional[Path] = typer.Option(
        None,
        "--checkpoint-dir",
        "-d",
        help="Directory to save checkpoints (default search path: ~/.foundry/checkpoints plus any $FOUNDRY_CHECKPOINT_DIRS entries)",
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
    checkpoint_dirs = _resolve_checkpoint_dirs(checkpoint_dir)
    primary_checkpoint_dir = checkpoint_dirs[0]

    console.print(f"[bold]Install target:[/bold] {primary_checkpoint_dir}\n")

    # Expand 'all' to all available models
    if "all" in models:
        models_to_install = list(REGISTERED_CHECKPOINTS.keys())
    elif "base-models" in models:
        models_to_install = ["rfd3", "proteinmpnn", "ligandmpnn", "rf3"]
    else:
        models_to_install = models

    # Install each model
    for model_name in models_to_install:
        install_model(model_name, primary_checkpoint_dir, force)
        console.print()

    console.print("[bold green]Installation complete![/bold green]")


@app.command(name="list-available")
def list_available():
    """List available model checkpoints."""
    console.print("[bold]Available models:[/bold]\n")
    for name, info in REGISTERED_CHECKPOINTS.items():
        console.print(f"  [cyan]{name:8}[/cyan] - {info.description}")


@app.command(name="list-installed")
def list_installed():
    """List installed checkpoints and their sizes."""
    checkpoint_dirs = _resolve_checkpoint_dirs(None)

    checkpoint_files: list[tuple[Path, float]] = []
    for checkpoint_dir in checkpoint_dirs:
        if not checkpoint_dir.exists():
            continue
        ckpts = list(checkpoint_dir.glob("*.ckpt")) + list(checkpoint_dir.glob("*.pt"))
        for ckpt in ckpts:
            size = ckpt.stat().st_size / (1024**3)  # GB
            checkpoint_files.append((ckpt, size))

    if not checkpoint_files:
        console.print(
            "[yellow]No checkpoint files found in any checkpoint directory[/yellow]"
        )
        raise typer.Exit(0)

    console.print("[bold]Installed checkpoints:[/bold]\n")
    total_size = 0
    for ckpt, size in sorted(checkpoint_files, key=lambda item: str(item[0])):
        total_size += size
        console.print(f"  {ckpt} {size:8.2f} GB")

    console.print(f"\n[bold]Total:[/bold] {total_size:.2f} GB")


@app.command(name="clean")
def clean(
    confirm: bool = typer.Option(
        True, "--confirm/--no-confirm", help="Ask for confirmation before deleting"
    ),
):
    """Remove all downloaded checkpoints."""
    checkpoint_dirs = _resolve_checkpoint_dirs(None)

    # List files to delete
    checkpoint_files: list[Path] = []
    for checkpoint_dir in checkpoint_dirs:
        if not checkpoint_dir.exists():
            continue
        checkpoint_files.extend(checkpoint_dir.glob("*.ckpt"))
        checkpoint_files.extend(checkpoint_dir.glob("*.pt"))

    if not checkpoint_files:
        console.print(
            "[yellow]No checkpoint files found in any checkpoint directory[/yellow]"
        )
        raise typer.Exit(0)

    console.print("[bold]Files to delete:[/bold]")
    total_size = 0
    for ckpt in sorted(checkpoint_files, key=str):
        size = ckpt.stat().st_size / (1024**3)  # GB
        total_size += size
        console.print(f"  {ckpt} ({size:.2f} GB)")

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
