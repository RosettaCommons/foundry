import pandas as pd
from beartype.typing import Any
from lightning_fabric.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree
from torch import nn
from wandb.integration.lightning.fabric import WandbLogger

from modelhub.utils.ddp import RankedLogger

ranked_logger = RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def print_config_tree(
    cfg: DictConfig,
    resolve: bool = False,
    console_width: int = 100,
    title: str = "CONFIG",
) -> None:
    """Prints the contents of a DictConfig as a tree structure using the Rich library.

    Args:
        cfg (DictConfig): A DictConfig composed by Hydra.
        resolve (bool): Whether to resolve reference fields of DictConfig. Default is False.
        console_width (int): The width of the console for printing. Default is 100.
    """
    console = Console(width=console_width)
    style = "dim"
    tree = Tree(title, style=style, guide_style=style)

    # Generate config tree in natural order
    for field in cfg:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(Syntax(branch_content, "yaml", word_wrap=True))

    # Print config tree using Rich's Console
    # (This call happens before instantiating other loggers, so we don't try to capture the output)
    console.print(tree)


@rank_zero_only
def print_model_parameters(model: nn.Module, name: str = "") -> None:
    """Prints the total and trainable parameters of a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model to analyze.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    data = {
        "Type": ["Total Parameters", "Trainable Parameters"],
        "Count": [total_params, trainable_params],
    }

    title = f"Model Parameters: {name}" if name else "Model Parameters"
    print_df_as_table(pd.DataFrame(data), title=title)


def log_hyperparameters_with_all_loggers(
    trainer: Any, cfg: dict | DictConfig, model: Any
):
    """Logs hyperparameters using all loggers in the trainer.

    Args:
        trainer: The training object containing loggers.
        cfg: Configuration dictionary containing hyperparameters.
        model: The model to be tracked by loggers like WandbLogger.
    """
    # If given a DictConfig, convert it to a dictionary
    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)

    for logger in trainer.fabric.loggers:
        # ...log hyperparameters to each Fabric logger
        # For Abstract Base Class of Fabric `Loggers`, see: https://lightning.ai/docs/fabric/stable/_modules/lightning/fabric/loggers/logger.html#Logger
        assert hasattr(
            logger, "log_hyperparams"
        ), f"Logger {logger} does not have a `log_hyperparams` method. Ensure that the logger is a subclass of Fabric's ABC `Logger`."
        try:
            logger.log_hyperparams(cfg)
        except NotImplementedError:
            pass

        # ... if the logger is a WandbLogger, `watch` the model so that we can track gradients, utilization, etc.
        # (NOTE: W&B ensures only rank 0 watches the model internally)
        # See: https://docs.wandb.ai/ref/python/watch/
        if isinstance(logger, WandbLogger):
            logger.watch(model)


def condense_count_columns_of_grouped_df(df: pd.DataFrame) -> pd.DataFrame:
    """Returns modified DF with single Count column if valid, otherwise original DF.

    Helpful to avoid repeating count columns in a DataFrame with multi-level columns.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    try:
        # Validate count structure
        count_cols = df.xs("count", level=1, axis=1)
        mean_cols = df.xs("mean", level=1, axis=1)

        # Check count consistency per row and column existence
        if not (count_cols.nunique(axis=1) == 1).all():
            return df

        # Build condensed dataframe
        condensed_df = mean_cols.rename(columns=lambda c: f"{c} (mean)")
        condensed_df["Count"] = count_cols.iloc[:, 0].astype(int)
        return condensed_df

    except (KeyError, IndexError):
        return df


def table_from_df(df: pd.DataFrame, title: str) -> Table:
    """Create a Rich Table from a DataFrame."""
    table = Table(title=title, show_header=True, header_style="bold cyan")

    # Add columns to the table
    for col in df.columns:
        table.add_column(col, justify="right", style="magenta", overflow="fold")

    # Iterate through DataFrame rows and add them to the table
    for _, row in df.iterrows():
        row_cells = []

        for col in df.columns:
            cell_value = row[col]

            # Determine formatting based on data type
            if pd.api.types.is_integer_dtype(df[col]):
                formatted_value = f"{int(cell_value):,}"
            elif pd.api.types.is_float_dtype(df[col]):
                formatted_value = f"{float(cell_value):,.4f}"
            else:
                formatted_value = str(cell_value)

            row_cells.append(formatted_value)

        table.add_row(*row_cells)

    return table


def safe_print(obj: Any, console_width=100) -> None:
    """Print a Rich object in a console- and logger-safe manner."""
    console = Console(force_terminal=False, color_system=None, width=console_width)

    # Capture the table as a string and log it
    with console.capture() as capture:
        console.print(obj)
    ranked_logger.info(f"\n{capture.get()}")


def print_df_as_table(df: pd.DataFrame, title: str, console_width: int = 100) -> None:
    """Pretty-print a DataFrame using Rich Table"""
    safe_print(table_from_df(df=df, title=title), console_width=console_width)
