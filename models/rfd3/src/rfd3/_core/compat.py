"""Read pre-1.0 checkpoint metadata without restoring the removed shared package."""

import pickle
import re
from os import PathLike
from typing import Any, BinaryIO

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf


def _model_path(path: str) -> str:
    if path == "foundry" or path.startswith("foundry."):
        suffix = path.removeprefix("foundry")
        if suffix.startswith(".inference_engines.checkpoint_registry"):
            return __package__.removesuffix("._core") + suffix.replace(
                ".inference_engines.checkpoint_registry", ".checkpoints", 1
            )
        return __package__ + suffix
    return path


def migrate_config(config: Any) -> Any:
    """Translate Hydra targets/import resolvers, leaving data paths and text intact.

    Keep interpolations unresolved so overrides are still applied before resolution.
    The input is not modified; model weights and optimizer state are never rewritten.
    """
    is_config = isinstance(config, (DictConfig, ListConfig))
    data = OmegaConf.to_container(config, resolve=False) if is_config else config

    def visit(value: Any, key: Any = None) -> Any:
        if isinstance(value, dict):
            return {k: visit(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [visit(v) for v in value]
        if isinstance(value, tuple):
            return tuple(visit(v) for v in value)
        if isinstance(value, str):
            if key == "_target_":
                return _model_path(value)
            return re.sub(
                r"(\$\{resolve_import:\s*)(foundry(?:\.[A-Za-z_]\w*)*)",
                lambda match: match[1] + _model_path(match[2]),
                value,
            )
        return value

    migrated = visit(data)
    return OmegaConf.create(migrated) if is_config else migrated


class _CheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        return super().find_class(_model_path(module), name)


class _CheckpointPickle:
    Unpickler = _CheckpointUnpickler

    @staticmethod
    def load(file: BinaryIO, **kwargs: Any) -> Any:
        return _CheckpointUnpickler(file, **kwargs).load()


def load_checkpoint(path: str | PathLike, **kwargs: Any) -> Any:
    """Load trusted model checkpoints, including old serialized config classes.

    This uses the same unrestricted pickle semantics as the pre-1.0 loader. The
    module translation is local to this unpickler; no global import aliases are set.
    """
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        pickle_module=_CheckpointPickle,
        **kwargs,
    )
    if isinstance(checkpoint, dict) and "train_cfg" in checkpoint:
        checkpoint["train_cfg"] = migrate_config(checkpoint["train_cfg"])
    return checkpoint
