"""Package boundaries and compatibility with checkpoints released before 1.0."""

import ast
import importlib
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import hydra
import pytest
import torch
from omegaconf import OmegaConf

MODELS = ("rf3", "rfd3", "rfd3na", "mpnn")
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("model", (*MODELS, "rfo"))
def test_model_imports_without_shared_package_or_unrelated_tools(model):
    # A fresh interpreter catches both missing transitive imports and imports that
    # an already-loaded sibling could otherwise conceal. RFO intentionally uses RF3.
    script = """
import importlib, importlib.abc, sys
model = sys.argv[1]
blocked = {'foundry', 'foundry_cli', 'rf3', 'rfd3', 'rfd3na', 'mpnn', 'rfo'} - {model}
if model == 'rfo':
    blocked.remove('rf3')
class BlockImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise AssertionError(f'{model} depends on {fullname}')
sys.meta_path.insert(0, BlockImports())
modules = {
 'rf3': ['inference_engines.rf3', 'trainers.rf3', '_core.callbacks.train_logging'],
 'rfd3': ['engine', 'trainer.rfd3', '_core.callbacks.train_logging'],
 'rfd3na': ['engine', 'trainer.rfd3na', '_core.callbacks.train_logging'],
 'mpnn': ['inference_engines.mpnn', 'trainers.mpnn', '_core.callbacks.metrics_logging'],
 'rfo': ['backprop.inference_pipeline.af3_inference_pipeline'],
}
for module in modules[model]:
    importlib.import_module(f'{model}.{module}')
assert not blocked.intersection(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, model],
        capture_output=True,
        text=True,
        timeout=90,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "FOUNDRY_DEVICE": "cpu"},
    )
    assert result.returncode == 0, result.stderr


def test_model_sources_and_configs_do_not_depend_on_shared_package():
    assert not (ROOT / "src/foundry").exists()
    for model in (*MODELS, "rfo"):
        blocked = {"foundry", "foundry_cli", *MODELS, "rfo"} - {model}
        if model == "rfo":
            blocked.remove("rf3")
        for path in (ROOT / "models" / model).rglob("*.py"):
            if "tests" in path.parts:
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    imports = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and not node.level:
                    imports = [node.module or ""]
                else:
                    continue
                assert all(name.split(".")[0] not in blocked for name in imports), path
        for path in (ROOT / "models" / model / "configs").rglob("*.yaml"):
            assert "foundry." not in path.read_text(), path


@pytest.mark.parametrize("model", MODELS)
def test_checkpoint_config_migration_preserves_interpolations_and_data(model):
    compat = importlib.import_module(f"{model}._core.compat")
    old = OmegaConf.create(
        {
            "steps": 7,
            "metric": {"_target_": "foundry.metrics.metric.MetricManager"},
            "nested": [{"_target_": "foundry.utils.weights.CheckpointConfig"}],
            "alias": "${steps}",
            "resolver": "${resolve_import:foundry.common,exists}",
            "input": "foundry.example.pdb",
            "note": "foundry.utils.weights.CheckpointConfig",
        }
    )
    updated = compat.migrate_config(old)
    assert old.metric._target_ == "foundry.metrics.metric.MetricManager"
    assert updated.metric._target_ == f"{model}._core.metrics.metric.MetricManager"
    assert updated.nested[0]._target_ == f"{model}._core.utils.weights.CheckpointConfig"
    assert OmegaConf.to_container(updated, resolve=False)["alias"] == "${steps}"
    updated.steps = 9
    assert updated.alias == 9
    assert updated.input == old.input
    assert updated.note == old.note
    assert OmegaConf.to_container(updated, resolve=False)["resolver"] == (
        f"${{resolve_import:{model}._core.common,exists}}"
    )
    manager = hydra.utils.instantiate(updated.metric)
    metric = importlib.import_module(f"{model}._core.metrics.metric")
    assert isinstance(manager, metric.MetricManager)
    assert compat.migrate_config(updated) == updated


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("legacy", [False, True], ids=["current", "pre-1.0"])
def test_checkpoint_load_remaps_serialized_classes_without_global_aliases(
    model, legacy, tmp_path, monkeypatch
):
    compat = importlib.import_module(f"{model}._core.compat")
    weights = importlib.import_module(f"{model}._core.utils.weights")
    config_type = weights.WeightLoadingConfig
    checkpoint = {
        "model": {"weight": torch.tensor([1.25, -2.0])},
        "optimizer": {"state": {"step": torch.tensor(3)}},
        "policy": config_type(),
        "train_cfg": OmegaConf.create(
            {"trainer": {"_target_": "foundry.trainers.fabric.FabricTrainer"}}
        ),
    }
    path = tmp_path / "model.ckpt"
    with monkeypatch.context() as patch:
        if legacy:
            # Simulate the actual old qualified name in the pickle stream, then
            # remove every fake module before loading, just as in a fresh install.
            for name in ("foundry", "foundry.utils", "foundry.utils.weights"):
                patch.setitem(sys.modules, name, ModuleType(name))
            patch.setattr(config_type, "__module__", "foundry.utils.weights")
            sys.modules["foundry.utils.weights"].WeightLoadingConfig = config_type
        torch.save(checkpoint, path)
    loaded = compat.load_checkpoint(path)
    assert isinstance(loaded["policy"], config_type)
    assert loaded["policy"].default_policy == checkpoint["policy"].default_policy
    assert loaded["train_cfg"].trainer._target_ == (
        f"{model}._core.trainers.fabric.FabricTrainer"
    )
    assert torch.equal(loaded["model"]["weight"], checkpoint["model"]["weight"])
    assert torch.equal(loaded["optimizer"]["state"]["step"], torch.tensor(3))
    assert not any(
        name == "foundry" or name.startswith("foundry.") for name in sys.modules
    )


def test_tools_register_resolvers_and_instantiate_callbacks_together(tmp_path):
    for model in MODELS:
        resolvers = importlib.import_module(f"{model}._core.hydra.resolvers")
        resolvers.register_resolvers()
        resolvers.register_resolvers()
        callbacks = importlib.import_module(f"{model}._core.callbacks.callback")
        instance = hydra.utils.instantiate(
            {
                "_target_": f"{model}._core.callbacks.metrics_logging.StoreValidationMetricsInDFCallback",
                "save_dir": str(tmp_path),
            }
            if model == "mpnn"
            else {
                "_target_": f"{model}._core.callbacks.train_logging.LogLearningRateCallback",
                "log_every_n": 1,
            }
        )
        assert isinstance(instance, callbacks.BaseCallback)
    assert OmegaConf.create({"x": "${resolve_import:os,path.join}"}).x is os.path.join


@pytest.mark.parametrize("model", MODELS)
def test_checkpoint_registries_are_owned_by_each_model(model, tmp_path, monkeypatch):
    registry = importlib.import_module(f"{model}.checkpoints")
    expected = {
        "rf3": {"rf3", "rf3_preprint_921", "rf3_preprint_124"},
        "rfd3": {"rfd3"},
        "rfd3na": {"rfd3na"},
        "mpnn": {"proteinmpnn", "ligandmpnn", "solublempnn"},
    }[model]
    assert set(registry.REGISTERED_CHECKPOINTS) == expected
    monkeypatch.setenv("FOUNDRY_CHECKPOINT_DIRS", str(tmp_path))
    for checkpoint in registry.REGISTERED_CHECKPOINTS.values():
        assert checkpoint.get_default_path() == tmp_path / checkpoint.filename
