"""Unit tests for foundry.utils.instantiators.

These helpers turn a hydra config group into a list of instantiated objects.
The contract worth pinning is the control flow, not the object type: a missing /
empty config yields an empty list, each sub-config is instantiated via its
``_target_``, and a sub-config that is not an instantiable ``DictConfig`` (no
``_target_`` key) raises ``InstantiationError``. The functions do not themselves
check that the result is a callback / logger, so the tests use a lightweight
stdlib target (``types.SimpleNamespace``) to exercise that flow directly.
"""

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from foundry.utils.instantiators import (
    InstantiationError,
    _can_be_instantiated,
    instantiate_callbacks,
    instantiate_loggers,
)

_TARGET = "types.SimpleNamespace"


def test_can_be_instantiated_true_with_target():
    assert _can_be_instantiated(OmegaConf.create({"_target_": _TARGET})) is True


def test_can_be_instantiated_false_without_target():
    assert _can_be_instantiated(OmegaConf.create({"x": 1})) is False


def test_can_be_instantiated_false_for_non_dictconfig():
    """A plain dict is not a DictConfig, so it is not instantiable."""
    assert _can_be_instantiated({"_target_": _TARGET}) is False


def test_instantiate_callbacks_none_returns_empty():
    assert instantiate_callbacks(None) == []


def test_instantiate_callbacks_empty_config_returns_empty():
    assert instantiate_callbacks(OmegaConf.create({})) == []


def test_instantiate_callbacks_builds_each_target_in_order():
    cfg = OmegaConf.create(
        {
            "first": {"_target_": _TARGET, "x": 1},
            "second": {"_target_": _TARGET, "x": 2},
        }
    )
    result = instantiate_callbacks(cfg)
    assert result == [SimpleNamespace(x=1), SimpleNamespace(x=2)]


def test_instantiate_callbacks_raises_on_missing_target():
    cfg = OmegaConf.create({"bad": {"x": 1}})
    with pytest.raises(InstantiationError):
        instantiate_callbacks(cfg)


def test_instantiate_loggers_none_returns_empty():
    assert instantiate_loggers(None) == []


def test_instantiate_loggers_builds_target():
    cfg = OmegaConf.create({"logger": {"_target_": _TARGET, "name": "run"}})
    assert instantiate_loggers(cfg) == [SimpleNamespace(name="run")]


def test_instantiate_loggers_raises_on_missing_target():
    cfg = OmegaConf.create({"bad": {"name": "run"}})
    with pytest.raises(InstantiationError):
        instantiate_loggers(cfg)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
