"""Unit tests for the pure Hydra resolver helpers in foundry.hydra.resolvers.

``register_resolvers`` is one-shot global registration with OmegaConf (side
effecting) and is not tested. The two resolver functions have a non-obvious
contract pinned here: ``resolve_import`` walks a dotted attribute path, and
``chain_type_info_to_regex`` builds an alternation regex from ChainType /
ChainTypeInfo enum members.
"""

import os

import pytest
from atomworks.enums import ChainType, ChainTypeInfo

from foundry.hydra.resolvers import chain_type_info_to_regex, resolve_import


def test_resolve_import_returns_the_module_when_no_attribute():
    assert resolve_import("os") is os


def test_resolve_import_walks_a_dotted_attribute_path():
    # os.path.join is reached by splitting "path.join" and chaining getattr.
    assert resolve_import("os", "path.join") is os.path.join


def test_resolve_import_resolves_a_single_attribute():
    assert resolve_import("os", "sep") == os.sep


def test_chain_type_info_to_regex_uses_chain_type_value():
    assert chain_type_info_to_regex("DNA") == str(ChainType.DNA.value)


def test_chain_type_info_to_regex_expands_a_chain_type_info_group():
    expected = "|".join(str(ct.value) for ct in ChainTypeInfo.PROTEINS)
    assert chain_type_info_to_regex("PROTEINS") == expected


def test_chain_type_info_to_regex_joins_multiple_args_with_pipe():
    result = chain_type_info_to_regex("DNA", "RNA")
    assert result == f"{ChainType.DNA.value}|{ChainType.RNA.value}"


def test_chain_type_info_to_regex_rejects_unknown_attribute():
    with pytest.raises(ValueError, match="Attribute not found"):
        chain_type_info_to_regex("NOT_A_CHAIN_TYPE")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
