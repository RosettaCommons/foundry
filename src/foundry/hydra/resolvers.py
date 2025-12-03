"""Resolvers for Hydra configuration files.

Documentation on custom resolvers:
- https://omegaconf.readthedocs.io/en/latest/custom_resolvers.html
"""

import importlib

from atomworks.enums import ChainType, ChainTypeInfo
from beartype.typing import Any
from omegaconf import OmegaConf

from ..common import run_once


#  (Custom resolvers)
@run_once
def register_resolvers():
    resolvers = {
        "resolve_import": resolve_import,
        "chain_type_info_to_regex": chain_type_info_to_regex,
    }

    for name, resolver in resolvers.items():
        OmegaConf.register_new_resolver(name, resolver)


def resolve_import(module_path: str, attribute_path: str = None) -> Any:
    """
    Import a module and access a specific attribute from it.

    Args:
        module_path (str): The path to the module.
        attribute_path (str): The path to the attribute within the module.

    Returns:
        The imported attribute.
    """
    module = importlib.import_module(module_path)
    if attribute_path is not None:
        # Split the attribute path to navigate through nested attributes
        attributes = attribute_path.split(".")
        attr = module
        for attr_name in attributes:
            attr = getattr(attr, attr_name)
        return attr
    else:
        return module


def chain_type_info_to_regex(*args) -> Any:
    """Convert a combination of ChainType or ChainTypeInfo attributes to a regex string.

    Primarily used for filtering a dataset by chain type prior to training/validation.

    Example filter:
    - "pn_unit_1_type.astype('str').str.match('${chain_type_info_to_regex:PROTEINS}')"

    """
    regex_str = ""

    for arg in args:
        if hasattr(ChainType, arg):
            regex_str += f"{getattr(ChainType, arg).value}|"
        elif hasattr(ChainTypeInfo, arg):
            chain_types_list = getattr(ChainTypeInfo, arg)
            for ct in chain_types_list:
                regex_str += f"{ct.value}|"
        else:
            raise ValueError(
                f"Attribute not found for ChainType or ChainTypeInfo: {arg}."
            )

    # Remove the trailing '|'
    regex_str = regex_str[:-1]

    return regex_str
