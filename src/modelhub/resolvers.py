"""Resolvers for Hydra configuration files."""

import importlib
from beartype.typing import Any


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
