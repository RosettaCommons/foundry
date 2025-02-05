import importlib
import sys
from typing import Any


def resolve_import(path: str) -> Any:
    """
    Import a module from a string path.
    If the module is not already imported, we dynamically import
    with `importlib.import_module` and return the module object.

    Args:
        path (str): The path to the module.

    Example usage with Hydra, assuming the module `rf2aa.setup` exists within the PYTHONPATH:
        ```yaml
        # config.yaml
        setup: ${resolve_import:rf2aa.setup}
        ```
    """
    namespace, name = path.rsplit(".", maxsplit=1)
    importlib.import_module(namespace)
    return sys.modules[namespace].__dict__[name]
