"""Bridge Foundry's nested structural dataset configs to AtomWorks 3 loaders."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from atomworks.io.config import ParseConfig
from atomworks.ml.datasets import PandasDataset
from atomworks.ml.datasets.loaders.cif import create_structure_loader
from atomworks.ml.datasets.parsers import MetadataRowParser


class StructuralDatasetWrapper(PandasDataset):
    """Load structures while retaining existing metadata parsers and samplers.

    Historical Foundry configs referenced an AtomWorks wrapper that is absent
    from both released AtomWorks 2.2.1 and 3. Reuse the native dataset's metadata
    index and failure handling, and its supported structure loader. This adapter
    preserves parser-produced crop queries and validation metadata unchanged.
    """

    def __init__(
        self,
        dataset: PandasDataset,
        dataset_parser: MetadataRowParser,
        transform: Callable | None = None,
        cif_parser_args: dict | None = None,
        save_failed_examples_to_dir: str | Path | None = None,
    ):
        self.dataset = dataset
        self.dataset_parser = dataset_parser
        parser_args = dict(cif_parser_args or {})
        if parser_args.get("cache_dir"):
            parser_args.setdefault("load_from_cache", True)
            parser_args.setdefault("save_to_cache", True)
        config = ParseConfig.from_preset("rcsb", **parser_args)
        self._structure_loader = create_structure_loader(
            storage="filesystem", parser_args=config
        )
        super().__init__(
            data=dataset.data,
            name=dataset.name,
            metadata=dataset.metadata,
            loader=self._load_structure,
            transform=transform,
            save_failed_examples_to_dir=save_failed_examples_to_dir,
        )

    def _load_structure(self, row: pd.Series) -> dict[str, Any]:
        parsed = self.dataset_parser.parse(row)
        loaded = self._structure_loader(pd.Series(parsed))
        # The original metadata parser owns its crop queries and extra_info.
        return {**loaded, **parsed}
