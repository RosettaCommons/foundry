"""CPU-only tests for the MPNN transform-pipeline builder's input validation.

These exercise the pure, data-free part of `build_mpnn_transform_pipeline` (the
`model_type` guard, which runs before any structure is loaded), so they run in the
generic gate — unlike the structure-loading tests in `test_pipeline.py`, which need a
DIGS PDB mirror and are held out of CI.
"""

import pytest
from atomworks.ml.transforms.base import Compose
from mpnn.pipelines.mpnn import build_mpnn_transform_pipeline


@pytest.mark.parametrize("model_type", ["protein_mpnn", "ligand_mpnn"])
def test_build_pipeline_valid_model_type_returns_compose(model_type: str):
    pipeline = build_mpnn_transform_pipeline(model_type=model_type)
    assert isinstance(pipeline, Compose)


@pytest.mark.parametrize("bad_model_type", ["bad", "", "ProteinMPNN", None])
def test_build_pipeline_invalid_model_type_raises(bad_model_type):
    with pytest.raises(ValueError, match="Unsupported model_type"):
        build_mpnn_transform_pipeline(model_type=bad_model_type)
