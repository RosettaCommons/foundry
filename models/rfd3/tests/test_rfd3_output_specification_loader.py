"""Regression tests for loading design specifications from RFD3 output metadata."""

import json

import pytest
from rfd3.inference.input_parsing import DesignInputSpecification


@pytest.mark.parametrize("metadata_key", ["specification", "input_specification"])
def test_from_rfd3_out_accepts_current_and_legacy_metadata_keys(
    tmp_path, metadata_key
):
    output = tmp_path / "design.json"
    output.write_text(json.dumps({metadata_key: {"length": "3"}}))

    restored = DesignInputSpecification.from_rfd3_out(str(output))

    assert restored.length == "3"
