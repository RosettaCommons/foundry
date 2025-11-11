import sys

import pytest
from rfd3.testing.testing_utils import (
    TEST_JSON_DATA,
    build_pipelines,
    instantiate_example,
)

pipes = build_pipelines("test-uncond")


@pytest.mark.fast
@pytest.mark.parametrize("example", ["partial_diffusion"])
def test_partial_diffusion(example):
    is_inference = True
    args = TEST_JSON_DATA[example]
    input = instantiate_example(args, is_inference=is_inference)
    example = pipes[is_inference](input)

    aa = example["atom_array"]
    assert "partial_t" in aa.get_annotation_categories(), "partial_t not in atom_array"


if __name__ == "__main__":
    pytest.main(sys.argv)
