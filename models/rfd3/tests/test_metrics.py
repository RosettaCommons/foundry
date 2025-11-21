import copy
import sys

import pytest
from rfd3.metrics.design_metrics import get_all_backbone_metrics
from rfd3.testing.testing_utils import (
    TEST_JSON_DATA,
    build_pipelines,
    instantiate_example,
)
from rfd3.trainer.trainer_utils import (
    _cleanup_virtual_atoms_and_assign_atom_name_elements,
)

# uncond_cfg = load_train_or_val_cfg("test-uncond")
# pipes = build_pipelines(uncond_cfg)

pipes = build_pipelines("test-uncond")


@pytest.mark.fast
@pytest.mark.parametrize(
    "example",
    [
        "af2-122",
    ],
)
@pytest.mark.parametrize("is_inference", [False])
def test_output_metrics(example, is_inference):
    """
    Tests metrics after inference
    """
    # Instantiate test args:
    args = copy.deepcopy(TEST_JSON_DATA[example])
    input = instantiate_example(args, is_inference=is_inference)

    # Pipe input
    example = pipes[is_inference](input)
    atom_array = example["atom_array"]
    assert (
        atom_array.is_motif_atom_unindexed.sum() == 0
    ), "No unindexed atoms should be present"

    # Convert to cleaned array
    atom_array = _cleanup_virtual_atoms_and_assign_atom_name_elements(atom_array)
    metrics = get_all_backbone_metrics(atom_array)
    print(metrics)


if __name__ == "__main__":
    pytest.main(sys.argv)
