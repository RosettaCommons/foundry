import copy
import sys

import numpy as np
import pytest
from rfd3.testing.testing_utils import (
    TEST_JSON_DATA,
    assert_tensor_or_array_equal,
    build_pipelines,
    instantiate_example,
    load_train_or_val_cfg,
)

uncond_cfg = load_train_or_val_cfg("test-uncond")
pipes = build_pipelines(composed_config=uncond_cfg)


@pytest.mark.fast
@pytest.mark.parametrize("example", ["af2-122", "1qys-1"])
@pytest.mark.parametrize("is_inference", [True, False])
def test_glycine_features_and_is_x(example, is_inference):
    # Instantiate test args:
    args = copy.deepcopy(TEST_JSON_DATA[example])
    input = instantiate_example(args, is_inference=is_inference)
    if not is_inference:
        assert (input["atom_array"].res_name == "GLY").sum() > 0

    # Pipe input once as glycines and once as alanines
    expected = pipes[is_inference](copy.deepcopy(input))["feats"]

    mask = input["atom_array"].res_name == "ALA"
    input["atom_array"].res_name[mask] = np.full(mask.sum(), "GLY")
    actual = pipes[is_inference](input)["feats"]

    bad_feats = []
    for feat in actual:
        # Assert array equality for input features
        e, g = expected[feat], actual[feat]
        try:
            assert_tensor_or_array_equal(
                g,
                e,
                "Features do not match for feature {}: {} != {}".format(feat, e, g),
            )
        except AssertionError as e:
            bad_feats.append(feat)

    assert bad_feats == [
        "is_central",
    ], "Expected only is_central to differ: {}".format(bad_feats)
    assert (
        actual["is_central"].sum() == actual["is_ca"].sum()
    ), "is_central and is_ca count should match"
    assert (
        expected["is_central"].sum() == expected["is_ca"].sum()
    ), "is_central and is_ca count should match in expected features"


if __name__ == "__main__":
    pytest.main(sys.argv)
