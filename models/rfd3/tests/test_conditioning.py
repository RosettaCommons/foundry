import sys

import numpy as np
import pytest
from rfd3.testing.testing_utils import (
    TEST_JSON_DATA,
    build_pipelines,
    instantiate_example,
    load_train_or_val_cfg,
)

rasa_cfg = load_train_or_val_cfg("test-uncond")
rasa_cfg.datasets.global_transform_args.meta_conditioning_probabilities.calculate_rasa = 1.0
rasa_pipes = build_pipelines(composed_config=rasa_cfg)


@pytest.mark.fast
@pytest.mark.parametrize("example_name", ["1nzy-1"])
@pytest.mark.parametrize("is_inference", [True, False])
def test_rasa_conditioning(example_name, is_inference):
    args = TEST_JSON_DATA[example_name]
    input = instantiate_example(args, is_inference=is_inference)

    # Assert model input is present
    example = rasa_pipes[is_inference](input)
    flags = example["atom_array"].get_annotation_categories()

    print("Got flags:", flags)

    # In both training and inference cases, assert that ref_rasa_bin is present
    assert (
        "ref_atomwise_rasa" in example["feats"]
    ), "ref_rasa_bin not in feats, got: {}".format(list(example["feats"].keys()))

    if not is_inference:
        assert (
            input["atom_array"].bonds is not None
        ), "Bonds not present in input atom array"
        assert "rasa" in flags, "rasa not in atom_array, got: {}".format(flags)
        assert (
            example["feats"]["ref_atomwise_rasa"].sum() > 0
        ), "Did not create any rasa information"


plddt_cfg = load_train_or_val_cfg("test-uncond")
plddt_cfg.datasets.global_transform_args.meta_conditioning_probabilities.featurize_plddt = 1.0
plddt_cfg.datasets.train.pdb.sub_datasets.interface.dataset.transform.meta_conditioning_probabilities.featurize_plddt = 1.0
plddt_pipes = build_pipelines(
    composed_config=plddt_cfg,
    train_dataset_nested_keys=["monomer_distillation"],
    b_factor_min=70,
)


@pytest.mark.fast
@pytest.mark.parametrize("example_name", ["af2-122"])
@pytest.mark.parametrize("is_inference", [True, False])
def test_plddt_conditioning(example_name, is_inference):
    args = TEST_JSON_DATA[example_name]
    if is_inference:
        args["plddt_enhanced"] = True
    input = instantiate_example(args, is_inference=is_inference)

    # Assert model input is present
    example = plddt_pipes[is_inference](input)
    if not is_inference:
        assert example["conditions"]["featurize_plddt"], "Sampled wrong condition"
        plddt = example["atom_array"].get_annotation("b_factor")
        mean_plddt = np.mean(plddt)
        assert mean_plddt > 80

    # In training case, assert that ref_plddt is present
    assert "ref_plddt" in example["feats"], "ref_plddt not in feats, got: {}".format(
        list(example["feats"].keys())
    )
    # assert (
    #     torch.as_tensor(example["feats"]["ref_plddt"]).float().mean() == 1
    # ), "Expected ref_plddt to be 1, got: {}".format(example["feats"]["ref_plddt"])


if __name__ == "__main__":
    pytest.main(sys.argv)
