import copy
import sys

import numpy as np
import pytest
from rfd3.inference.input_parsing import DesignInputSpecification
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


def _build_pipeline_atom_array(args):
    """Run input parsing through `to_pipeline_input` and return the atom array.

    Mirrors what `instantiate_example` does for inference but returns the
    parsed atom array directly so we can assert on coordinate centering.
    """
    spec = DesignInputSpecification.safe_init(**args)
    pipeline_input = spec.to_pipeline_input(example_id="example")
    return pipeline_input["atom_array"]


@pytest.mark.fast
def test_partial_diffusion_respects_ori_token():
    """User-supplied `ori_token` must shift the structure during partial diffusion.

    Regression test: previously `_set_origin` hard-coded `ori_token=None` for
    partial diffusion, silently dropping the user's request.
    """
    base = copy.deepcopy(TEST_JSON_DATA["partial_diffusion"])
    base.pop("ori_token", None)
    base.pop("infer_ori_strategy", None)

    aa_default = _build_pipeline_atom_array(copy.deepcopy(base))

    shift = np.array([50.0, 0.0, 0.0], dtype=np.float32)
    args_shift = copy.deepcopy(base)
    args_shift["ori_token"] = shift.tolist()
    aa_shift = _build_pipeline_atom_array(args_shift)

    assert aa_default.array_length() == aa_shift.array_length()
    delta = aa_shift.coord.mean(axis=0) - aa_default.coord.mean(axis=0)
    # Coords are translated by -ori_token at parse time, so the post-parse
    # whole-structure mean must drop by ~50 Å on x.
    assert delta[0] == pytest.approx(
        -50.0, abs=1.5
    ), f"ori_token=[50,0,0] should shift coords by -50 in x; got delta={delta}"
    assert (
        abs(delta[1]) < 1.5 and abs(delta[2]) < 1.5
    ), f"ori_token=[50,0,0] should not move y/z; got delta={delta}"


@pytest.mark.fast
def test_partial_diffusion_defaults_to_diffused_region_com():
    """When neither `ori_token` nor `infer_ori_strategy` is supplied, partial
    diffusion must center on the diffused-region COM (matches training
    convention `center_option=diffuse`).

    Regression test: previously this branch centered on the joint
    target+diffused COM, biasing the model to drag the diffused region toward
    the motif's COM.
    """
    base = copy.deepcopy(TEST_JSON_DATA["partial_diffusion"])
    base.pop("ori_token", None)
    base.pop("infer_ori_strategy", None)
    aa = _build_pipeline_atom_array(base)

    is_motif = aa.is_motif_atom_with_fixed_coord.astype(bool)
    if not is_motif.any() or not (~is_motif).any():
        pytest.skip("Test fixture has no separable motif/diffused split")

    diffused_com = aa.coord[~is_motif].mean(axis=0)
    assert np.allclose(
        diffused_com, 0, atol=1e-3
    ), f"diffused-region COM should be at origin after centering; got {diffused_com}"


if __name__ == "__main__":
    pytest.main(sys.argv)
