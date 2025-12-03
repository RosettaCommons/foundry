from rfd3.inference.input_parsing import DesignInputSpecification
from rfd3.testing.testing_utils import PIPES, TEST_JSON_DATA


def test_legacy_pipeline_equivalence():
    from transforms.test_pipeline_regression import (
        _assert_features_equal,
        _assert_pipeline_results_equal,
        assert_same_atom_array,
    )

    args_new = TEST_JSON_DATA["brk-new"]
    args_legacy = TEST_JSON_DATA["brk-legacy"]

    spec_new = DesignInputSpecification.safe_init(**args_new)
    spec_legacy = DesignInputSpecification.safe_init(**args_legacy)

    spec_new_input = spec_new.to_pipeline_input("new")
    spec_legacy_input = spec_legacy.to_pipeline_input("legacy")

    aa_in_new = spec_new_input["atom_array"]
    aa_in_old = spec_legacy_input["atom_array"]

    # assert equivalent
    assert_same_atom_array(
        aa_in_new,
        aa_in_old,
        compare_coords=True,
        compare_bonds=True,
        # (All annotation categories present in the expected atom array are compared)
        annotations_to_compare=set(
            list(aa_in_old.get_annotation_categories())
            + list(aa_in_new.get_annotation_categories())
        ),
    )

    is_inference = True
    example_new = PIPES[is_inference](spec_new_input)
    example_legacy = PIPES[is_inference](spec_legacy_input)

    aa_new = example_new["atom_array"]
    aa_old = example_legacy["atom_array"]

    _assert_features_equal(
        example_new["feats"],
        example_legacy["feats"],
        "Brianne King's features for non-loopy",
        "inference",
    )

    assert_same_atom_array(
        aa_new,
        aa_old,
        compare_coords=True,
        compare_bonds=True,
        # (All annotation categories present in the expected atom array are compared)
        annotations_to_compare=set(
            list(aa_old.get_annotation_categories())
            + list(aa_new.get_annotation_categories())
        ),
    )
    _assert_pipeline_results_equal(
        example_new,
        example_legacy,
        "Brianne King's example for non-loopy",
        "inference",
    )


if __name__ == "__main__":
    test_legacy_pipeline_equivalence()
