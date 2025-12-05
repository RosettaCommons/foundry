import sys

import numpy as np
import pytest
from rfd3.inference.symmetry.checks import check_atom_array_is_symmetric
from rfd3.inference.symmetry.contigs import (
    expand_contig_unsym_motif,
    get_unsym_motif_mask,
)
from rfd3.testing.testing_utils import (
    TEST_JSON_DATA,
    build_pipelines,
    instantiate_example,
)
from rfd3.transforms.conditioning_base import get_motif_features

pipes = build_pipelines("test-uncond")
np.set_printoptions(threshold=np.inf)


@pytest.mark.fast
@pytest.mark.parametrize("example", ["1j79_C2", "1e3v_C2", "1bfr_C2", "6t8h_C3"])
@pytest.mark.parametrize("is_inference", [True])
def test_symmetrized_motif(example, is_inference):
    # check wheter there are symm annotations after pipeline
    # check wether the symmetrized motif matches the expected sym id (testing general sym motif)
    args = TEST_JSON_DATA[example]
    input = instantiate_example(args, is_inference=is_inference)
    example = pipes[is_inference](input)
    aa = example["atom_array"]
    assert (
        "sym_entity_id" in aa.get_annotation_categories()
    ), "sym_entity_id not in atom_array"

    sym_motif_mask = get_motif_features(aa)["is_motif_atom"]
    if args["symmetry"].get("is_unsym_motif"):
        unsym_motif_names = args["symmetry"]["is_unsym_motif"].split(",")
        unsym_motif_names = expand_contig_unsym_motif(unsym_motif_names)
        is_unsym_motif = get_unsym_motif_mask(aa, unsym_motif_names)
        sym_motif_mask = sym_motif_mask & ~is_unsym_motif
    symmetrized_motifs = aa[sym_motif_mask]

    assert check_atom_array_is_symmetric(
        symmetrized_motifs
    ), "Symmetrized motif is not symmetric"


if __name__ == "__main__":
    pytest.main(sys.argv)
