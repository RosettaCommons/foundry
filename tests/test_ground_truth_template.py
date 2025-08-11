from functools import partial

import numpy as np
import pytest
import torch
from cifutils.constants import (
    STANDARD_AA,
    STANDARD_DNA,
    STANDARD_RNA,
)
from cifutils.enums import ChainType
from datahub.transforms.atomize import AtomizeByCCDName
from datahub.transforms.base import Compose
from datahub.utils.rng import create_rng_state_from_seeds, rng_state
from datahub.utils.testing import cached_parse
from datahub.utils.token import get_af3_token_center_masks
from jaxtyping import Float
from torch import Tensor

from modelhub.data.ground_truth_template import (
    DEFAULT_DISTOGRAM_BINS,
    FeaturizeNoisedGroundTruthAsTemplateDistogram,
    TokenGroupNoiseScaleSampler,
    af3_noise_scale_distribution_wrapped,
    af3_noise_scale_to_noise_level,
)
from modelhub.utils.torch_utils import assert_no_nans, assert_same_shape

TEST_CASES = ["6wtf", "5ocm"]


def calc_distogram(coords: Float[Tensor, "n 3"]) -> Float[Tensor, "n n"]:
    return torch.cdist(coords, coords, p=2, compute_mode="donot_use_mm_for_euclid_dist")


def test_default_distogram_bins():
    assert len(DEFAULT_DISTOGRAM_BINS) == 63
    assert (
        torch.bucketize(torch.linspace(0, 22, 230), DEFAULT_DISTOGRAM_BINS).max() == 63
    )
    assert (
        torch.bucketize(torch.linspace(0, 22, 200), DEFAULT_DISTOGRAM_BINS).min() == 0
    )


@pytest.fixture
def setup_data_and_pipeline():
    def _setup(pdb_id):
        data = cached_parse(pdb_id)
        pipe = Compose(
            [
                AtomizeByCCDName(
                    atomize_by_default=True,
                    res_names_to_ignore=STANDARD_AA + STANDARD_RNA + STANDARD_DNA,
                    move_atomized_part_to_end=False,
                    validate_atomize=False,
                ),
            ],
            track_rng_state=False,
        )
        out = pipe(data)
        return out

    return _setup


@pytest.mark.parametrize("pdb_id", TEST_CASES)
@pytest.mark.parametrize(
    "transform_args",
    [
        {
            # Protein-only, high noise
            "noise_scale_distribution": lambda size: torch.ones(size) * 10.0,
            "allowed_chain_types": [ChainType.POLYPEPTIDE_L],
        },
        {
            # Protein and small molecules, no noise, no inter-molecule masking
            # (should be the same as the ground truth distogram)
            "noise_scale_distribution": lambda size: torch.zeros(size),
            "allowed_chain_types": [
                *ChainType.get_polymers(),
                *ChainType.get_non_polymers(),
            ],
            "p_provide_inter_molecule_distances": 1.0,
        },
        {
            # All chain types supported, but non-polymers have low noise, and polymers have high noise
            "noise_scale_distribution": TokenGroupNoiseScaleSampler(
                mask_and_sampling_fns=(
                    (
                        lambda arr: np.isin(arr.chain_type, ChainType.get_polymers()),
                        partial(
                            af3_noise_scale_distribution_wrapped,
                            upper_noise_level=af3_noise_scale_to_noise_level(
                                16.0
                            ).item(),
                        ),
                    ),
                    (
                        lambda arr: np.isin(
                            arr.chain_type, ChainType.get_non_polymers()
                        ),
                        partial(
                            af3_noise_scale_distribution_wrapped,
                            upper_noise_level=af3_noise_scale_to_noise_level(
                                2.0
                            ).item(),
                        ),
                    ),
                )
            ),
            "allowed_chain_types": [
                *ChainType.get_polymers(),
                *ChainType.get_non_polymers(),
            ],
        },
    ],
)
def test_distogram_featurization(
    pdb_id: str, transform_args: dict, setup_data_and_pipeline
):
    out = setup_data_and_pipeline(pdb_id)
    transform = FeaturizeNoisedGroundTruthAsTemplateDistogram(**transform_args)

    with rng_state(create_rng_state_from_seeds(12345)):
        atom_array = out["atom_array"]
        out["is_unconditional"] = False

        # Build ground-truth distogram
        token_center_mask = get_af3_token_center_masks(atom_array)
        token_center_atom_array = atom_array[token_center_mask]
        token_coord = torch.as_tensor(token_center_atom_array.coord)

        distogram = calc_distogram(token_coord)
        ground_truth_distogram_bins = torch.bucketize(distogram, DEFAULT_DISTOGRAM_BINS)

        # Featurize with the given arguments
        pipeline_output = transform(out)["feats"]
        has_distogram_condition = pipeline_output["has_distogram_condition"]

        output = torch.argmax(pipeline_output["distogram_condition"], dim=-1)

        assert has_distogram_condition.any(), "No distogram conditions found!"

        # Uncomment the code below to visualize the distogram and has_distogram_condition
        # _, axes = plt.subplots(1, 2, figsize=(12, 6))
        # cmap_output = plt.cm.get_cmap('RdYlGn_r')
        # axes[0].imshow(output, cmap=cmap_output, interpolation='none')
        # axes[1].imshow(has_distogram_condition, cmap='gray', interpolation='none')
        # plt.savefig('distogram_visualization.png', bbox_inches='tight')

        assert_same_shape(output, ground_truth_distogram_bins)
        assert_no_nans(output)

        # Mask of inter-molecule distances
        is_inter_molecule = (
            token_center_atom_array.molecule_id[:, None]
            != token_center_atom_array.molecule_id
        )

        noise_sum = (
            transform_args["noise_scale_distribution"](atom_array).sum()
            if isinstance(
                transform_args["noise_scale_distribution"], TokenGroupNoiseScaleSampler
            )
            else transform_args["noise_scale_distribution"](1).sum()
        )

        if noise_sum == 0:
            if transform_args.get("p_provide_inter_molecule_distances", 0.0) == 0.0:
                # ... except for inter-molecule distances
                assert (output[is_inter_molecule] == len(DEFAULT_DISTOGRAM_BINS)).all()
                assert not has_distogram_condition[is_inter_molecule].any()
            else:
                # (all distances)
                assert (
                    output[has_distogram_condition]
                    == ground_truth_distogram_bins[has_distogram_condition]
                ).all(), "Unnoised output should match distogram bins"
                assert (
                    output[~has_distogram_condition] == len(DEFAULT_DISTOGRAM_BINS)
                ).all(), "Values without condition should be max distance bin"

            # All values without distogram condition should be maximum distance bin
            assert (
                output[~has_distogram_condition] == len(DEFAULT_DISTOGRAM_BINS)
            ).all(), "All values without distogram condition should be the same"
        else:
            # Noised output should be different from the distogram bins
            assert not (
                output == ground_truth_distogram_bins
            ).all(), "Noised output should be different from the distogram bins"

            # Check that all values with distogram condition have been noised...
            assert (
                output[has_distogram_condition]
                != ground_truth_distogram_bins[has_distogram_condition]
            ).any(), "Values with distogram condition should be noised"
            # ... and that no values without conditions have been noised
            assert (
                output[~has_distogram_condition] == len(DEFAULT_DISTOGRAM_BINS)
            ).all(), "Values without distogram condition should be max distance bin"

        # Check chain type conditions
        if "supported_chain_types" in transform_args:
            tokens_with_supported_chain_types = np.isin(
                token_center_atom_array.chain_type,
                transform_args["supported_chain_types"],
            )
            tokens_with_supported_chain_types_l_l = (
                tokens_with_supported_chain_types[:, None]
                & tokens_with_supported_chain_types
            )
            assert (
                has_distogram_condition & tokens_with_supported_chain_types_l_l
            ).any(), "Supported chain types should be noised"
            assert not (
                has_distogram_condition & ~tokens_with_supported_chain_types_l_l
            ).any(), "Unsupported chain types should not be noised"

        # Inter-molecule distances should be the maximum distance bin and have no distogram condition
        if transform_args.get("p_provide_inter_molecule_distances", 0.0) == 0.0:
            assert (output[is_inter_molecule] == len(DEFAULT_DISTOGRAM_BINS)).all()
            assert not has_distogram_condition[is_inter_molecule].any()

        # Sanity check: we should have at least 5 unique values in the output
        unique_values = torch.unique(output)
        # Assert we have more than 30 unique values in the noised output
        assert (
            len(unique_values) > 30
        ), "We should have more than 30 unique values in the noised output"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
