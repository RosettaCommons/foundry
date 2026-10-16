"""CPU integration checks for cyclic input validation and feature transport."""

from pathlib import Path

import pytest
import torch
from atomworks.ml.utils.token import get_token_starts
from pydantic import ValidationError
from rfd3.inference.input_parsing import (
    DesignInputSpecification,
    create_atom_array_from_design_specification,
)
from rfd3.model.cfg_utils import strip_f
from rfd3.model.layers.blocks import RelativePositionEncodingWithIndexRemoval
from rfd3.transforms.pipelines import build_atom14_base_pipeline

RFD3_ROOT = Path(__file__).parents[1]
TARGET = RFD3_ROOT / "docs" / "input_pdbs" / "4zxb_cropped.pdb"


@pytest.fixture(scope="module")
def inference_pipeline():
    return build_atom14_base_pipeline(
        is_inference=True,
        sigma_data=16.0,
        diffusion_batch_size=1,
        generate_conformers=False,
        provide_reference_conformer_when_unmasked=False,
        ground_truth_conformer_policy="ground_truth",
        use_element_for_atom_names_of_atomized_tokens=False,
        n_atoms_per_token=14,
        central_atom="CB",
        atom_1d_features={},
        token_1d_features={},
    )


def _assert_cyclic_rpe(output: dict, cyclic_chain: str) -> None:
    """Check actual RPE wrapping and isolation from all context pairs."""
    features = output["feats"]
    tokens = output["atom_array"][get_token_starts(output["atom_array"])]
    selected_id = features["cyclic_asym_ids"][0]
    selected = torch.as_tensor(tokens.chain_id == cyclic_chain)

    assert features["cyclic_asym_ids"].dtype == torch.int64
    assert features["cyclic_asym_ids"].shape == (1,)
    assert torch.equal(features["asym_id"] == selected_id, selected)

    rpe = RelativePositionEncodingWithIndexRemoval(r_max=4, s_max=2, c_z=7)
    cyclic = rpe(features)
    linear = rpe({k: v for k, v in features.items() if k != "cyclic_asym_ids"})
    selected_indices = torch.where(selected)[0]
    first, last = selected_indices[0], selected_indices[-1]
    assert not torch.equal(cyclic[first, last], linear[first, last])
    outside_selected_chain = ~(selected[:, None] & selected[None, :])
    assert torch.equal(cyclic[outside_selected_chain], linear[outside_selected_chain])


@pytest.mark.parametrize(
    ("contig", "cyclic_chain"),
    [("3,/0,E6-10", "A"), ("E6-10,/0,3", "B")],
)
def test_binder_pipeline_accepts_chain_before_or_after_target(
    contig, cyclic_chain, inference_pipeline
):
    spec = DesignInputSpecification(
        input=str(TARGET), contig=contig, cyclic_chains=[cyclic_chain]
    )
    atom_array, metadata = spec.build(return_metadata=True)

    selected = atom_array[atom_array.chain_id == cyclic_chain]
    assert set(selected.src_component) == {"3P"}
    assert metadata["cyclic_chains"] == [cyclic_chain]
    output = inference_pipeline(spec.to_pipeline_input("binder"))
    _assert_cyclic_rpe(output, cyclic_chain)


@pytest.mark.parametrize(
    "value",
    ["A", [1], [""], ["A", "A"], ["A", "B"]],
)
def test_public_field_rejects_invalid_chain_lists(value):
    with pytest.raises(ValidationError, match="cyclic_chains"):
        DesignInputSpecification(length="3", cyclic_chains=value)


def test_unsupported_modes_are_rejected_at_public_dispatches():
    with pytest.raises(ValueError, match="dialect 2"):
        DesignInputSpecification.safe_init(dialect=1, length="3", cyclic_chains=["A"])
    with pytest.raises(ValueError, match="dialect 2"):
        create_atom_array_from_design_specification(
            dialect=1, length="3", cyclic_chains=["A"]
        )
    with pytest.raises(ValidationError, match="partial diffusion"):
        DesignInputSpecification(input=str(TARGET), partial_t=1.0, cyclic_chains=["A"])
    with pytest.raises(ValidationError, match="symmetry"):
        DesignInputSpecification(
            length="3",
            symmetry={"id": "C2", "is_symmetric_motif": False},
            cyclic_chains=["A"],
        )


def test_build_rejects_absent_source_derived_and_mixed_chains():
    with pytest.raises(ValueError, match="absent"):
        DesignInputSpecification(length="3", cyclic_chains=["B"]).build()

    source = DesignInputSpecification(
        input=str(TARGET),
        contig="E6-10",
        select_fixed_atoms=False,
        select_unfixed_sequence=True,
        cyclic_chains=["A"],
    )
    with pytest.raises(ValueError, match="complete de novo"):
        source.build()

    mixed = DesignInputSpecification(
        input=str(TARGET), contig="3,E6-10", cyclic_chains=["A"]
    )
    with pytest.raises(ValueError, match="complete de novo"):
        mixed.build()


@pytest.mark.parametrize("length", [10, 12])
def test_full_monomer_pipeline_transports_id_to_actual_rpe(length, inference_pipeline):
    spec = DesignInputSpecification(length=str(length), cyclic_chains=["A"])
    output = inference_pipeline(spec.to_pipeline_input("cyclic"))
    features = output["feats"]
    _assert_cyclic_rpe(output, "A")
    assert torch.equal(strip_f(features, [])["cyclic_asym_ids"], torch.tensor([0]))


def test_disabled_request_does_not_add_cyclic_features(inference_pipeline):
    output = inference_pipeline(
        DesignInputSpecification(length="3", cyclic_chains=[]).to_pipeline_input(
            "linear"
        )
    )
    assert "cyclic_asym_ids" not in output["feats"]
