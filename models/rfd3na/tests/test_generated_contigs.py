"""Generated polymers must not require a fictitious motif structure."""

import numpy as np
import pytest
from biotite.structure import filter_amino_acids, get_residue_starts
from pydantic import ValidationError
from rfd3na.inference.input_parsing import DesignInputSpecification
from rfd3na.inference.parsing import InputSelection

from foundry.utils.components import ComponentValidationError


@pytest.mark.parametrize(
    "contig,length,counts",
    [
        ("4-4R", "4", (0, 4, 0)),
        ("3D", "3", (0, 0, 3)),
        ("2,/0,3R,/0,4D", "9", (2, 3, 4)),
        (" 2R , /0 , 3D ", "5", (0, 2, 3)),
    ],
)
def test_generated_contig_builds_without_input(contig, length, counts):
    spec = DesignInputSpecification(contig=contig, length=length)
    assert spec.atom_array_input is None
    atoms = spec.build()
    residues = atoms[get_residue_starts(atoms)]
    assert (
        int(filter_amino_acids(residues).sum()),
        int(np.isin(residues.res_name, ["A", "C", "G", "U"]).sum()),
        int(np.isin(residues.res_name, ["DA", "DC", "DG", "DT"]).sum()),
    ) == counts
    assert not atoms.is_motif_atom_with_fixed_coord.any()
    assert np.isfinite(atoms.coord).all()


def test_input_backed_rna_motif_retains_atoms_coordinates_and_mask():
    source = DesignInputSpecification(contig="4R", length="4").build()
    source.coord = np.arange(len(source) * 3, dtype=np.float32).reshape(-1, 3) * 0.01
    result = DesignInputSpecification(
        input=None,
        atom_array_input=source,
        ori_token=[0, 0, 0],
        contig="A1-2,2R",
        length="4",
    ).build()
    assert len(get_residue_starts(result)) == 4
    src = source[np.isin(source.res_id, [1, 2])]
    dst = result[np.isin(result.res_id, [1, 2])]
    assert list(zip(src.res_name, src.atom_name)) == list(
        zip(dst.res_name, dst.atom_name)
    )
    np.testing.assert_array_equal(src.coord, dst.coord)
    assert dst.is_motif_atom_with_fixed_coord.all()


@pytest.mark.parametrize(
    "args",
    [
        {"contig": "A1-3,4R"},
        {"contig": "4R", "select_hotspots": "A1"},
        {"contig": "4R", "unindex": "A1"},
        {"contig": "4X"},
        {"contig": "4R", "partial_t": 0.5},
        {"contig": "/0"},
        {"contig": "4R,"},
    ],
)
def test_input_backed_and_malformed_requests_still_fail(args):
    with pytest.raises(ValidationError):
        DesignInputSpecification(**args)


def test_generated_length_still_enforced():
    with pytest.raises(ComponentValidationError):
        DesignInputSpecification(contig="4R", length="5").build()


def test_selected_tokens_cannot_be_dropped_without_input():
    selection = InputSelection(
        raw="A1", data={"A1": ["CA"]}, mask=np.ones(1, dtype=bool), tokens=None
    )
    with pytest.raises(ValueError, match="Input atoms"):
        selection.get_tokens(None)
