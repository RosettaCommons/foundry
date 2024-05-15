import torch
import pytest
import numpy as np
from rf2aa.data.loaders.polymer_partners import join_msas_by_taxid
from rf2aa.chemical import initialize_chemdata

initialize_chemdata()

msa_a = {
    "msa": torch.stack(
        [
            torch.tensor(
                [
                    0,
                    0,
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                ]
            ),
            torch.arange(12),
        ]
    ).permute(1, 0),
    "taxid": np.array(
        [
            "query",
            "1239",
            "1239",
            "1239",
            "0031",
            "0031",
            "94111",
            "3184",
            "3184",
            "3184",
            "3184",
            "2222",
        ]
    ),
}

msa_b = {
    "msa": torch.stack(
        [
            torch.tensor(
                [
                    1,
                    1,
                    2,
                    2,
                    1,
                    2,
                    2,
                    1,
                    1,
                    2,
                    1,
                ]
            ),
            torch.arange(11),
        ]
    ).permute(1, 0),
    "taxid": np.array(
        [
            "query",
            "1239",
            "1239",
            "1239",
            "0031",
            "0031",
            "94111",
            "94111",
            "3184",
            "3184",
            "4444",
        ]
    ),
}

paired_msa_out = torch.tensor(
    [
        [0, 0, 1, 0],
        [0, 6, 1, 7],
        [0, 7, 1, 8],
        [1, 10, 2, 9],
        [0, 1, 1, 1],
        [1, 3, 2, 3],
        [1, 2, 2, 2],
        [0, 5, 1, 4],
        [1, 4, 2, 5],
        [1, 8, 20, 20],
        [1, 9, 20, 20],
        [1, 11, 20, 20],
        [20, 20, 2, 6],
        [20, 20, 1, 10],
    ]
)


@pytest.mark.parametrize(
    "msa_a, msa_b, paired_msa_out", [(msa_a, msa_b, paired_msa_out)]
)
def test_msa_pairing(msa_a, msa_b, paired_msa_out):
    paired_msa_dict = join_msas_by_taxid(msa_a, msa_b)
    assert paired_msa_dict["msa"].shape[0] == 14
    assert paired_msa_dict["msa"].shape[1] == 4
    assert len(paired_msa_dict["taxid"]) == 14
    assert len(paired_msa_dict["is_paired"]) == 14

    assert torch.sum(paired_msa_dict["is_paired"]).item() == 9

    paired_taxids = paired_msa_dict["taxid"][paired_msa_dict["is_paired"]]
    unpaired_taxids = paired_msa_dict["taxid"][~paired_msa_dict["is_paired"]]

    paired_taxid_counts = dict(zip(*np.unique(paired_taxids, return_counts=True)))
    unpaired_taxid_counts = dict(zip(*np.unique(unpaired_taxids, return_counts=True)))

    assert paired_taxid_counts["0031"] == 2
    assert paired_taxid_counts["1239"] == 3
    assert paired_taxid_counts["3184"] == 2
    assert paired_taxid_counts["94111"] == 1
    assert paired_taxid_counts["query"] == 1
    assert len(paired_taxid_counts) == 5

    assert unpaired_taxid_counts["3184"] == 2
    assert unpaired_taxid_counts["4444"] == 1
    assert unpaired_taxid_counts["2222"] == 1
    assert unpaired_taxid_counts["94111"] == 1
    assert len(unpaired_taxid_counts) == 4

    paired_msa = paired_msa_dict["msa"]
    for i in range(paired_msa_out.shape[0]):
        desired_row = paired_msa_out[i][None]
        assert (
            (desired_row == paired_msa).all(dim=1).any().item()
        ), f"Missing row {desired_row} in paired_msa."
