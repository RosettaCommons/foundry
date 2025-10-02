import numpy as np
import pytest
import torch
from lightning.fabric import seed_everything
from omegaconf import DictConfig
from rf3.chemical import NHEAVY, heavyatom_mask
from rf3.metrics.metric_utils import (
    find_bin_midpoints,
    unbin_logits,
)
from rf3.utils.predicted_error import compile_af3_confidence_outputs


def test_compile_af3_confidence_outputs():
    L = 100

    # Spoofing the outputs from the model
    seed_everything(42)
    outputs = {
        "confidence": {
            "rf2aa_seq": torch.randint(0, 21, (L,)),
            "plddt_logits": torch.rand(2, L, NHEAVY, 50),
            "pae_logits": torch.rand(2, L, L, 64),
            "pde_logits": torch.rand(2, L, L, 64),
            "chain_iid_token_lvl": torch.randint(0, 10, (L,)).numpy(),
        }
    }
    is_real_atom = heavyatom_mask[outputs["confidence"]["rf2aa_seq"]]
    outputs["confidence"]["is_real_atom"] = is_real_atom

    # Spoof the confidence loss Hydra configuration
    cfg = DictConfig(
        {
            "plddt": {
                "weight": 1.0,
                "n_bins": 50,
                "max_value": 1.0,
            },
            "pae": {
                "weight": 1.0,
                "n_bins": 64,
                "max_value": 32,
            },
            "pde": {
                "weight": 1.0,
                "n_bins": 64,
                "max_value": 32,
            },
        }
    )

    output = compile_af3_confidence_outputs(
        plddt_logits=outputs["confidence"]["plddt_logits"],
        pae_logits=outputs["confidence"]["pae_logits"],
        pde_logits=outputs["confidence"]["pde_logits"],
        chain_iid_token_lvl=outputs["confidence"]["chain_iid_token_lvl"],
        is_real_atom=is_real_atom,
        example_id="test",
        confidence_loss_cfg=cfg,
    )

    num_chains = len(np.unique(outputs["confidence"]["chain_iid_token_lvl"]))
    num_interfaces = num_chains * (num_chains - 1) // 2
    num_batches = outputs["confidence"]["plddt_logits"].shape[0]

    df = output["confidence_df"]

    target_columns = [
        "example_id",
        "chain_chainwise",
        "chainwise_plddt",
        "chainwise_pde",
        "chainwise_pae",
        "overall_plddt",
        "overall_pde",
        "overall_pae",
        "batch_idx",
        "chain_i_interface",
        "chain_j_interface",
        "pae_interface",
        "pde_interface",
        "min_pae_interface",
        "min_pde_interface",
    ]

    print(df.columns.tolist())
    print(target_columns)
    assert df.columns.tolist() == target_columns, "Dataframe columns not set correctly"
    assert df.shape == (
        num_batches * (num_interfaces + num_chains),
        len(target_columns),
    ), "Dataframe shape not set correctly"


def test_unbin_pae_logits():
    L = 100
    max_distance = 32
    n_bins = 64

    seed_everything(42)
    outputs = {
        "confidence": {
            "rf2aa_seq": torch.randint(0, 21, (L,)),
            "plddt_logits": torch.rand(1, L, NHEAVY, 50),
            "pae_logits": torch.rand(1, L, L, 64),
            "pde_logits": torch.rand(1, L, L, 64),
            "chain_iid_token_lvl": torch.randint(0, 10, (L,)).numpy(),
        }
    }
    is_real_atom = heavyatom_mask[outputs["confidence"]["rf2aa_seq"]]
    outputs["confidence"]["is_real_atom"] = is_real_atom

    pae_unbinned = unbin_logits(
        outputs["confidence"]["pae_logits"].permute(0, 3, 1, 2).float(),
        max_distance=max_distance,
        num_bins=n_bins,
    )

    assert torch.allclose(torch.mean(pae_unbinned), torch.tensor(15.99), atol=1e-2)
    assert pae_unbinned.shape == (1, L, L)


def test_unbin_pde_logits():
    L = 100
    max_distance = 32
    n_bins = 64

    seed_everything(42)
    outputs = {
        "confidence": {
            "rf2aa_seq": torch.randint(0, 21, (L,)),
            "plddt_logits": torch.rand(1, L, NHEAVY, 50),
            "pae_logits": torch.rand(1, L, L, 64),
            "pde_logits": torch.rand(1, L, L, 64),
            "chain_iid_token_lvl": torch.randint(0, 10, (L,)).numpy(),
        }
    }
    is_real_atom = heavyatom_mask[outputs["confidence"]["rf2aa_seq"]]
    outputs["confidence"]["is_real_atom"] = is_real_atom

    pde_unbinned = unbin_logits(
        outputs["confidence"]["pae_logits"].permute(0, 3, 1, 2).float(),
        max_distance=max_distance,
        num_bins=n_bins,
    )

    assert torch.allclose(torch.mean(pde_unbinned), torch.tensor(16.00), atol=1e-2)

    assert pde_unbinned.shape == (1, L, L)


def test_unbin_plddt_logits():
    L = 100
    max_distance = 1.0
    n_bins = 50

    seed_everything(42)
    outputs = {
        "confidence": {
            "rf2aa_seq": torch.randint(0, 21, (L,)),
            "plddt_logits": torch.rand(1, L, NHEAVY, 50),
            "pae_logits": torch.rand(1, L, L, 64),
            "pde_logits": torch.rand(1, L, L, 64),
            "chain_iid_token_lvl": torch.randint(0, 10, (L,)).numpy(),
        }
    }
    is_real_atom = heavyatom_mask[outputs["confidence"]["rf2aa_seq"]]
    outputs["confidence"]["is_real_atom"] = is_real_atom

    plddt_unbinned = unbin_logits(
        outputs["confidence"]["plddt_logits"].permute(0, 3, 1, 2).float(),
        max_distance,
        n_bins,
    )

    assert plddt_unbinned.shape == (1, L, NHEAVY)


def test_bin_midpoints():
    max_distance = 32
    num_bins = 64
    expected_bins = torch.linspace(0.25, 31.75, 64, device="cpu")
    pae_bins = find_bin_midpoints(max_distance, num_bins)
    assert torch.allclose(pae_bins, expected_bins)


if __name__ == "__main__":
    pytest.main([__file__])
