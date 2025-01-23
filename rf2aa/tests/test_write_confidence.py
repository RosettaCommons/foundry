import numpy as np
import torch

from functools import partial
from rf2aa.chemical import initialize_chemdata, ChemicalData as ChemData
from rf2aa.metrics.metric_utils import  \
        write_confidence_metrics, \
        unbin_rf3_metrics, \
        unbin_logits, \
        find_bin_midpoints 
from rf2aa.metrics.predicted_error import WriteAF3Confidence
from rf2aa.set_seed import seed_all
from tempfile import NamedTemporaryFile
import pandas as pd
from itertools import combinations


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = AttrDict(value)

    def __getattr__(self, item):
        if item in self:
            return self[item]
        raise AttributeError(f"'AttrDict' object has no attribute '{item}'")

    def __setattr__(self, key, value):
        self[key] = value

def test_write_confidence():
    L = 100
    init = partial(initialize_chemdata)
    init()

    bins = 10
    seed_all(42)
    outputs = {
        "confidence": {
            "rf2aa_seq": torch.randint(0, 21, (L,)),
            "plddt_logits": torch.rand(2, L, ChemData().NHEAVY, 50),
            "pae_logits": torch.rand(2, L, L, 64),
            "pde_logits": torch.rand(2, L, L, 64),
            "chain_iid_token_lvl": torch.randint(0, 10, (L,)).numpy(),
        }
    }
    is_real_atom = ChemData().heavyatom_mask[outputs["confidence"]["rf2aa_seq"]]
    outputs["confidence"]["is_real_atom"] = is_real_atom
    data = AttrDict({
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
        }
    })
    confidence_writer = WriteAF3Confidence(**data)
    df = confidence_writer(None, outputs, {"example_id": "test", "is_real_atom": is_real_atom})
    num_chains = len(np.unique(outputs["confidence"]["chain_iid_token_lvl"]))
    num_interfaces = num_chains * (num_chains - 1) // 2
    num_batches = outputs["confidence"]["plddt_logits"].shape[0]

    target_columns = ['example_id', 'chain_chainwise', 'chainwise_plddt', 'chainwise_pde', 'chainwise_pae', 'overall_plddt', 'overall_pde', 'overall_pae', 'batch_idx', 'chain_i_interface', 'chain_j_interface', 'pae_interface', 'pde_interface']
    assert df.columns.tolist() == target_columns, 'Dataframe columns not set correctly'
    assert df.shape == (num_batches * (num_interfaces + num_chains), len(target_columns)), 'Dataframe shape not set correctly'


def test_unbin_pae_logits():
    L = 100
    max_distance = 32
    n_bins = 64
    init = partial(initialize_chemdata)
    init()

    seed_all(42)
    outputs = {
        "confidence": {
            "rf2aa_seq": torch.randint(0, 21, (L,)),
            "plddt_logits": torch.rand(1, L, ChemData().NHEAVY, 50),
            "pae_logits": torch.rand(1, L, L, 64),
            "pde_logits": torch.rand(1, L, L, 64),
            "chain_iid_token_lvl": torch.randint(0, 10, (L,)).numpy(),
        }
    }
    data = AttrDict({
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
        }
    })
    is_real_atom = ChemData().heavyatom_mask[outputs["confidence"]["rf2aa_seq"]]
    outputs["confidence"]["is_real_atom"] = is_real_atom

    pae_unbinned = unbin_logits(
        outputs["confidence"]["pae_logits"].permute(0,3,1,2).float(),
        max_distance=max_distance,
        num_bins=n_bins
    )


    assert torch.allclose(torch.mean(pae_unbinned), torch.tensor(15.99) , atol=1e-2)
    assert pae_unbinned.shape == (1, L, L)

def test_unbin_pde_logits():
    L = 100
    max_distance = 32
    n_bins = 64
    init = partial(initialize_chemdata)
    init()

    seed_all(42)
    outputs = {
        "confidence": {
            "rf2aa_seq": torch.randint(0, 21, (L,)),
            "plddt_logits": torch.rand(1, L, ChemData().NHEAVY, 50),
            "pae_logits": torch.rand(1, L, L, 64),
            "pde_logits": torch.rand(1, L, L, 64),
            "chain_iid_token_lvl": torch.randint(0, 10, (L,)).numpy(),
        }
    }
    data = AttrDict({
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
        }
    })
    is_real_atom = ChemData().heavyatom_mask[outputs["confidence"]["rf2aa_seq"]]
    outputs["confidence"]["is_real_atom"] = is_real_atom

    pde_unbinned = unbin_logits(
        outputs["confidence"]["pae_logits"].permute(0,3,1,2).float(),
        max_distance=max_distance,
        num_bins=n_bins
    )

    assert torch.allclose(torch.mean(pde_unbinned), pde , atol=1e-2)
    assert torch.allclose(torch.mean(pde_unbinned), torch.tensor(16.00) , atol=1e-2)

    assert pde_unbinned.shape == (1, L, L)

def test_unbin_plddt_logits():
    L = 100
    max_distance = 1.0
    n_bins = 50
    init = partial(initialize_chemdata)
    init()

    seed_all(42)
    outputs = {
        "confidence": {
            "rf2aa_seq": torch.randint(0, 21, (L,)),
            "plddt_logits": torch.rand(1, L, ChemData().NHEAVY, 50),
            "pae_logits": torch.rand(1, L, L, 64),
            "pde_logits": torch.rand(1, L, L, 64),
            "chain_iid_token_lvl": torch.randint(0, 10, (L,)).numpy(),
        }
    }
    data = AttrDict({
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
        }
    })
    is_real_atom = ChemData().heavyatom_mask[outputs["confidence"]["rf2aa_seq"]]
    outputs["confidence"]["is_real_atom"] = is_real_atom

    plddt_unbinned = unbin_logits(outputs["confidence"]["plddt_logits"].permute(0,3,1,2).float(), max_distance, n_bins)

    assert torch.allclose(torch.mean(plddt_unbinned), plddt, atol=1e-2)
    assert plddt_unbinned.shape == (1, L, ChemData().NHEAVY)

def test_bin_midpoints():
    max_distance = 32
    num_bins = 64
    expected_bins = torch.linspace(0.25, 31.75, 64, device="cpu")
    pae_bins = find_bin_midpoints(max_distance, num_bins)
    assert torch.allclose(pae_bins, expected_bins)
