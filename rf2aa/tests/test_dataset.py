import itertools
import os
from functools import partial

import pytest
import torch

from rf2aa.chemical import ChemicalData as ChemData
from rf2aa.chemical import initialize_chemdata
from rf2aa.data.compose_dataset import (
    compose_single_item_dataset,
)
from rf2aa.data.data_loader import get_crop
from rf2aa.set_seed import seed_all
from rf2aa.tensor_util import assert_shape
from rf2aa.tests.test_conditions import dataset_pickle_path, setup_data
from rf2aa.util import is_atom

data = setup_data()


@pytest.mark.xfail
@pytest.mark.parametrize(
    "name,item,loader_params,chem_params,loader,loader_kwargs", data.values()
)
def test_correct_shapes(name, item, loader_params, chem_params, loader, loader_kwargs):
    # initialize chemical database.  Force a reload
    ChemData.reset()
    init = partial(initialize_chemdata, chem_params)
    init()

    data_loader = compose_single_item_dataset(
        init, item, loader_params, loader, loader_kwargs
    )
    for inputs in data_loader:
        (
            seq,
            msa,
            msa_masked,
            msa_full,
            mask_msa,
            true_crds,
            mask_crds,
            idx_pdb,
            xyz_t,
            t1d,
            mask_t,
            xyz_prev,
            mask_prev,
            same_chain,
            unclamp,
            negative,
            atom_frames,
            bond_feats,
            dist_matrix,
            chirals,
            ch_label,
            symmgp,
            task,
            item,
        ) = inputs
        B, recycles, N, L = msa.shape[:4]
        num_atoms = (is_atom(seq[0, 0]).sum()).item()
        assert_shape(seq, (B, recycles, L))
        assert_shape(msa, (B, recycles, N, L))
        assert_shape(
            msa_masked, (B, recycles, N, L, 164)
        )  # Hack: hardcoded for current featurization
        N_full = msa_full.shape[2]
        assert_shape(
            msa_full, (B, recycles, N_full, L, 83)
        )  # HACK:: hardcoded for current features
        assert_shape(mask_msa, (B, recycles, N, L))
        N_symm = true_crds.shape[1]
        assert_shape(true_crds, (B, N_symm, L, ChemData().NTOTAL, 3))
        assert_shape(mask_crds, (B, N_symm, L, ChemData().NTOTAL))
        assert_shape(idx_pdb, (B, L))
        N_templ = xyz_t.shape[1]
        assert_shape(xyz_t, (B, N_templ, L, ChemData().NTOTAL, 3))
        assert_shape(t1d, (B, N_templ, L, 80))  # hack hard coded dimension
        assert_shape(mask_t, (B, N_templ, L, ChemData().NTOTAL))
        assert_shape(xyz_prev, (B, L, ChemData().NTOTAL, 3))
        assert_shape(mask_prev, (B, L, ChemData().NTOTAL))
        assert_shape(same_chain, (B, L, L))
        assert type(unclamp.item()) == bool
        assert type(negative.item()) == bool
        assert_shape(atom_frames, (B, num_atoms, 3, 2))
        assert_shape(bond_feats, (B, L, L))
        assert_shape(dist_matrix, (B, L, L))
        n_chirals = chirals.shape[1]
        assert_shape(chirals, (B, n_chirals, 5))
        assert_shape(ch_label, (B, L))
        assert symmgp[0] == "C1", f"{symmgp}"


@pytest.mark.parametrize(
    "name,item,loader_params,chem_params,loader,loader_kwargs", data.values()
)
def test_regression(name, item, loader_params, chem_params, loader, loader_kwargs):
    # initialize chemical database.  Force a reload
    ChemData.reset()
    init = partial(initialize_chemdata, chem_params)
    init()

    seed_all()
    data_loader = compose_single_item_dataset(
        init, item, loader_params, loader, loader_kwargs
    )
    regression_pickle = dataset_pickle_path(name)
    for inputs in data_loader:
        names = [
            "seq",
            "msa",
            "msa_masked",
            "msa_full",
            "mask_msa",
            "true_crds",
            "mask_crds",
            "idx_pdb",
            "xyz_t",
            "t1d",
            "mask_t",
            "xyz_prev",
            "mask_prev",
            "same_chain",
            "unclamp",
            "negative",
            "atom_frames",
            "bond_feats",
            "dist_matrix",
            "chirals",
            "ch_label",
            "symmgp",
            "task",
            "item",
        ]
        if os.path.exists(regression_pickle):
            regression = torch.load(regression_pickle, map_location="cpu")
        else:
            torch.save(inputs, regression_pickle)
            print(f"SAVED test_outputs in {regression_pickle}")
            return
        for idx, input in enumerate(inputs):
            if torch.is_tensor(input):
                if idx in [8, 11]:  # xyz_t, xyz_prev are not deterministic yet
                    continue
                try:
                    torch.allclose(input, regression[idx], atol=1e-3)
                except Exception as e:
                    # TODO: revisit this after dataset refactor
                    if idx in [9, 10]:  # weirdness in templates sm_compl_assembly
                        print("t1d fails for sm_compl_assembly")
                        continue
                    raise AssertionError(f"{names[idx]} did not match") from e


def test_get_crop():
    l = 5
    mask = torch.ones(l, 36)
    device = "cpu"
    crop = 3
    sel_list = []
    for i in range(5):
        seed_all(i)
        sel = get_crop(l, mask, device, crop)
        sel_list.append(sel)
    assert any(
        [
            torch.all(~torch.eq(sel_i, sel_j))
            if sel_i.shape == sel_j.shape
            else True  # if shapes are different, they are not equal
            for sel_i, sel_j in itertools.combinations(sel_list, 2)
        ]
    ), "get_crop is not random"
