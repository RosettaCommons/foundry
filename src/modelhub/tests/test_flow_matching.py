from functools import partial
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf

from modelhub.chemical import ChemicalData as ChemData
from modelhub.chemical import initialize_chemdata
from modelhub.data.dataloader_adaptor import prepare_input_fm
from modelhub.flow_matching.data_transforms import (
    center_chain_backbone,
    convert_dataloader_inputs_to_rigids,
    rigids_to_xyz,
)
from modelhub.flow_matching.sampler import Sampler
from modelhub.loss.loss import calc_crd_rmsd
from modelhub.tensor_util import assert_equal
from modelhub.tests.test_conditions import dataset_pickle_path, datasets, model_pickle_path
from modelhub.training.recycling import unpack_outputs
from modelhub.util_module import XYZConverter


def mock_model_outputs(dataset):
    filepath = model_pickle_path(dataset, "rf2aa")
    out = torch.load(filepath, map_location="cuda:0")
    output = unpack_outputs(out["outputs"], out["latents"], return_raw=False)
    return output


@pytest.mark.parametrize("dataset", datasets.keys())
def test_convert_dataloader_inputs_to_rigids(dataset):
    chem_params = OmegaConf.create(
        {
            "use_phospate_frames_for_NA": False,
        }
    )
    ChemData.reset()
    init = partial(initialize_chemdata, chem_params)
    init()

    dataset_pickle = dataset_pickle_path(dataset)
    inputs = torch.load(dataset_pickle, map_location="cpu")
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
    seq_unmasked = msa[:, 0, 0][0]
    L = seq_unmasked.shape[0]
    rigids = convert_dataloader_inputs_to_rigids(inputs, "cpu")

    expected_keys = [
        "aatypes_1",
        "rotmats_1",
        "trans_1",
        "res_mask",
        "diffuse_mask",
        "diffuse_mask_seq",
    ]
    assert set(rigids.keys()) == set(expected_keys)

    assert_equal(rigids["aatypes_1"], seq_unmasked[None])
    assert rigids["res_mask"].shape == (1, L)

    reconstructed_xyz = rigids_to_xyz(rigids["rotmats_1"], rigids["trans_1"])
    if len(true_crds.shape) == 4:
        true_crds = true_crds[None]
    if len(mask_crds.shape) == 3:
        mask_crds = mask_crds[None]

    true_crds = true_crds[0][0][..., :3, :]
    mask_crds = mask_crds[0][0][..., :3]
    true_crds = center_chain_backbone(true_crds, mask_crds)[None]
    mask_crds = mask_crds[None]
    rmsd = calc_crd_rmsd(reconstructed_xyz, true_crds, mask_crds)

    # note: reconstructing nucleic acid examples is less accurate and requires a higher tolerance
    assert rmsd < 1e-1
    assert torch.allclose(
        reconstructed_xyz[mask_crds], true_crds[mask_crds], atol=5 * 1e-1
    )


@pytest.mark.parametrize("dataset", datasets.keys())
def test_prepare_input_fm(dataset):
    chem_params = OmegaConf.create(
        {
            "use_phospate_frames_for_NA": False,
        }
    )
    ChemData.reset()
    init = partial(initialize_chemdata, chem_params)
    init()

    dataset_pickle = dataset_pickle_path(dataset)
    inputs = torch.load(dataset_pickle, map_location="cpu")

    xyz_converter = XYZConverter().to("cuda:0")
    network_input = prepare_input_fm(inputs, MockInterpolant(), xyz_converter, "cuda:0")
    # test that MockInterpolant was called
    # test that the template features were updated
    # test that t2d does not have values for atoms in the rotation areas


@pytest.mark.parametrize("dataset", datasets.keys())
def test_sampler(dataset):
    chem_params = OmegaConf.create(
        {
            "use_phospate_frames_for_NA": False,
        }
    )
    ChemData.reset()
    init = partial(initialize_chemdata, chem_params)
    init()

    mock_model = MockModel()
    xyz_converter = XYZConverter().to("cuda:0")
    sampler = Sampler(
        model=mock_model,
        num_timesteps=4,
        min_t=0.1,
        interpolant=MockInterpolant(),
        xyz_converter=xyz_converter,
        is_training=True,
    )
    dataset_pickle = dataset_pickle_path(dataset)
    inputs = torch.load(dataset_pickle, map_location=sampler.device)
    outputs = mock_model_outputs(dataset)
    with patch(
        "modelhub.flow_matching.sampler.recycle_step_packed", return_value=outputs
    ) as recycling_fn:
        with patch("modelhub.flow_matching.sampler.Sampler._take_step") as euler_step:
            sampler.sample(inputs)
            recycling_fn.assert_called_once()


class MockInterpolant:
    @property
    def _device(self):
        return "cuda:0"

    @property
    def device(self):
        return "cuda:0"

    def set_device(self, gpu):
        pass

    def corrupt_batch(self, batch):
        batch["rotmats_t"], batch["trans_t"] = batch["rotmats_1"], batch["trans_1"]
        return batch


class MockModel:
    def __init__(self) -> None:
        self.device = "cuda:0"

    def __call__(self, batch):
        return (None,) * 12
