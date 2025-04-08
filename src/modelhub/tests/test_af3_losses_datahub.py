import random

import numpy as np
import pytest
import torch
from datahub.transforms.center_random_augmentation import random_augmentation
from tests.datasets.conftest import (
    AF3_PDB_DATASET,
)
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from modelhub.alignment import weighted_rigid_align
from modelhub.loss.af3_losses import DiffusionLoss


@pytest.mark.parametrize("pdb_dataset", [AF3_PDB_DATASET])
def test_weighted_rigid_align(pdb_dataset):
    """
    test that the weighted_rigid_align function aligns the coordinates correctly.
    """
    NUM_RANDOM_EXAMPLES = 10

    # Set the seed for reproducibility
    seed = 42

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # Select deterministic examples to profile
    # NOTE: TEST_FILTERS ensures we don't end up with any huge examples that would slow down the test
    deterministic_indices = np.random.choice(
        len(pdb_dataset), NUM_RANDOM_EXAMPLES, replace=False
    )

    # Create a Subset of the dataset with the selected indices
    subset = Subset(pdb_dataset, deterministic_indices)

    # Create a DataLoader for the subset
    data_loader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x,
    )

    for sample in tqdm(data_loader):
        example = sample[0]
        network_input = {
            "X_noisy_L": example["ground_truth"]["coord_atom_lvl"] + example["noise"],
            "t": example["t"],
            "f": example["feats"],
        }

        loss_input = {
            "X_gt_L": example["ground_truth"]["coord_atom_lvl"],
            "crd_mask_L": example["ground_truth"]["mask_atom_lvl"],
            "X_rep_atoms_I": example["ground_truth"]["coord_token_lvl"],
            "crd_mask_rep_atoms_I": example["ground_truth"]["mask_token_lvl"],
        }
        network_output = {
            "X_L": example["ground_truth"]["coord_atom_lvl"],
        }
        w_L = torch.ones_like(loss_input["crd_mask_L"])
        is_resolved = loss_input["crd_mask_L"]
        X_align_L = weighted_rigid_align(
            network_output["X_L"],
            loss_input["X_gt_L"],
            loss_input["crd_mask_L"][0],
            w_L,
        )
        assert torch.allclose(
            X_align_L[is_resolved], loss_input["X_gt_L"][is_resolved], atol=1e-4
        ), "The aligned coordinates are not close to the ground truth coordinates."
        D = loss_input["X_gt_L"].shape[0]
        rotated_X = random_augmentation(network_output["X_L"], D)
        rotated_X_clone = rotated_X.clone()
        X_align_L = weighted_rigid_align(
            rotated_X,
            loss_input["X_gt_L"],
            loss_input["crd_mask_L"][0],
            w_L,
        )

        assert torch.allclose(
            X_align_L[is_resolved], rotated_X_clone[is_resolved], atol=1e-4
        ), "The aligned coordinates are not close to the ground truth coordinates."


@pytest.mark.parametrize("pdb_dataset", [AF3_PDB_DATASET])
def test_diffusion_loss(pdb_dataset):
    """
    Tests the diffusion loss function.
    """
    loss = DiffusionLoss(
        weight=1.0,
        sigma_data=16.0,
        alpha_dna=5.0,
        alpha_rna=5.0,
        alpha_ligand=10.0,
        edm_lambda=True,
        se3_invariant_loss=True,
        clamp_diffusion_loss=False,
    )

    NUM_RANDOM_EXAMPLES = 5

    # Set the seed for reproducibility
    seed = 42

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # Select deterministic examples to profile
    # NOTE: TEST_FILTERS ensures we don't end up with any huge examples that would slow down the test
    deterministic_indices = np.random.choice(
        len(pdb_dataset), NUM_RANDOM_EXAMPLES, replace=False
    )

    # Create a Subset of the dataset with the selected indices
    subset = Subset(pdb_dataset, deterministic_indices)

    # Create a DataLoader for the subset
    data_loader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x,
    )

    for sample in tqdm(data_loader):
        example = sample[0]
        network_input = {
            "X_noisy_L": example["ground_truth"]["coord_atom_lvl"] + example["noise"],
            "t": example["t"],
            "f": example["feats"],
        }

        loss_input = {
            "X_gt_L": example["ground_truth"]["coord_atom_lvl"],
            "crd_mask_L": example["ground_truth"]["mask_atom_lvl"],
            "X_rep_atoms_I": example["ground_truth"]["coord_token_lvl"],
            "crd_mask_rep_atoms_I": example["ground_truth"]["mask_token_lvl"],
        }
        network_output = {
            "X_L": example["ground_truth"]["coord_atom_lvl"],
        }
        loss_value, loss_dict = loss(
            network_input,
            network_output,
            loss_input,
        )
        diffusion_loss = loss_dict["diffusion_loss"].mean()
        assert torch.allclose(diffusion_loss, torch.tensor(0.0)), (
            f"The loss value is not zero when provided the ground truth. got: {diffusion_loss}"
        )

        # rescale the noisy inputs and make sure their loss is 1
        sigma_data = loss.sigma_data
        t = network_input["t"]
        null_pred = (sigma_data**2 / (sigma_data**2 + t**2))[
            ..., None, None
        ] * network_input["X_noisy_L"]
        loss_value, loss_dict = loss(
            network_input,
            {"X_L": null_pred},
            loss_input,
        )
        diffusion_loss = loss_dict["diffusion_loss"].mean()
        diffusion_std_dev = loss_dict["diffusion_loss"].std()
        # NOTE: this test is relatively brittle
        assert torch.allclose(diffusion_loss, torch.tensor(1.0), atol=0.5), (
            f"The average loss value is not 1 (within 0.25 tolerance) when provided the noisy inputs. got: {diffusion_loss}"
        )
        assert torch.allclose(diffusion_std_dev, torch.tensor(0.0), atol=0.5), (
            f"The std deviation value is not within 0.2 when provided the noisy inputs. got: {diffusion_std_dev}"
        )
