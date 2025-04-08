from pathlib import Path
from typing import Dict

import hydra
import numpy as np
import pandas as pd
import torch

from modelhub.data.compose_dataset import set_data_loader_params
from modelhub.data.data_loader import get_train_valid_set


def _compute_name_of_row(row: pd.Series) -> str:
    """
    Computes a unique identifier for single chain protein ligand complexes.
    """
    ligand = row["LIGAND"][0]
    lig_str = ligand[0] + ligand[1] + "-" + ligand[2]
    name = row["CHAINID"] + "_asm" + str(int(row["ASSEMBLY"])) + "_" + lig_str
    return name


def add_negative_sets(
    train_ID_dict: Dict[str, np.ndarray],
    valid_ID_dict: Dict[str, np.ndarray],
    weights_dict: Dict[str, np.ndarray],
    train_dict: Dict[str, pd.DataFrame],
    valid_dict: Dict[str, pd.DataFrame],
    base_path: Path = Path("/projects/ml/RF2_allatom"),
    farthest_len_min: int = 356,
):
    """add_negative_sets This function adds three types of
    negative examples for protein small molecule complexes:
        - Farthest ball crops: crops large proteins around the residue
        farthest from the ligand, assuming that the ligand won't bind there.
        - Docked negatives: ligands that AutoDock Vina think are
        unlikely to bind to the true binding site of a different ligand
        for a given protein.
        - Permuted negatives: property matched ligands with low fingerprint
        similarity to the original ligand, following the DUD-e protocol
        for negative decoy generation.

    The function basically just reads in the precomputed data and matches
    it to the phase 3 datasets. The actual computation is done
    in the branch psturm_negatives.

    Args:
        train_ID_dict (Dict[str, np.ndarray]): See get_train_valid_set.
        valid_ID_dict (Dict[str, np.ndarray]): See get_train_valid_set.
        weights_dict (Dict[str, np.ndarray]): See get_train_valid_set.
        train_dict (Dict[str, pd.DataFrame]): See get_train_valid_set.
        valid_dict (Dict[str, pd.DataFrame]): See get_train_valid_set.
        base_path (Path, optional): Where the negative datasets are precomputed.
            Defaults to Path("/projects/ml/RF2_allatom").
        farthest_len_min (int, optional): The minimum size for proteins that are
            going to be in the farthest ball crop negative set. Defaults to 356.

    Returns:
        _type_: A tuple, (train_ID_dict, valid_ID_dict, weights_dict, train_dict, valid_dict),
        with the negative sets added.
    """
    sm_compl_train = train_dict["sm_compl"]
    sm_compl_valid = valid_dict["sm_compl"]

    autodocked_train = pd.read_pickle(base_path / "autodocked_decoys_train.pkl")
    autodocked_valid = pd.read_pickle(base_path / "autodocked_decoys_valid.pkl")
    property_matched_train = pd.read_pickle(
        base_path / "property_matched_decoys_train.pkl"
    )
    property_matched_valid = pd.read_pickle(
        base_path / "property_matched_decoys_valid.pkl"
    )

    sm_compl_train["name"] = [
        _compute_name_of_row(row) for _, row in sm_compl_train.iterrows()
    ]
    sm_compl_valid["name"] = [
        _compute_name_of_row(row) for _, row in sm_compl_valid.iterrows()
    ]
    autodocked_train["name"] = [
        _compute_name_of_row(row) for _, row in autodocked_train.iterrows()
    ]
    autodocked_valid["name"] = [
        _compute_name_of_row(row) for _, row in autodocked_valid.iterrows()
    ]
    property_matched_train["name"] = [
        _compute_name_of_row(row) for _, row in property_matched_train.iterrows()
    ]
    property_matched_valid["name"] = [
        _compute_name_of_row(row) for _, row in property_matched_valid.iterrows()
    ]

    sm_compl_names_train = set(sm_compl_train["name"].unique())
    sm_compl_names_valid = set(sm_compl_valid["name"].unique())

    autodocked_train_filtered = autodocked_train[
        (autodocked_train["name"].isin(sm_compl_names_train))
    ]
    autodocked_valid_filtered = autodocked_valid[
        autodocked_valid["name"].isin(sm_compl_names_valid)
    ]
    property_matched_train_filtered = property_matched_train[
        property_matched_train["name"].isin(sm_compl_names_train)
    ]
    property_matched_valid_filtered = property_matched_valid[
        property_matched_valid["name"].isin(sm_compl_names_valid)
    ]

    furthest_negative_train = sm_compl_train[
        sm_compl_train["LEN_EXIST"] > farthest_len_min
    ].copy()
    furthest_negative_valid = sm_compl_valid[
        sm_compl_valid["LEN_EXIST"] > farthest_len_min
    ].copy()

    train_cluster_to_weight = dict(
        zip(train_ID_dict["sm_compl"], weights_dict["sm_compl"])
    )

    autocked_ids_train = autodocked_train_filtered.dropna(subset="NONBINDING_LIGANDS")[
        "CLUSTER"
    ].unique()
    autocked_ids_valid = autodocked_valid_filtered.dropna(subset="NONBINDING_LIGANDS")[
        "CLUSTER"
    ].unique()
    property_matched_ids_train = property_matched_train_filtered.dropna(
        subset="NONBINDING_LIGANDS"
    )["CLUSTER"].unique()
    property_matched_ids_valid = property_matched_valid_filtered.dropna(
        subset="NONBINDING_LIGANDS"
    )["CLUSTER"].unique()
    furthest_negative_ids_train = furthest_negative_train["CLUSTER"].unique()
    furthest_negative_ids_valid = furthest_negative_valid["CLUSTER"].unique()

    autodocked_train_weights = torch.stack(
        [train_cluster_to_weight[cluster_id] for cluster_id in autocked_ids_train]
    )
    property_matched_train_weights = torch.stack(
        [
            train_cluster_to_weight[cluster_id]
            for cluster_id in property_matched_ids_train
        ]
    )
    furthest_negative_train_weights = torch.stack(
        [
            train_cluster_to_weight[cluster_id]
            for cluster_id in furthest_negative_ids_train
        ]
    )

    train_ID_dict["sm_compl_docked_neg"] = autocked_ids_train
    valid_ID_dict["sm_compl_docked_neg"] = autocked_ids_valid
    weights_dict["sm_compl_docked_neg"] = autodocked_train_weights
    train_dict["sm_compl_docked_neg"] = autodocked_train
    valid_dict["sm_compl_docked_neg"] = autodocked_valid

    train_ID_dict["sm_compl_permuted_neg"] = property_matched_ids_train
    valid_ID_dict["sm_compl_permuted_neg"] = property_matched_ids_valid
    weights_dict["sm_compl_permuted_neg"] = property_matched_train_weights
    train_dict["sm_compl_permuted_neg"] = property_matched_train
    valid_dict["sm_compl_permuted_neg"] = property_matched_valid

    train_ID_dict["sm_compl_furthest_neg"] = furthest_negative_ids_train
    valid_ID_dict["sm_compl_furthest_neg"] = furthest_negative_ids_valid
    weights_dict["sm_compl_furthest_neg"] = furthest_negative_train_weights
    train_dict["sm_compl_furthest_neg"] = furthest_negative_train
    valid_dict["sm_compl_furthest_neg"] = furthest_negative_valid

    # These next two are only validation sets, so they don't get added
    # to the training data. Important to note: the structures
    # here are not defined, so one should not use them for structural loss comparisons.
    # They are only meant to be data for the binder/non-binder prediction task.
    dude_actives_df = pd.read_pickle(base_path / "dude_valid_actives.pkl")
    valid_dict["dude_actives"] = dude_actives_df
    valid_ID_dict["dude_actives"] = dude_actives_df["CLUSTER"].unique()

    dude_inactives_df = pd.read_pickle(base_path / "dude_valid_inactives.pkl")
    valid_dict["dude_inactives"] = dude_inactives_df
    valid_ID_dict["dude_inactives"] = dude_inactives_df["CLUSTER"].unique()
    return train_ID_dict, valid_ID_dict, weights_dict, train_dict, valid_dict


def main(
    in_file: str = "/projects/ml/RF2_allatom/all_sample_lengths_crop_1K.pt",
    out_file: str = "/projects/ml/RF2_allatom/all_sample_lengths_crop_1K_no_negatives.pt",
):
    with hydra.initialize(version_base=None, config_path="../../rf2aa/config/train"):
        config = hydra.compose(config_name="base")
    loader_params = set_data_loader_params(config.loader_params)
    (
        train_ID_dict,
        valid_ID_dict,
        weights_dict,
        train_dict,
        valid_dict,
        homo,
        chid2hash,
        chid2taxid,
        chid2smpartners,
    ) = get_train_valid_set(loader_params)
    train_ID_dict, valid_ID_dict, weights_dict, train_dict, valid_dict = (
        add_negative_sets(
            train_ID_dict,
            valid_ID_dict,
            weights_dict,
            train_dict,
            valid_dict,
        )
    )

    train_ID_dict["atomize_pdb"] = train_ID_dict["pdb"]
    valid_ID_dict["atomize_pdb"] = valid_ID_dict["pdb"]
    weights_dict["atomize_pdb"] = weights_dict["pdb"]
    train_dict["atomize_pdb"] = train_dict["pdb"]
    valid_dict["atomize_pdb"] = valid_dict["pdb"]

    # define atomize_pdb train/valid sets, which use the same examples as pdb set
    train_ID_dict["atomize_complex"] = train_ID_dict["compl"]
    valid_ID_dict["atomize_complex"] = valid_ID_dict["compl"]
    weights_dict["atomize_complex"] = weights_dict["compl"]
    train_dict["atomize_complex"] = train_dict["compl"]
    valid_dict["atomize_complex"] = valid_dict["compl"]

    correct_dataset_ordering_old = [
        "pdb",
        "fb",
        "compl",
        "neg_compl",
        "na_compl",
        "neg_na_compl",
        "distil_tf",
        "tf",
        "neg_tf",
        "rna",
        "dna",
        "sm_compl",
        "metal_compl",
        "sm_compl_multi",
        "sm_compl_covale",
        "sm_compl_asmb",
        "sm",
        "sm_compl_docked_neg",
        "sm_compl_permuted_neg",
        "sm_compl_furthest_neg",
        "atomize_pdb",
        "atomize_complex",
    ]
    datasets_to_remove = [
        "sm_compl_docked_neg",
        "sm_compl_permuted_neg",
        "sm_compl_furthest_neg",
    ]
    lengths_tensor = torch.load(in_file)

    lengths_of_datasets = [
        len(train_ID_dict[key]) for key in correct_dataset_ordering_old
    ]
    is_valid_mask = torch.full((lengths_tensor.shape[0],), True, dtype=torch.bool)

    for dataset in datasets_to_remove:
        dataset_idx = correct_dataset_ordering_old.index(dataset)
        dataset_len = lengths_of_datasets[dataset_idx]
        offset = sum(lengths_of_datasets[:dataset_idx])
        is_valid_mask[offset : offset + dataset_len] = False

    lengths_tensor_new = lengths_tensor[is_valid_mask]
    torch.save(lengths_tensor_new, out_file)


if __name__ == "__main__":
    main()
