from typing import Tuple
import torch
import numpy as np
from rf2aa.data_loader import (
    get_train_valid_set,
    default_dataloader_params,
    loader_sm_compl_assembly,
)


def load_negative_item(train_valid_sets: Tuple, task: str = "sm_compl_permuted_neg", load_from_train: bool = False):
    train_id_dict, valid_id_dict, _, train_dict, valid_dict, _, chid2hash, chid2taxid = train_valid_sets
    if load_from_train:
        data_df = train_dict[task]
        filtered_data_df = data_df[data_df["CLUSTER"] == train_id_dict[task][0]]
    else:
        data_df = valid_dict[task]
        filtered_data_df = data_df[data_df["CLUSTER"] == valid_id_dict[task][0]]

    item = filtered_data_df.iloc[0].to_dict()

    negative_item = None
    ligand_smiles_str = None
    if "LIG_SMILES" in item:
        ligand_smiles_list = item["LIG_SMILES"]
        ligand_smiles_str = np.random.choice(ligand_smiles_list)
        if task == "dude_inactives":
            negative_item = True
    elif "NEG_REMAP_INDEX" in item:
        negative_item_df_index = item["NEG_REMAP_INDEX"]
        negative_item = data_df.loc[negative_item_df_index].to_dict()
    elif "REMAP_INDICES" in item:
        negative_item_choices = item["REMAP_INDICES"]
        negative_item_df_index = np.random.choice(negative_item_choices)
        negative_item = data_df.loc[negative_item_df_index].to_dict()

    loader_out = loader_sm_compl_assembly(
        item,
        default_dataloader_params,
        chid2hash=chid2hash,
        chid2taxid=chid2taxid,
        task=task,
        num_protein_chains=1,
        num_ligand_chains=1,
        selected_negative_item=negative_item,
        ligand_smiles_str=ligand_smiles_str,
    )
    return loader_out


def main():
    train_valid_sets = get_train_valid_set(default_dataloader_params)
    tasks = [
        "sm_compl_docked_neg",
        "sm_compl_permuted_neg",
        "sm_compl_furthest_neg",
        "dude_actives",
        "dude_inactives",
    ]

    for task in tasks:
        print(task.center(50, "="))
        negative_out = load_negative_item(train_valid_sets, task=task)
        for index, sub_item in enumerate(negative_out):
            if isinstance(sub_item, torch.Tensor):
                print(f"{index}: {sub_item.shape}, {sub_item.dtype}")
            else:
                print(f"{index}: {sub_item}")


if __name__ == "__main__":
    main()
