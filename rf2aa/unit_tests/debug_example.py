import torch
from rf2aa.data_loader import (
    get_train_valid_set,
    default_dataloader_params,
    loader_sm_compl_assembly,
)

from typing import Tuple
from argparse import ArgumentParser


def create_parser():
    parser = ArgumentParser()
    parser.add_argument("pdb_id", type=str, help="PDB ID of item to load")
    parser.add_argument("task", type=str, help="which task/dataset to load from")
    parser.add_argument(
        "--integer_index", type=int, default=0, help="which item in filtered df to load"
    )
    parser.add_argument(
        "--load_from_train",
        action="store_true",
    )
    return parser


def load_negative_item(
    pdb_id: str,
    train_valid_sets: Tuple,
    task: str = "sm_compl_permuted_neg",
    integer_index: int = 0,
    load_from_train: bool = False,
):
    _, _, _, train_dict, valid_dict, _, chid2hash, chid2taxid = train_valid_sets
    if load_from_train:
        data_df = train_dict[task]
        filtered_data_df = data_df[data_df["CHAINID"] == pdb_id]
    else:
        data_df = valid_dict[task]
        filtered_data_df = data_df[data_df["CHAINID"] == pdb_id]

    item = filtered_data_df.iloc[integer_index].to_dict()

    negative_item = None
    ligand_smiles_str = None
    if "LIG_SMILES" in item:
        ligand_smiles_list = item["LIG_SMILES"]
        ligand_smiles_str = ligand_smiles_list[0]
        if task == "dude_inactives":
            negative_item = True
    elif "NEG_REMAP_INDEX" in item:
        negative_item_df_index = item["NEG_REMAP_INDEX"]
        negative_item = data_df.loc[negative_item_df_index].to_dict()
    elif "REMAP_INDICES" in item:
        negative_item_choices = item["REMAP_INDICES"]
        negative_item_df_index = negative_item_choices[0]
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
    parser = create_parser()
    args = parser.parse_args()
    train_valid_sets = get_train_valid_set(default_dataloader_params)

    negative_out = load_negative_item(
        pdb_id=args.pdb_id,
        train_valid_sets=train_valid_sets,
        task=args.task,
        integer_index=args.integer_index,
        load_from_train=args.load_from_train,
    )
    for index, sub_item in enumerate(negative_out):
        if isinstance(sub_item, torch.Tensor):
            print(f"{index}: {sub_item.shape}, {sub_item.dtype}")
        else:
            print(f"{index}: {sub_item}")


if __name__ == "__main__":
    main()
