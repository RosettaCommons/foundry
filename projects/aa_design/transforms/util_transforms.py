# see datahub.ransforms.feature_aggregation
import copy
import time
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from biotite.structure import AtomArray
from cifutils.enums import ChainTypeInfo
from datahub.encoding_definitions import AF3SequenceEncoding
from datahub.transforms._checks import (
    check_atom_array_annotation,
    check_contains_keys,
    check_is_instance,
)
from datahub.transforms.atom_array import get_within_entity_idx
from datahub.transforms.base import Transform
from datahub.utils.token import (
    get_af3_token_representative_idxs,
    get_token_starts,
    is_glycine,
    is_protein_unknown,
    is_purine,
    is_pyramidine,
    is_standard_aa_not_glycine,
    is_unknown_nucleotide,
)


class TimerWrapper(Transform):
    def check_input(self, *args, **kwargs):
        pass

    def __init__(self, transform):
        self.transform = transform
    
    def forward(self, data):
        start = time.time()
        data = self.transform.forward(data)
        print(f"Time taken: {time.time() - start} s  || Transform: {self.transform}")
        return data


class AggregateFeaturesLikeAF3WithoutMSA(Transform):
    """
    Exactly like AggregateFeaturesLikeAF3 but without MSAs

    Removed comments for readability, no additional code is in this function, just removed msa parts
    """

    requires_previous_transforms = [
        "AtomizeByCCDName",
        "EncodeAF3TokenLevelFeatures",
    ]
    incompatible_previous_transforms = ["AggregateFeaturesLikeAF3", "AggregateFeaturesLikeAF3WithoutMSA"]

    def check_input(self, data) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["coord_to_be_noised", "chain_iid", "occupancy"])

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregates features into the format expected by AlphaFold 3.

        This method processes the input data, combining MSA features, ground truth
        structures, and other relevant information into a standardized format.

        Args:
            data (Dict[str, Any]): The input data dictionary containing MSA features,
                atom array, and other relevant information.

        Returns:
            Dict[str, Any]: The processed data dictionary with aggregated features.
        """
        # Initialize feats dictionary if not present
        if "feats" not in data:
            data["feats"] = {}

        data["feats"]["ref_atom_name_chars"] = F.one_hot(data["feats"]["ref_atom_name_chars"].long(), num_classes=64)
        data["feats"]["ref_element"] = F.one_hot(data["feats"]["ref_element"].long(), num_classes=128)
        data["feats"]["ref_pos"] = torch.nan_to_num(data["feats"]["ref_pos"], nan=0.0)

        # Process ground truth structure
        atom_array = data["atom_array"]
        coord_atom_lvl = atom_array.coord
        mask_atom_lvl = atom_array.occupancy > 0.0
        _token_rep_idxs = get_af3_token_representative_idxs(atom_array)
        coord_token_lvl = atom_array.coord[_token_rep_idxs]
        mask_token_lvl = atom_array.occupancy[_token_rep_idxs] > 0.0
        token_starts = get_token_starts(atom_array)
        token_level_array = atom_array[token_starts]
        chain_iid_token_lvl = token_level_array.chain_iid
        if "ground_truth" not in data:
            data["ground_truth"] = {}

        data["ground_truth"].update(
            {
                "coord_atom_lvl": torch.tensor(coord_atom_lvl),  # [n_atoms, 3]
                "mask_atom_lvl": torch.tensor(mask_atom_lvl),  # [n_atoms]
                "coord_token_lvl": torch.tensor(coord_token_lvl),  # [n_tokens, 3], using the representative tokens
                "mask_token_lvl": torch.tensor(mask_token_lvl),  # [n_tokens], using the representative tokens
                "chain_iid_token_lvl": chain_iid_token_lvl,  # numpy.ndarray of strings with shape (n_tokens,)
            }
        )
        data["coord_atom_lvl_to_be_noised"] = torch.tensor(atom_array.coord_to_be_noised)

        return data

def add_backbone_and_sidechain_annotations(atom_array: AtomArray) -> AtomArray:
    """
    Adds the backbone and sidechain annotations to the AtomArray.

    Args:
        atom_array (AtomArray): The AtomArray to which the annotations will be added.

    Returns:
        AtomArray: The AtomArray with the added annotations.
    """
    # Get the backbone atoms
    atomized = atom_array.atomize
    is_protein = np.isin(atom_array.chain_type, ChainTypeInfo.PROTEINS)
    backbone_atoms = ["N", "CA", "C", "O"]
    backbone_mask = np.isin(atom_array.atom_name, backbone_atoms) & is_protein
    backbone_mask = backbone_mask | atomized
    sidechain_mask = ~backbone_mask & ~atomized & is_protein

    # Add the annotations
    atom_array.set_annotation("is_backbone", backbone_mask)
    atom_array.set_annotation("is_sidechain", sidechain_mask)

    return atom_array

####################################################################################################
# Changes to datahub base transforms (instead of creating new branches)
####################################################################################################

# def is_protein_gap(ccd_code_array: np.ndarray) -> np.ndarray:
#     return np.asarray(ccd_code_array) == GAP

#from datahub.utils.token import get_af3_token_representative_masks
def get_af3_token_representative_masks(atom_array: AtomArray, central_atom: str = 'CA') -> np.ndarray:

    pyramidine_representative_atom = is_pyramidine(atom_array.res_name) & (atom_array.atom_name == "C2")
    purine_representative_atom = is_purine(atom_array.res_name) & (atom_array.atom_name == "C4")
    unknown_na_representative_atom = is_unknown_nucleotide(atom_array.res_name) & (atom_array.atom_name == "C4")

    glycine_representative_atom = is_glycine(atom_array.res_name) & (atom_array.atom_name == "CA")
    protein_residue_not_glycine_representative_atom = is_standard_aa_not_glycine(atom_array.res_name) & (
        atom_array.atom_name == central_atom  # only change
    )
    unknown_protein_residue_representative_atom = (is_protein_unknown(atom_array.res_name)) & (
        atom_array.atom_name == "CA"
    )
    atoms = atom_array.atomize

    return (
        pyramidine_representative_atom
        | purine_representative_atom
        | unknown_na_representative_atom
        | glycine_representative_atom
        | protein_residue_not_glycine_representative_atom
        | unknown_protein_residue_representative_atom
        | atoms
    )
# Fixed copy anntotation:
def copy_annotation(atom_array: AtomArray, annotation_to_copy: str, new_annotation: str) -> AtomArray:

    assert (
        new_annotation not in atom_array.get_annotation_categories() and new_annotation != "coord"
    ), f"Annotation {new_annotation} already exists in the AtomArray."

    if annotation_to_copy == "coord":
        # We must handle the special case of copying the coordinates (since "coord" is not technically an annotation)
        atom_array.set_annotation(new_annotation, atom_array.coord.copy())
    else:
        atom_array.set_annotation(new_annotation, copy.deepcopy(atom_array.get_annotation(annotation_to_copy)))

    return atom_array
class CopyAnnotation(Transform):
    """Copies an existing annotation from the AtomArray and assigns it a new name."""

    def __init__(self, annotation_to_copy: str, new_annotation: str):
        self.annotation_to_copy = annotation_to_copy
        self.new_annotation = new_annotation

    def check_input(self, data: dict):
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)

        assert (
            self.annotation_to_copy == "coord"
            or self.annotation_to_copy in data["atom_array"].get_annotation_categories()
        ), f"Annotation {self.annotation_to_copy} does not exist in the AtomArray."

        assert (
            self.new_annotation not in data["atom_array"].get_annotation_categories()
        ), f"Annotation {self.new_annotation} already exists in the AtomArray."

    def forward(self, data: dict) -> dict:
        data["atom_array"] = copy_annotation(
            data["atom_array"], annotation_to_copy=self.annotation_to_copy, new_annotation=self.new_annotation
        )
        return data



class EncodeAF3TokenLevelFeatures(Transform):

    def __init__(self, sequence_encoding: AF3SequenceEncoding, encode_residues_to: int = None):
        self.sequence_encoding = sequence_encoding
        self.encode_residues_to = encode_residues_to  # for spoofing the restype

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(
            data,
            [
                "atomize",
                "pn_unit_iid",
                "chain_entity",
                "res_name",
                "within_chain_res_idx",
            ],
        )

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array = data["atom_array"]

        # ... get token-level array
        token_starts = get_token_starts(atom_array)
        token_level_array = atom_array[token_starts]

        # ... identifier tokens
        # ... (residue)
        residue_index = token_level_array.within_chain_res_idx
        # ... (token)
        token_index = np.arange(len(token_starts))
        # ... (chain instance)
        asym_name, asym_id = np.unique(token_level_array.pn_unit_iid, return_inverse=True)
        # ... (chain entity)
        entity_name, entity_id = np.unique(token_level_array.pn_unit_entity, return_inverse=True)
        # ... (within chain entity)
        sym_name, sym_id = get_within_entity_idx(token_level_array, level="pn_unit")

        # ... molecule type
        _aa_like_res_names = self.sequence_encoding.all_res_names[self.sequence_encoding.is_aa_like]
        is_protein = np.isin(token_level_array.res_name, _aa_like_res_names)

        _rna_like_res_names = self.sequence_encoding.all_res_names[self.sequence_encoding.is_rna_like]
        is_rna = np.isin(token_level_array.res_name, _rna_like_res_names)

        _dna_like_res_names = self.sequence_encoding.all_res_names[self.sequence_encoding.is_dna_like]
        is_dna = np.isin(token_level_array.res_name, _dna_like_res_names)

        is_ligand = ~(is_protein | is_rna | is_dna)

        # ... sequence tokens
        res_names = token_level_array.res_name
        if self.encode_residues_to is not None:
            is_masked = ~token_level_array.token_has_sequence
            res_names[is_masked] = np.full(np.sum(is_masked), self.encode_residues_to, dtype=res_names.dtype)

        restype = self.sequence_encoding.encode(res_names)
        data["encoded"] = {"seq": restype}  # For msa's
        restype = F.one_hot(torch.tensor(restype), num_classes=self.sequence_encoding.n_tokens).numpy()

        # ... add to data dict
        if "feats" not in data:
            data["feats"] = {}
        if "feat_metadata" not in data:
            data["feat_metadata"] = {}

        # ... add to data dict
        data["feats"] |= {
            "residue_index": residue_index,  # (N_tokens) (int)
            "token_index": token_index,  # (N_tokens) (int)
            "asym_id": asym_id,  # (N_tokens) (int)
            "entity_id": entity_id,  # (N_tokens) (int)
            "sym_id": sym_id,  # (N_tokens) (int)
            "restype": restype,  # (N_tokens, 32) (float, one-hot)
            "is_protein": is_protein,  # (N_tokens) (bool)
            "is_rna": is_rna,  # (N_tokens) (bool)
            "is_dna": is_dna,  # (N_tokens) (bool)
            "is_ligand": is_ligand,  # (N_tokens) (bool)
        }
        data["feat_metadata"] |= {
            "asym_name": asym_name,  # (N_asyms)
            "entity_name": entity_name,  # (N_entities)
            "sym_name": sym_name,  # (N_entities)
        }

        return data