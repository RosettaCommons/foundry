# see atomworks.ml.ransforms.feature_aggregation
import time
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from atomworks.constants import STANDARD_AA
from atomworks.enums import ChainTypeInfo
from atomworks.io.utils.sequence import (
    is_purine,
    is_pyrimidine,
)
from atomworks.ml.encoding_definitions import AF3SequenceEncoding
from atomworks.ml.transforms._checks import (
    check_atom_array_annotation,
    check_contains_keys,
    check_is_instance,
)
from atomworks.ml.transforms.atom_array import get_within_entity_idx
from atomworks.ml.transforms.base import Transform
from atomworks.ml.utils.token import (
    get_token_count,
    get_token_starts,
    is_glycine,
    is_protein_unknown,
    is_standard_aa_not_glycine,
    is_unknown_nucleotide,
    spread_token_wise,
)
from biotite.structure import AtomArray

af3_sequence_encoding = AF3SequenceEncoding()


def assert_single_representative(token, central_atom="CB"):
    mask = get_af3_token_representative_masks(token, central_atom=central_atom)
    assert (
        np.sum(mask) == 1
    ), f"No representative atom (CB) found. mask: {mask}\nToken: {token}"


def assert_single_token(token):
    assert get_token_count(token) == 1, f"Token is not a single token: {token}"
    assert_single_representative(token)


def add_representative_atom(token, central_atom="CB"):
    if get_af3_token_representative_masks(token, central_atom=central_atom).sum() == 1:
        return token
    length = token.array_length()
    token.atomize = np.array([True] + [False] * (length - 1), dtype=bool)
    assert_single_representative(token)
    return token


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


class IPDB(Transform):
    def forward(self, data):
        aa = data["atom_array"]  # noqa
        import ipdb

        ipdb.set_trace()
        return data


sequence_encoding = AF3SequenceEncoding()

_aa_like_res_names = sequence_encoding.all_res_names[sequence_encoding.is_aa_like]
_rna_like_res_names = sequence_encoding.all_res_names[sequence_encoding.is_rna_like]
_dna_like_res_names = sequence_encoding.all_res_names[sequence_encoding.is_dna_like]


class AssignTypes(Transform):
    """
    Assigns types to the atoms in the atom array using af3 sequence encoding scheme.
    """

    def check_input(self, data):
        assert "atom_array" in data, "Input data must contain 'atom_array'."

    def forward(self, data):
        data["atom_array"] = assign_types_(data["atom_array"])
        return data


def assign_types_(atom_array):
    token_starts = get_token_starts(atom_array)
    res_names = atom_array[token_starts].res_name
    token_id = np.arange(get_token_count(atom_array), dtype=np.uint32)  # [n_tokens]
    atom_to_token_map = spread_token_wise(atom_array, token_id)

    is_protein = np.isin(res_names, _aa_like_res_names).astype(bool)
    is_residue = np.isin(res_names, STANDARD_AA).astype(bool)
    is_rna = np.isin(res_names, _rna_like_res_names).astype(bool)
    is_dna = np.isin(res_names, _dna_like_res_names).astype(bool)
    is_ligand = ~(is_protein | is_rna | is_dna).astype(bool)

    # Set annotations
    atom_array.set_annotation("is_protein", is_protein[atom_to_token_map])
    atom_array.set_annotation("is_rna", is_rna[atom_to_token_map])
    atom_array.set_annotation("is_dna", is_dna[atom_to_token_map])
    atom_array.set_annotation("is_ligand", is_ligand[atom_to_token_map])
    atom_array.set_annotation("is_residue", is_residue[atom_to_token_map])

    return atom_array


class AggregateFeaturesLikeAF3WithoutMSA(Transform):
    """
    Exactly like AggregateFeaturesLikeAF3 but without MSAs

    Removed comments for readability, no additional code is in this function, just removed msa parts
    """

    requires_previous_transforms = [
        "AtomizeByCCDName",
        "EncodeAF3TokenLevelFeatures",
        "AddAF3TokenBondFeatures",
        "UnindexFlaggedTokens",
    ]
    incompatible_previous_transforms = [
        "AggregateFeaturesLikeAF3",
        "AggregateFeaturesLikeAF3WithoutMSA",
    ]

    def check_input(self, data) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(
            data, ["coord_to_be_noised", "chain_iid", "occupancy"]
        )

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

        data["feats"]["ref_atom_name_chars"] = F.one_hot(
            data["feats"]["ref_atom_name_chars"].long(), num_classes=64
        ).float()
        data["feats"]["ref_element"] = F.one_hot(
            data["feats"]["ref_element"].long(), num_classes=128
        ).float()
        data["feats"]["ref_pos"] = torch.nan_to_num(data["feats"]["ref_pos"], nan=0.0)

        # Process ground truth structure
        atom_array = data["atom_array"]

        coord_atom_lvl = atom_array.coord
        mask_atom_lvl = atom_array.occupancy > 0.0
        token_starts = get_token_starts(atom_array)
        token_level_array = atom_array[token_starts]
        chain_iid_token_lvl = token_level_array.chain_iid
        if "ground_truth" not in data:
            data["ground_truth"] = {}

        data["ground_truth"].update(
            {
                "coord_atom_lvl": torch.tensor(coord_atom_lvl),  # [n_atoms, 3]
                "mask_atom_lvl": torch.tensor(mask_atom_lvl),  # [n_atoms]
                "chain_iid_token_lvl": chain_iid_token_lvl,  # numpy.ndarray of strings with shape (n_tokens,)
                "is_original_unindexed_token": torch.from_numpy(
                    data["ground_truth"].get(
                        "is_original_unindexed_token",
                        np.zeros(len(token_starts), dtype=bool),
                    )
                ).bool(),  # [n_tokens]
            }
        )
        data["coord_atom_lvl_to_be_noised"] = torch.tensor(
            atom_array.coord_to_be_noised
        )

        # Remove any token bond features relating to unindexed tokens
        if "token_bonds" in data["feats"]:
            token_bonds = data["feats"]["token_bonds"]
            mask = data["feats"]["is_motif_token_unindexed"]

            # tokens bonded to unindexed & unindexed bonded to tokens
            token_bonds[mask, :] = False
            token_bonds[:, mask] = False

        # Add partial t during inference
        if "partial_t" in atom_array.get_annotation_categories():
            assert data["is_inference"], "Partial diffusion only inference!"
            data["feats"]["partial_t"] = torch.from_numpy(
                atom_array.get_annotation("partial_t")
            )

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


# from atomworks.ml.utils.token import get_af3_token_representative_masks
def get_af3_token_representative_masks(
    atom_array: AtomArray, central_atom: str = "CA"
) -> np.ndarray:
    pyrimidine_representative_atom = is_pyrimidine(atom_array.res_name) & (
        atom_array.atom_name == "C2"
    )
    purine_representative_atom = is_purine(atom_array.res_name) & (
        atom_array.atom_name == "C4"
    )
    unknown_na_representative_atom = is_unknown_nucleotide(atom_array.res_name) & (
        atom_array.atom_name == "C4"
    )

    glycine_representative_atom = is_glycine(atom_array.res_name) & (
        atom_array.atom_name == "CA"
    )
    protein_residue_not_glycine_representative_atom = is_standard_aa_not_glycine(
        atom_array.res_name
    ) & (
        atom_array.atom_name == central_atom  # only change
    )
    unknown_protein_residue_representative_atom = (
        is_protein_unknown(atom_array.res_name)
    ) & (atom_array.atom_name == "CA")
    atoms = atom_array.atomize

    _token_rep_mask = (
        pyrimidine_representative_atom
        | purine_representative_atom
        | unknown_na_representative_atom
        | glycine_representative_atom
        | protein_residue_not_glycine_representative_atom
        | unknown_protein_residue_representative_atom
        | atoms
    )
    return _token_rep_mask


class RemoveTokensWithoutCorrespondingCentralAtom(Transform):
    """
    Remove tokens with missing central atoms.
    """

    def __init__(self, central_atom: str = "CA"):
        self.central_atom = central_atom

    def check_input(self, data):
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["atom_name", "res_name"])

    def forward(self, data):
        central_atom = self.central_atom
        atom_array = data["atom_array"]
        pyrimidine_mask = is_pyrimidine(atom_array.res_name)
        purine_mask = is_purine(atom_array.res_name)
        unknown_na_mask = is_unknown_nucleotide(atom_array.res_name)
        glycine_mask = is_glycine(atom_array.res_name)
        aa_not_glycine_mask = is_standard_aa_not_glycine(atom_array.res_name)
        unknown_aa_mask = is_protein_unknown(atom_array.res_name)

        anything_else_mask = ~(
            pyrimidine_mask
            | purine_mask
            | unknown_na_mask
            | glycine_mask
            | aa_not_glycine_mask
            | unknown_aa_mask
        )

        def _get_if_central_atom_present_mask(atom_array, case_mask, central_atom):
            token_starts = get_token_starts(atom_array[case_mask])
            central_atom_mask = atom_array[case_mask].atom_name == central_atom
            if len(token_starts) == central_atom_mask.sum():
                ## all tokens have central atom, *vast majority*
                return case_mask
            else:
                ## find the missing ones, *very rare*
                out_mask = case_mask
                all_token_starts = get_token_starts(atom_array)
                token_start_mask = case_mask[all_token_starts]
                case_token_starts = all_token_starts[token_start_mask]

                for item in case_token_starts:
                    res_start = item
                    idx = all_token_starts.tolist().index(res_start)
                    res_mask = np.bool_(np.zeros(len(atom_array)))
                    if idx == len(all_token_starts) - 1:
                        res_mask[res_start:] = True
                    else:
                        res_end = all_token_starts[idx + 1]
                        res_mask[res_start:res_end] = True
                    res_array = atom_array[res_mask]

                    # remove if central atom not present
                    if (res_array.atom_name == central_atom).sum() == 0:
                        out_mask = out_mask & ~res_mask
                return out_mask

        keep_mask = (
            _get_if_central_atom_present_mask(atom_array, pyrimidine_mask, "C2")
            | _get_if_central_atom_present_mask(atom_array, purine_mask, "C4")
            | _get_if_central_atom_present_mask(atom_array, unknown_na_mask, "C4")
            | _get_if_central_atom_present_mask(atom_array, glycine_mask, "CA")
            | _get_if_central_atom_present_mask(
                atom_array, aa_not_glycine_mask, central_atom
            )
            | _get_if_central_atom_present_mask(atom_array, unknown_aa_mask, "CA")
            | anything_else_mask
        )

        data["atom_array"] = atom_array[keep_mask]
        return data


class EncodeAF3TokenLevelFeatures(Transform):
    def __init__(
        self, sequence_encoding: AF3SequenceEncoding, encode_residues_to: int = None
    ):
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
        asym_name, asym_id = np.unique(
            token_level_array.pn_unit_iid, return_inverse=True
        )
        # ... (chain entity)
        entity_name, entity_id = np.unique(
            token_level_array.pn_unit_entity, return_inverse=True
        )
        # ... (within chain entity)
        sym_name, sym_id = get_within_entity_idx(token_level_array, level="pn_unit")

        # ... molecule type
        _aa_like_res_names = self.sequence_encoding.all_res_names[
            self.sequence_encoding.is_aa_like
        ]
        is_protein = np.isin(token_level_array.res_name, _aa_like_res_names)

        _rna_like_res_names = self.sequence_encoding.all_res_names[
            self.sequence_encoding.is_rna_like
        ]
        is_rna = np.isin(token_level_array.res_name, _rna_like_res_names)

        _dna_like_res_names = self.sequence_encoding.all_res_names[
            self.sequence_encoding.is_dna_like
        ]
        is_dna = np.isin(token_level_array.res_name, _dna_like_res_names)

        is_ligand = ~(is_protein | is_rna | is_dna)

        # Get is_polar features
        polar_restypes = np.array(
            [
                "SER",
                "THR",
                "ASN",
                "GLN",
                "TYR",
                "CYS",
                "HIS",
                "LYS",
                "ARG",
                "ASP",
                "GLU",
            ]
        )
        is_polar = is_protein & np.isin(token_level_array.res_name, polar_restypes)

        # ... sequence tokens
        res_names = token_level_array.res_name
        if self.encode_residues_to is not None:
            is_masked = ~token_level_array.is_motif_atom_with_fixed_seq
            res_names[is_masked] = np.full(
                np.sum(is_masked), self.encode_residues_to, dtype=res_names.dtype
            )

        restype = self.sequence_encoding.encode(res_names)
        data["encoded"] = {"seq": restype}  # For msa's
        restype = F.one_hot(
            torch.tensor(restype), num_classes=self.sequence_encoding.n_tokens
        ).numpy()

        # ... Add termini annotations (n_tok, 2)
        terminus_type = np.zeros(
            (
                len(token_level_array),
                2,
            ),
            dtype=restype.dtype,
        )
        terminus_type[token_level_array.is_C_terminus, 0] = 1
        terminus_type[token_level_array.is_N_terminus, 1] = 1

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
            "terminus_type": terminus_type,  # (N_tokens, 2) (int)
            "is_polar": is_polar,  # (N_tokens) (bool)
        }
        data["feat_metadata"] |= {
            "asym_name": asym_name,  # (N_asyms)
            "entity_name": entity_name,  # (N_entities)
            "sym_name": sym_name,  # (N_entities)
        }

        return data
