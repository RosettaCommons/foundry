"""
Design Transforms for the Atom14 pipeline
"""

from typing import Any, Dict

import biotite.structure as struc
import numpy as np
import torch
import torch.nn.functional as F
from atomworks.constants import (
    ELEMENT_NAME_TO_ATOMIC_NUMBER,
)
from atomworks.enums import GroundTruthConformerPolicy
from atomworks.io.utils.selection import get_residue_starts
from atomworks.ml.transforms._checks import (
    check_contains_keys,
    check_is_instance,
)
from atomworks.ml.transforms.af3_reference_molecule import (
    _encode_atom_names_like_af3,
    get_af3_reference_molecule_features,
)
from atomworks.ml.transforms.base import (
    Transform,
)
from atomworks.ml.utils.geometry import (
    masked_center,
    random_rigid_augmentation,
)
from atomworks.ml.utils.token import (
    apply_token_wise,
    get_token_starts,
)
from biotite.structure import AtomArray
from rfd3.constants import VIRTUAL_ATOM_ELEMENT_NAME
from rfd3.transforms.conditioning_base import (
    UnindexFlaggedTokens,
    get_motif_features,
)
from rfd3.transforms.rasa import discretize_rasa
from rfd3.transforms.util_transforms import (
    AssignTypes,
    add_backbone_and_sidechain_annotations,
    get_af3_token_representative_masks,
)
from rfd3.transforms.virtual_atoms import PadTokensWithVirtualAtoms

from foundry.utils.ddp import RankedLogger  # noqa

#####################################################################################################
# Other design transforms
#####################################################################################################

ranked_logger = RankedLogger(__name__, rank_zero_only=True)


class SubsampleToTypes(Transform):
    """
    Remove all types not specified as allowed_types
    Possible allowed types:
        - is_protein
        - is_ligand
        - is_dna
        - is_rna
    """

    requires_previous_transforms = [AssignTypes]

    def __init__(
        self,
        allowed_types: list | str = ["is_protein"],
    ):
        self.allowed_types = allowed_types
        if not self.allowed_types == "ALL":
            for k in allowed_types:
                if not k.startswith("is_"):
                    raise ValueError(f"Allowed types must start with 'is_', got {k}")

    def check_input(self, data: dict):
        check_contains_keys(data, ["atom_array"])

    def forward(self, data):
        atom_array = data["atom_array"]

        # ... Subsampling
        if not self.allowed_types == "ALL":
            is_allowed = np.zeros_like(atom_array.is_protein, dtype=bool)
            for allowed_type in self.allowed_types:
                is_allowed = np.logical_or(
                    np.asarray(is_allowed, dtype=bool),
                    np.asarray(
                        atom_array.get_annotation(allowed_type), dtype=bool
                    ).copy(),
                )
            atom_array = atom_array[is_allowed]

        # ... Assert any protein remains
        if atom_array.array_length() == 0:
            raise ValueError(
                "No tokens found in the atom array! Example ID: {}".format(
                    data.get("example_id", "unknown")
                )
            )

        if atom_array.is_protein.sum() == 0:
            raise ValueError(
                "No protein atoms found in the atom array. Example ID: {}".format(
                    data.get("example_id", "unknown")
                )
            )

        data["atom_array"] = atom_array
        return data


class CreateDesignReferenceFeatures(Transform):
    """
    Traditional AF3 will create a bunch of reference features based on the sequence and molecular identity.
    For our design, we do not have access to sequence so these features are useless

    However, this is a great place to add atom-level features as explicit conditioning or implicit
    classifier free guidance.

    Reduces time to process from ~0.5 to ~0.1 s on avg.
    """

    requires_previous_transforms = [UnindexFlaggedTokens, AssignTypes]

    def __init__(
        self,
        generate_conformers,
        generate_conformers_for_non_protein_only,
        provide_reference_conformer_when_unmasked,
        ground_truth_conformer_policy,
        provide_elements_for_unindexed_components,
        use_element_for_atom_names_of_atomized_tokens,
        **kwargs,
    ):
        self.generate_conformers = generate_conformers
        self.generate_conformers_for_non_protein_only = (
            generate_conformers_for_non_protein_only
        )
        self.provide_reference_conformer_when_unmasked = (
            provide_reference_conformer_when_unmasked
        )
        if provide_reference_conformer_when_unmasked:
            self.ground_truth_conformer_policy = GroundTruthConformerPolicy[
                ground_truth_conformer_policy
            ]
        else:
            self.ground_truth_conformer_policy = GroundTruthConformerPolicy.IGNORE

        self.provide_elements_for_unindexed_components = (
            provide_elements_for_unindexed_components
        )

        self.conformer_generation_kwargs = {
            "conformer_generation_timeout": 2.0,
            "use_element_for_atom_names_of_atomized_tokens": use_element_for_atom_names_of_atomized_tokens,
        } | kwargs

    def check_input(self, data: dict):
        check_contains_keys(data, ["atom_array"])

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        I = atom_array.array_length()
        token_starts = get_token_starts(atom_array)
        token_level_array = atom_array[token_starts]
        L = token_level_array.array_length()

        # ... Set up default reference features
        ref_pos = np.zeros_like(atom_array.coord, dtype=np.float32)
        ref_pos[~atom_array.is_motif_atom_with_fixed_coord, :] = 0
        ref_mask = np.zeros((I,), dtype=bool)
        ref_charge = np.zeros((I,), dtype=np.int8)
        ref_pos_is_ground_truth = np.zeros((I,), dtype=bool)

        # ... For elements provide only the elements for unindexed components
        ref_element = np.zeros((I,), dtype=np.int64)
        # if self.provide_elements_for_unindexed_components:
        #    ref_element[atom_array.is_motif_atom_unindexed] = atom_array.atomic_number[
        #        atom_array.is_motif_atom_unindexed
        #    ]

        # ... For atom names, provide all (spoofed) names in the atom array
        ref_atom_name_chars = _encode_atom_names_like_af3(atom_array.atom_name)
        _res_start_ends = get_residue_starts(atom_array, add_exclusive_stop=True)
        _res_starts, _res_ends = _res_start_ends[:-1], _res_start_ends[1:]
        ref_space_uid = struc.segments.spread_segment_wise(
            _res_start_ends, np.arange(len(_res_starts), dtype=np.int64)
        )

        o = get_motif_features(atom_array)
        is_motif_atom, is_motif_token = o["is_motif_atom"], o["is_motif_token"]
        is_motif_token = is_motif_token[token_starts]

        # ... Add a flag for atoms with zero occupancy
        has_zero_occupancy = atom_array.occupancy == 0.0
        if data["is_inference"] and has_zero_occupancy.any():
            ranked_logger.warning(
                "Found non-zero occupancy in input, setting occupancy to 1"
            )
            has_zero_occupancy = np.full_like(has_zero_occupancy, False)

        # ... Token features for token type;
        # [1, 0, 0]: non-motif
        # [0, 1, 0]: indexed motif
        # [0, 0, 1]: unindexed motif
        is_motif_token_unindexed = atom_array.is_motif_atom_unindexed[token_starts]
        motif_token_class = np.zeros((L,), dtype=np.int8)
        motif_token_class[is_motif_token] = 1
        motif_token_class[is_motif_token_unindexed] = 2
        motif_token_type = np.eye(3, dtype=np.int8)[
            motif_token_class
        ]  # one-hot, (L, 3)

        # ... Provide GT as reference coordinates even when unfixed
        motif_pos = np.nan_to_num(atom_array.coord.copy())
        motif_pos = motif_pos * (is_motif_atom[..., None])

        # ... Create reference features for unmasked subset (where we are allowed to use gt)
        has_sequence = (
            atom_array.is_motif_atom_with_fixed_seq
            & ~atom_array.is_motif_atom_unindexed
        )  # (n_atoms,)

        if self.generate_conformers_for_non_protein_only:
            has_sequence = has_sequence & ~atom_array.is_protein

        if np.any(has_sequence):
            # Subset atom level
            atom_array_unmasked = atom_array[has_sequence]

            # We always want to generate conformers if there are ligand atoms that are diffused
            if (
                self.generate_conformers
                or np.logical_and(
                    atom_array_unmasked.is_ligand, ~atom_array_unmasked.is_motif_atom
                ).any()
            ):
                atom_array_unmasked.set_annotation(
                    "ground_truth_conformer_policy",
                    np.full(
                        atom_array_unmasked.array_length(),
                        self.ground_truth_conformer_policy.value,
                    ),
                )

                # Compute the reference features
                # ... Create a copy of atom_array_unmasked and replace the atom_names with gt_atom_names for reference conformer generation
                atom_array_unmasked_with_gt_atom_name = atom_array_unmasked.copy()
                atom_array_unmasked_with_gt_atom_name.atom_name = (
                    atom_array_unmasked_with_gt_atom_name.gt_atom_name
                )
                reference_features_unmasked = get_af3_reference_molecule_features(
                    atom_array_unmasked_with_gt_atom_name,
                    cached_residue_level_data=data["cached_residue_level_data"]
                    if "cached_residue_level_data" in data
                    else None,
                    **self.conformer_generation_kwargs,
                )[0]  ## returns tuple, need to index 0

                ref_atom_name_chars[has_sequence] = reference_features_unmasked[
                    "ref_atom_name_chars"
                ]
                ref_mask[has_sequence] = reference_features_unmasked["ref_mask"]
                ref_element[has_sequence] = reference_features_unmasked["ref_element"]
                ref_charge[has_sequence] = reference_features_unmasked["ref_charge"]
                ref_pos_is_ground_truth[has_sequence] = reference_features_unmasked[
                    "ref_pos_is_ground_truth"
                ]

                # If requested, include the reference conformers for unmasked atoms
                if self.provide_reference_conformer_when_unmasked:
                    ref_pos[has_sequence] = reference_features_unmasked["ref_pos"]
            else:
                # Generate simple features
                ref_charge[has_sequence] = atom_array_unmasked.charge
                ref_element[has_sequence] = (
                    atom_array_unmasked.atomic_number
                    if "atomic_number" in atom_array.get_annotation_categories()
                    else np.vectorize(ELEMENT_NAME_TO_ATOMIC_NUMBER.get)(
                        atom_array.element
                    )
                )

        reference_features = {
            "ref_atom_name_chars": ref_atom_name_chars,  # (n_atoms, 4)
            "ref_pos": ref_pos,  # (n_atoms, 3)
            "ref_mask": ref_mask,  # (n_atoms)
            "ref_element": ref_element,  # (n_atoms)
            "ref_charge": ref_charge,  # (n_atoms)
            "ref_space_uid": ref_space_uid,  # (n_atoms)
            "ref_pos_is_ground_truth": ref_pos_is_ground_truth,  # (n_atoms)
            "has_zero_occupancy": has_zero_occupancy,  # (n_atoms)
            # Conditional masks
            # "ref_is_motif_atom": is_motif_atom,  # (n_atoms, 2)
            # "ref_is_motif_atom_mask": atom_array.is_motif_atom.copy(),  # (n_atoms)
            # "ref_is_motif_token": is_motif_token,  # (n_tokens, 2)
            # "ref_motif_atom_type": motif_atom_type,  # (n_atoms, 3)  # 3 types of atom conditions
            "ref_is_motif_atom_with_fixed_coord": atom_array.is_motif_atom_with_fixed_coord.copy(),  # (n_atoms)
            "ref_is_motif_atom_unindexed": atom_array.is_motif_atom_unindexed.copy(),
            "ref_motif_token_type": motif_token_type,  # (n_tokens, 3)  # 3 types of token
            "motif_pos": motif_pos,  # (n_atoms, 3)  # GT pos for motif atoms
        }

        # TEMPORARY HACK TO CREATE MOTIF FEATURES AGAIN
        # f = get_motif_features(atom_array)
        # token_starts = get_token_starts(atom_array)
        # # Annots
        # atom_array = data["atom_array"]
        # atom_array.set_annotation("is_motif_atom", f["is_motif_atom"])
        # atom_array.set_annotation("is_motif_token", f["is_motif_token"])
        # data["atom_array"] = atom_array
        # # Ref feats
        # motif_atom_class = np.zeros((I,), dtype=np.int8)
        # motif_atom_class[atom_array.is_motif_atom] = 1
        # motif_atom_class[atom_array.is_motif_atom_unindexed] = 2
        # motif_atom_type = np.eye(3, dtype=np.int8)[motif_atom_class]  # one-hot, (I, 3)
        # is_motif_atom = torch.nn.functional.one_hot(
        #     torch.from_numpy(atom_array.is_motif_atom).long(), num_classes=2
        # ).numpy()
        # is_motif_token = torch.nn.functional.one_hot(
        #     torch.from_numpy(atom_array.is_motif_token[token_starts]).long(),
        #     num_classes=2,
        # ).numpy()
        # reference_features["ref_motif_atom_type"] = motif_atom_type
        # reference_features["ref_is_motif_atom"] = is_motif_atom
        # reference_features["ref_is_motif_atom_mask"] = atom_array.is_motif_atom.copy()
        # reference_features["ref_is_motif_token"] = is_motif_token
        # reference_features["is_motif_atom"] = atom_array.is_motif_atom.astype(
        #     bool
        # ).copy()
        # reference_features["is_motif_token"] = f["is_motif_token"][token_starts]
        # # END TEMPORARY HACK

        if "feats" not in data:
            data["feats"] = {}
        data["feats"].update(reference_features)

        return data


class FeaturizeAtoms(Transform):
    def __init__(self, n_bins=4):
        self.n_bins = n_bins

    def forward(self, data):
        atom_array = data["atom_array"]

        if "feats" not in data:
            data["feats"] = {}

        if (
            data["is_inference"]
            and "rasa_bin" in atom_array.get_annotation_categories()
        ):
            rasa_binned = atom_array.get_annotation("rasa_bin").copy()
        elif "rasa" in atom_array.get_annotation_categories():
            rasa_binned = discretize_rasa(
                atom_array,
                n_bins=self.n_bins - 1,
                keep_protein_motif=data["conditions"]["keep_protein_motif_rasa"],
            )
        else:
            rasa_binned = np.full(
                atom_array.array_length(), self.n_bins - 1, dtype=np.int64
            )
        rasa_oh = F.one_hot(
            torch.from_numpy(rasa_binned).long(), num_classes=self.n_bins
        ).numpy()
        data["feats"]["ref_atomwise_rasa"] = rasa_oh[
            ..., :-1
        ]  # exclude last bin from being fed to the model

        if "active_donor" in atom_array.get_annotation_categories():
            data["feats"]["active_donor"] = torch.tensor(
                np.float64(atom_array.active_donor)
            ).long()
        else:
            data["feats"]["active_donor"] = torch.tensor(
                np.zeros(len(atom_array))
            ).long()

        if "active_acceptor" in atom_array.get_annotation_categories():
            data["feats"]["active_acceptor"] = torch.tensor(
                np.float64(atom_array.active_acceptor)
            ).long()
        else:
            data["feats"]["active_acceptor"] = torch.tensor(
                np.zeros(len(atom_array))
            ).long()

        return data


class AddIsXFeats(Transform):
    """
    Adds boolean masks to the atom array based on the sequence type

    Assigned types to atom array (X):
        - is_backbone
        - is_sidechain
        - is_virtual
        - is_central
        - is_ca
    Xs only returned as features (requires previous assignment):
        - is_motif_atom_with_fixed_coord
        - is_motif_atom_unindexed
        - is_motif_atom_with_fixed_seq
    """

    requires_previous_transforms = [
        AssignTypes,
        PadTokensWithVirtualAtoms,
        "UnindexFlaggedTokens",
    ]

    def __init__(
        self,
        X,
        central_atom,
        extra_atom_level_feats: list[str] = [],
        extra_token_level_feats: list[str] = [],
    ):
        self.X = X
        self.central_atom = central_atom
        self.update_atom_array = False
        self.extra_atom_level_feats = extra_atom_level_feats
        self.extra_token_level_feats = extra_token_level_feats

    def check_input(self, data):
        check_contains_keys(data, ["atom_array", "feats"])

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        atom_array = add_backbone_and_sidechain_annotations(atom_array)
        token_level_array = atom_array[get_token_starts(atom_array)]
        _token_rep_mask = get_af3_token_representative_masks(
            atom_array, central_atom=self.central_atom
        )
        _token_rep_idxs = np.where(_token_rep_mask)[0]

        # ... Basic features
        if "is_backbone" in self.X:
            is_backbone = data["atom_array"].get_annotation("is_backbone")
            data["feats"]["is_backbone"] = torch.from_numpy(is_backbone).to(
                dtype=torch.bool
            )

        if "is_sidechain" in self.X:
            is_sidechain = data["atom_array"].get_annotation("is_sidechain")
            data["feats"]["is_sidechain"] = torch.from_numpy(is_sidechain).to(
                dtype=torch.bool
            )

        # Virtual atom feats
        if "is_virtual" in self.X:
            data["feats"]["is_virtual"] = (
                atom_array.element == VIRTUAL_ATOM_ELEMENT_NAME
            )

        for x in [
            "is_motif_atom_with_fixed_coord",
            "is_motif_atom_with_fixed_seq",
            "is_motif_atom_unindexed",
        ]:
            if x not in self.X:
                continue
            if "atom" in x:
                mask = atom_array.get_annotation(x).copy().astype(bool)
            else:
                mask = token_level_array.get_annotation(x).copy().astype(bool)
            data["feats"][x] = mask

        if "is_motif_token_with_fully_fixed_coord" in self.X:
            mask = apply_token_wise(
                atom_array,
                atom_array.is_motif_atom_with_fixed_coord.astype(bool),
                function=lambda x: np.all(x, axis=-1),
            )
            data["feats"]["is_motif_token_with_fully_fixed_coord"] = mask

        # ... Central and CA
        if "is_central" in self.X:
            data["feats"]["is_central"] = _token_rep_mask

        if "is_ca" in self.X:
            # Split into components to handle separately
            atom_array_indexed = atom_array[~atom_array.is_motif_atom_unindexed]
            _token_rep_mask_indexed = get_af3_token_representative_masks(
                atom_array_indexed, central_atom="CA"
            )
            if atom_array.is_motif_atom_unindexed.any():
                atom_array_unindexed = atom_array[atom_array.is_motif_atom_unindexed]

                # Ensure is_ca represents one and the first atom only for unindexed tokens
                def first_nonzero(n):
                    assert n > 0
                    x = np.zeros(n, dtype=bool)
                    x[0] = 1
                    return x

                starts = get_token_starts(atom_array_unindexed, add_exclusive_stop=True)
                _token_rep_mask_unindexed = np.concatenate(
                    [
                        first_nonzero(end - start)
                        for start, end in zip(starts[:-1], starts[1:])
                    ]
                )
                _token_rep_mask = np.concatenate(
                    [
                        _token_rep_mask_indexed,
                        _token_rep_mask_unindexed,
                    ],
                    axis=0,
                )
            else:
                _token_rep_mask = _token_rep_mask_indexed
            data["feats"]["is_ca"] = _token_rep_mask

        return data


PPI_PERTURB_SCALE = 2.0
PPI_PERTURB_COM_SCALE = 1.5


class MotifCenterRandomAugmentation(Transform):
    requires_previous_transforms = ["BatchStructuresForDiffusionNoising"]

    def __init__(
        self,
        batch_size,
        sigma_perturb,
        center_option,
    ):
        """
        Randomly augments the coordinates of the motif center for diffusion training.
        During inference, this behaviour is handled by the sampler at every step
        """

        self.batch_size = batch_size
        self.sigma_perturb = sigma_perturb
        self.center_option = center_option

    def check_input(self, data: dict):
        pass

    def forward(self, data):
        """
        Applies CenterRandomAugmentation

        And supplies the same rotated ground-truth coordinates as the input feature
        """
        if data["is_inference"]:
            return data  # ori token behaviour set when creating atom array & in sampler

        xyz = data["coord_atom_lvl_to_be_noised"]  # (batch_size, n_atoms, 3)
        mask_atom_lvl = data["ground_truth"]["mask_atom_lvl"]
        mask_atom_lvl = (
            mask_atom_lvl & ~data["feats"]["is_motif_atom_unindexed"]
        )  # Avoid double weighting

        # Handle the diffferent centering options
        is_motif_atom_with_fixed_coord = torch.tensor(
            data["atom_array"].is_motif_atom_with_fixed_coord, dtype=torch.bool
        )
        if torch.any(is_motif_atom_with_fixed_coord):
            if self.center_option == "motif":
                center_mask = is_motif_atom_with_fixed_coord.clone()
            elif self.center_option == "diffuse":
                center_mask = (~is_motif_atom_with_fixed_coord).clone()
            else:
                center_mask = torch.ones(mask_atom_lvl.shape, dtype=torch.bool)
        else:
            center_mask = torch.ones(mask_atom_lvl.shape, dtype=torch.bool)

        mask_atom_lvl = mask_atom_lvl & center_mask
        mask_atom_lvl_expanded = mask_atom_lvl.expand(xyz.shape[0], -1)

        # Masked center during training (nb not motif mask - just non-zero occupancy)
        xyz = masked_center(xyz, mask_atom_lvl_expanded)

        # Random offset
        sigma_perturb = self.sigma_perturb
        if data["sampled_condition_name"] == "ppi":
            sigma_perturb = sigma_perturb * PPI_PERTURB_SCALE

        xyz = (
            xyz
            + torch.randn(
                (
                    self.batch_size,
                    3,
                ),
                device=xyz.device,
            )[:, None, :]
            * self.sigma_perturb
        )

        # Apply random spin
        xyz = random_rigid_augmentation(xyz, batch_size=self.batch_size, s=0)
        data["coord_atom_lvl_to_be_noised"] = xyz

        return data


class AugmentNoise(Transform):
    requires_previous_transforms = ["SampleEDMNoise", "AddIsXFeats"]

    def __init__(
        self,
        sigma_perturb_com,
        batch_size,
    ):
        """
        Scaled perturbation to the offset between motif and diffused region based on time
        """
        self.sigma_perturb_com = sigma_perturb_com
        self.batch_size = batch_size

    def check_input(self, data: dict):
        check_contains_keys(data, ["noise", "coord_atom_lvl_to_be_noised"])
        check_contains_keys(data, ["feats"])

    def forward(self, data: dict) -> dict:
        is_motif_atom_with_fixed_coord = data["feats"]["is_motif_atom_with_fixed_coord"]
        device = data["coord_atom_lvl_to_be_noised"].device
        data["noise"][..., is_motif_atom_with_fixed_coord, :] = 0.0

        # Add perturbation to the centre-of-mass
        if not data["is_inference"] or not is_motif_atom_with_fixed_coord.any():
            sigma_perturb_com = self.sigma_perturb_com
            if data["sampled_condition_name"] == "ppi":
                sigma_perturb_com = sigma_perturb_com * PPI_PERTURB_COM_SCALE
            eps = torch.randn(self.batch_size, 3, device=device) * sigma_perturb_com
            maxt = 38
            eps = eps * torch.clip((data["t"][:, None] / maxt) ** 3, min=0, max=1)
            data["noise"][..., ~is_motif_atom_with_fixed_coord, :] += eps[:, None, :]

        # ... Zero out noise going into motif
        data["noise"][..., is_motif_atom_with_fixed_coord, :] = 0.0

        assert data["coord_atom_lvl_to_be_noised"].shape == data["noise"].shape
        return data


class AddGroundTruthSequence(Transform):
    """
    Adds token level sequence to the ground truth.

    Adds:
        ['ground_truth']['seq_token_lvl'] (torch.Tensor): The ground truth token level sequence [L,]
    """

    def __init__(self, sequence_encoding):
        self.sequence_encoding = sequence_encoding

    def check_input(self, data):
        check_contains_keys(data, ["atom_array"])

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        token_starts = get_token_starts(atom_array)
        res_names = atom_array.res_name[token_starts]
        restype = self.sequence_encoding.encode(res_names)

        if "ground_truth" not in data:
            data["ground_truth"] = {}

        data["ground_truth"]["sequence_gt_I"] = torch.from_numpy(restype)
        data["ground_truth"]["sequence_valid_mask"] = torch.from_numpy(
            ~np.isin(res_names, ["UNK", "X", "DX", "<G>"])
        )

        return data


class AddAdditional1dFeaturesToFeats(Transform):
    """
    Adds any net.token_initializer.token_1d_features and net.diffusion_module.diffusion_atom_encoder.atom_1d_features present in the atomarray but not in data['feats'] to data['feats']
    Args:
        - autofill_zeros_if_not_present_in_atomarray: self explanatory
        - token_1d_features: List of single-item dictionaries, corresponding to feature_name: n_feature_dims. Should be hydra interpolated from
            net.token_initializer.token_1d_features
        - atom_1d_features: List of single-item dictionaries, corresponding to feature_name: n_feature_dims. Should be hydra interpolated from
            net.diffusion_module.diffusion_atom_encoder.atom_1d_features
    """

    incompatible_previous_transforms = ["AddAdditional1dFeaturesToFeats"]

    def __init__(
        self,
        token_1d_features,
        atom_1d_features,
        autofill_zeros_if_not_present_in_atomarray=False,
    ):
        self.autofill = autofill_zeros_if_not_present_in_atomarray
        self.token_1d_features = token_1d_features
        self.atom_1d_features = atom_1d_features

    def check_input(self, data) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)

    def generate_feature(self, feature_name, n_dims, data, feature_type):
        if feature_name in data["feats"].keys():
            return data
        elif feature_name in data["atom_array"].get_annotation_categories():
            feature_array = torch.tensor(
                data["atom_array"].get_annotation(feature_name)
            ).float()

            # ensure that feature_array is a 2d array with second dim n_dims
            if len(feature_array.shape) == 1 and n_dims == 1:
                feature_array = feature_array.unsqueeze(1)
            elif len(feature_array.shape) != 2:
                raise ValueError(
                    f"{feature_type} 1d_feature `{feature_name}` must be a 2d array, got {len(feature_array.shape)}d."
                )
            if feature_array.shape[1] != n_dims:
                raise ValueError(
                    f"{feature_type} 1d_feature `{feature_name}` dimensions in atomarray ({feature_array.shape[-1]}) does not match dimension declared in config, ({n_dims})"
                )

        elif self.autofill:
            feature_array = torch.zeros((len(data["atom_array"]), n_dims)).float()

        # not in data['feats'], not in atomarray, and autofill is off
        else:
            raise ValueError(
                f"{feature_type} 1d_feature `{feature_name}` is not present in atomarray, and autofill is False"
            )

        if feature_type == "token":
            feature_array = torch.tensor(
                apply_token_wise(
                    data["atom_array"], feature_array.numpy(), np.mean, axis=0
                )
            ).float()

        data["feats"][feature_name] = feature_array
        return data

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Checks if the 1d_features are present in data['feats']. If not present, adds them from the atomarray.
        If annotation is not present in atomarray, either autofills the feature with 0s or throws an error
        """
        if "feats" not in data.keys():
            data["feats"] = {}

        for feature_name, n_dims in self.token_1d_features.items():
            data = self.generate_feature(feature_name, n_dims, data, "token")

        for feature_name, n_dims in self.atom_1d_features.items():
            data = self.generate_feature(feature_name, n_dims, data, "atom")

        return data


class FeaturizepLDDT(Transform):
    """
    Provides:
        0 for unknown pLDDT
       +1 for high pLDDT
       -1 for low pLDDT
    """

    def __init__(
        self,
        skip,
    ):
        self.skip = skip
        self.bsplit = 80  # Threshold for splitting pLDDT into high and low

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        token_starts = get_token_starts(atom_array)
        I = len(token_starts)
        zeros = np.zeros(I, dtype=int)
        if data["is_inference"]:
            if "ref_plddt" not in atom_array.get_annotation_categories():
                ref_plddt = zeros
            else:
                ref_plddt = atom_array.get_annotation("ref_plddt")[token_starts]
        elif (
            self.skip
            or "b_factor" not in atom_array.get_annotation_categories()
            or not data["conditions"]["featurize_plddt"]
        ):
            ref_plddt = zeros
        else:
            plddt = atom_array.get_annotation("b_factor")
            mean_plddt = np.mean(plddt)
            ref_plddt = zeros + (1 if mean_plddt >= self.bsplit else -1)

        # Provide only non-zero values for diffused tokens
        ref_plddt = (
            ~(get_motif_features(atom_array)["is_motif_token"][token_starts])
            * ref_plddt
        )
        data["feats"]["ref_plddt"] = ref_plddt
        return data
