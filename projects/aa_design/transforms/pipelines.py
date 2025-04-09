"""
Atom14 Dataloaders
Atom-only version, no sequence/element decoding
"""
import ast
import warnings
from typing import List

import biotite.structure as struc
import numpy as np
import torch
from cifutils.constants import (
    AF3_EXCLUDED_LIGANDS,
    GAP,
    STANDARD_AA,
    STANDARD_DNA,
    STANDARD_RNA,
)
from cifutils.utils.selection import get_residue_starts
from datahub.encoding_definitions import AF3SequenceEncoding
from datahub.transforms._checks import (
    check_contains_keys,
)
from datahub.transforms.af3_reference_molecule import (
    _encode_atom_names_like_af3,
    get_af3_reference_molecule_features,
)
from datahub.transforms.atom_array import (
    AddGlobalAtomIdAnnotation,
    AddGlobalTokenIdAnnotation,
    AddWithinChainInstanceResIdx,
    AddWithinPolyResIdxAnnotation,
    ComputeAtomToTokenMap,
)
from datahub.transforms.atomize import AtomizeByCCDName, FlagNonPolymersForAtomization
from datahub.transforms.base import (
    AddData,
    Compose,
    ConditionalRoute,
    ConvertToTorch,
    Identity,
    RandomRoute,
    SubsetToKeys,
    Transform,
)
from datahub.transforms.bonds import AddAF3TokenBondFeatures
from datahub.transforms.covalent_modifications import (
    FlagAndReassignCovalentModifications,
)
from datahub.transforms.crop import CropContiguousLikeAF3, CropSpatialLikeAF3
from datahub.transforms.diffusion.batch_structures import (
    BatchStructuresForDiffusionNoising,
)
from datahub.transforms.diffusion.edm import SampleEDMNoise
from datahub.transforms.featurize_unresolved_residues import (
    MaskPolymerResiduesWithUnresolvedFrameAtoms,
    PlaceUnresolvedTokenAtomsOnRepresentativeAtom,
    PlaceUnresolvedTokenOnClosestResolvedTokenInSequence,
)
from datahub.transforms.filters import (
    FilterToSpecifiedPNUnits,
    HandleUndesiredResTokens,
    RemoveHydrogens,
    RemoveNucleicAcidTerminalOxygen,
    RemovePolymersWithTooFewResolvedResidues,
    RemoveTerminalOxygen,
    RemoveUnresolvedPNUnits,
)
from datahub.utils.geometry import (
    masked_center,
    random_rigid_augmentation,
)
from datahub.utils.token import (
    get_token_count,
    get_token_starts,
    spread_token_wise,
)

from modelhub.common import exists
from projects.aa_design.transforms.masking import CreateMasks
from projects.aa_design.transforms.util_transforms import (
    AggregateFeaturesLikeAF3WithoutMSA,
    CopyAnnotation,
    EncodeAF3TokenLevelFeatures,
    add_backbone_and_sidechain_annotations,
    get_af3_token_representative_masks,
)

######################################################################################
# Common transforms
######################################################################################
af3_sequence_encoding = AF3SequenceEncoding()

def get_pre_crop_transforms(
    undesired_res_names: list[str] = AF3_EXCLUDED_LIGANDS,
):

    return [
        RemoveHydrogens(),
        FilterToSpecifiedPNUnits(
            extra_info_key_with_pn_unit_iids_to_keep="all_pn_unit_iids_after_processing"
        ),  # Filter to non-clashing PN units
        RemoveTerminalOxygen(),
        RemoveUnresolvedPNUnits(),  # Remove PN units that are unresolved early (and also after cropping)
        RemovePolymersWithTooFewResolvedResidues(min_residues=4),  # Remove polymers with too few resolved residues
        MaskPolymerResiduesWithUnresolvedFrameAtoms(),

        HandleUndesiredResTokens(undesired_res_names),  # e.g., non-standard residues
        FlagAndReassignCovalentModifications(),
        FlagNonPolymersForAtomization(),
        AddGlobalAtomIdAnnotation(),
        AtomizeByCCDName(
            atomize_by_default=True,
            res_names_to_ignore=STANDARD_AA + STANDARD_RNA + STANDARD_DNA,
            move_atomized_part_to_end=False,
            validate_atomize=False,
        ),
        RemoveNucleicAcidTerminalOxygen(),
        AddWithinChainInstanceResIdx(),
        AddWithinPolyResIdxAnnotation(),
    ]

def get_crop_transform(
    crop_size: int = 256,
    crop_center_cutoff_distance: float = 15.0,
    crop_contiguous_probability: float = 0.5,
    crop_spatial_probability: float = 0.5,
    max_atoms_in_crop: int | None = None,
):
    if (crop_contiguous_probability > 0 or crop_spatial_probability > 0):
        assert np.isclose(
            crop_contiguous_probability + crop_spatial_probability, 1.0, atol=1e-6
        ), "Crop probabilities must sum to 1.0"
        assert crop_size > 0, "Crop size must be greater than 0"
        assert crop_center_cutoff_distance > 0, "Crop center cutoff distance must be greater than 0"
    
    cropping_transform = RandomRoute(
        transforms=[
            CropContiguousLikeAF3(
                crop_size=crop_size,
                keep_uncropped_atom_array=True,
                max_atoms_in_crop=max_atoms_in_crop,
            ),
            CropSpatialLikeAF3(
                crop_size=crop_size,
                crop_center_cutoff_distance=crop_center_cutoff_distance,
                keep_uncropped_atom_array=True,
                max_atoms_in_crop=max_atoms_in_crop,
            ),
        ],
        probs=[crop_contiguous_probability, crop_spatial_probability],
    )

    return [
        ConditionalRoute(
        condition_func=lambda data: data.get("is_inference", False),
        transform_map={
            True: Identity(),
            False: cropping_transform,
        },
    )]

def get_diffusion_transforms(
    *,
    sigma_data: float,
    diffusion_batch_size: int,
):
    return [
        ComputeAtomToTokenMap(),
        ConvertToTorch(keys=["encoded", "feats"]),

        # Prepare coordinates for noising (without modifying the ground truth)
        # ...add placeholder coordinates for noising
        CopyAnnotation(annotation_to_copy="coord", new_annotation="coord_to_be_noised"),

        # ...handling of unresolved residues (NOTE: best done after inputs are processed.)
        PlaceUnresolvedTokenAtomsOnRepresentativeAtom(annotation_to_update="coord_to_be_noised"),
        PlaceUnresolvedTokenOnClosestResolvedTokenInSequence(annotation_to_update="coord_to_be_noised", annotation_to_copy="coord_to_be_noised"),

        # Feature aggregation 
        AggregateFeaturesLikeAF3WithoutMSA(),

        # ...batching and noise sampling for diffusion
        BatchStructuresForDiffusionNoising(batch_size=diffusion_batch_size),
        SampleEDMNoise(sigma_data=sigma_data, diffusion_batch_size=diffusion_batch_size),
    ]

# Turn off warnings for now
print("Turning off runtime warnings!")
warnings.filterwarnings("ignore", category=RuntimeWarning)

######################################################################################
# Pipelines
######################################################################################

def build_atom14_base_pipeline(  # 1.0540871620178223 s
    *,
    # Training or inference (required)
    is_inference: bool,  # If True, we skip cropping, etc.
    return_atom_array: bool = False,
    
    # Crop params
    allowed_types: List[str] = ['is_protein'],
    crop_size: int = 256,
    crop_center_cutoff_distance: float = 15.0,
    crop_contiguous_probability: float = 0.5,
    crop_spatial_probability: float = 0.5,
    max_atoms_in_crop: int | None = None,
    
    # Training Hypers
    sigma_data: float = 16.0,
    diffusion_batch_size: int = 32,
    seed: int = 42,

    # Design params
    train_masks,
    n_atoms_per_token: int = 14,
    central_atom: str = 'CB',
    ori_token: List[float] = None,
    sigma_perturb: float = 0.0,

    **_,  # dump additional kwargs (e.g. msa stuff)
):
    '''
    All-Atom design pipeline
    '''
    # Preamble
    backbone_token_name=GAP  # <G>

    # Add any data necessary for downstream transforms
    transforms = [
        AddData({
            "is_inference": is_inference,
        })
    ]

    # Pre-crop transforms
    transforms += get_pre_crop_transforms()
    transforms += [
        SubsampleToTypes(sequence_encoding=af3_sequence_encoding, allowed_types=allowed_types),
        # CenterCoordinates(),  # center global frame of protein
    ]
    transforms += get_crop_transform(
        crop_size=crop_size, crop_center_cutoff_distance=crop_center_cutoff_distance,
        crop_contiguous_probability=crop_contiguous_probability, crop_spatial_probability=crop_spatial_probability
    )
    
    # Design Transforms
    transforms += [
        # ... Add global token features (since number of tokens is fixed after cropping)
        AddGlobalTokenIdAnnotation(),

        # ... Create masks (NOTE: Modulates token count, and resets global token id if necessary)
        CreateMasks(
            train_masks=train_masks,
            sequence_encoding=af3_sequence_encoding,
            seed=seed,
        ),

        # ... Virtual atom padding (NOTE: Last transform which modulates atom count)
        PadMaskedResiduesWithAtoms(
            n_atoms_per_token=n_atoms_per_token, 
            atom_to_pad_from=central_atom, 
            mask_atom_names=True,  # ensures atom names are unique and masked (sequence still kept)
            virtual_atom_element_name='X'
        ),  # 0.1 s

        # ... AF3 token level encoding with sequence masking
        EncodeAF3TokenLevelFeatures(
            sequence_encoding=af3_sequence_encoding,
            encode_residues_to=backbone_token_name),

        # ... Atom-level reference features
        CreateDesignReferenceFeatures(),

        # ... Add useful features for losses / metrics
        AddIsXFeats(X = [
            # Basic
            'is_backbone', 
            'is_sidechain', 
            # Virtual atom
            'is_virtual', 
            'is_central', 
            'is_ca',
            # Conditioning
            'is_motif_token', 
            'is_motif_atom', 
            'is_motif_atom_with_fixed_pos',
            'is_motif_atom_without_index'
            ], central_atom=central_atom, 
            virtual_atom_element_name='X'),
        AddAF3TokenBondFeatures(),
        AddGroundTruthSequence(sequence_encoding=af3_sequence_encoding),
    ]

    # EDM-style wrap-up  (no additional features added at this point)
    transforms += get_diffusion_transforms(sigma_data=sigma_data, diffusion_batch_size=diffusion_batch_size)
    
    # ... Random augmentation accounting for motif

    transforms += [
        MotifCenterRandomAugmentation(batch_size=diffusion_batch_size, sigma_perturb=sigma_perturb),
        RemoveNoiseFromUnmaskedAtoms(),
    ]

    # Subset to necessary keys only
    keys_to_keep = [
        "example_id",
        "feats",
        "t",
        "noise",
        "ground_truth",
        "coord_atom_lvl_to_be_noised",
        "symmetry_resolution",
        "extra_info",
    ]
    if not is_inference:
        keys_to_keep.append("sampled_mask_name")
    if return_atom_array:
        keys_to_keep.append("atom_array")
        transforms.append(CleanupMaskedAtomArray())
    transforms.append(SubsetToKeys(keys_to_keep))

    pipeline = Compose(transforms)
    return pipeline


class SubsampleToTypes(Transform):
    def __init__(self, sequence_encoding=af3_sequence_encoding, allowed_types=['is_protein']):
        self.sequence_encoding = sequence_encoding
        self.allowed_types = allowed_types
        for k in allowed_types:
            if not k.startswith('is_'):
                raise ValueError(f"Allowed types must start with 'is_', got {k}")

    def check_input(self, data: dict):
        check_contains_keys(data, ["atom_array"])
    
    def forward(self, data):
        atom_array = data['atom_array']
        token_starts = get_token_starts(atom_array)
        res_names = atom_array[token_starts].res_name
        token_id = np.arange(get_token_count(atom_array), dtype=np.uint32)  # [n_tokens]
        atom_to_token_map = spread_token_wise(atom_array, token_id)
        
        # ... Get data types first
        if not all([(req in atom_array.get_annotation_categories())
            for req in ['is_protein', 'is_rna', 'is_dna', 'is_ligand']]):
        
            _aa_like_res_names = self.sequence_encoding.all_res_names[self.sequence_encoding.is_aa_like]
            is_protein = np.isin(res_names, _aa_like_res_names)

            _rna_like_res_names = self.sequence_encoding.all_res_names[self.sequence_encoding.is_rna_like]
            is_rna = np.isin(res_names, _rna_like_res_names)

            _dna_like_res_names = self.sequence_encoding.all_res_names[self.sequence_encoding.is_dna_like]
            is_dna = np.isin(res_names, _dna_like_res_names)

            is_ligand = ~(is_protein | is_rna | is_dna)

            _aa_like_res_names = self.sequence_encoding.all_res_names[self.sequence_encoding.is_aa_like]
            is_protein = np.isin(res_names, _aa_like_res_names)

            # Set annotations
            atom_array.set_annotation("is_protein", is_protein[atom_to_token_map])
            atom_array.set_annotation("is_rna", is_rna[atom_to_token_map])
            atom_array.set_annotation("is_dna", is_dna[atom_to_token_map])
            atom_array.set_annotation("is_ligand", is_ligand[atom_to_token_map])
        else:
            raise ValueError("Already has protein annotations, call this function first.")

        # ... Subsampling
        is_allowed = np.zeros_like(is_protein, dtype=bool)
        for allowed_type in self.allowed_types:
            is_allowed = np.logical_or(
                is_allowed, atom_array[token_starts].get_annotation(allowed_type)
            )
        atom_array = atom_array[is_allowed[atom_to_token_map]]

        if atom_array.array_length() == 0:
            raise ValueError("No protein tokens found in the atom array!")
            
        data['atom_array'] = atom_array
        return data        


class CreateDesignReferenceFeatures(Transform):
    '''
    Traditional AF3 will create a bunch of reference features based on the sequence and molecular identity.
    For our design, we do not have access to sequence so these features are useless

    However, this is a great place to add atom-level features as explicit conditioning or implicit
    classifier free guidance.

    Reduces time to process from ~0.5 to ~0.1 s.
    
    Old keys: ['ref_pos', 'ref_mask', 'ref_element', 'ref_charge', 'ref_atom_name_chars', 'ref_space_uid']

    New keys: ['ref_element', 'ref_atom_name_chars']
    '''
    requires_previous_transforms = ['CreateMasks']

    def __init__(self, **kwargs):
        DEFAULT_KWARGS = dict(
            conformer_generation_timeout=2.0,
            should_generate_automorphisms_with_rdkit=False,   
        )
        self.conformer_generation_kwargs = DEFAULT_KWARGS | kwargs

    def check_input(self, data: dict):
        check_contains_keys(data, ["atom_array"])
    
    def forward(self, data: dict) -> dict:
        atom_array = data['atom_array']
        I = atom_array.array_length()
        token_starts = get_token_starts(atom_array)
        token_level_array = atom_array[token_starts]
        token_has_sequence = token_level_array.token_has_sequence
        L = token_level_array.array_length()

        # ... Set up default reference features
        ref_pos = np.zeros_like(atom_array.coord, dtype=np.float32)
        ref_mask = np.zeros((I,), dtype=bool)
        ref_element = np.zeros((I,), dtype=np.int64)
        ref_charge = np.zeros((I,), dtype=np.int8)
        ref_pos_is_ground_truth = np.zeros((I,), dtype=bool)

        ref_atom_name_chars = _encode_atom_names_like_af3(atom_array.atom_name)
        _res_start_ends = get_residue_starts(atom_array, add_exclusive_stop=True)
        _res_starts, _res_ends = _res_start_ends[:-1], _res_start_ends[1:]
        ref_space_uid = struc.segments.spread_segment_wise(_res_start_ends, np.arange(len(_res_starts), dtype=np.int64))

        is_motif_atom = torch.nn.functional.one_hot(torch.from_numpy(atom_array.is_motif_atom).long(), num_classes=2).numpy()
        is_motif_token = torch.nn.functional.one_hot(torch.from_numpy(token_level_array.is_motif_token).long(), num_classes=2).numpy()

        # Token feature for token type;
        is_motif_token_without_index = atom_array.is_motif_atom_without_index[token_starts]
        motif_token_type = np.zeros((L, 3), dtype=np.int8)
        motif_token_type[is_motif_token] = 1
        motif_token_type[is_motif_token_without_index] = 2

        motif_atom_type = np.zeros((I, 3), dtype=np.int8)
        motif_atom_type[atom_array.is_motif_atom] = 1
        motif_atom_type[atom_array.is_motif_atom_without_index] = 2
        
        # ... Create reference features for unmasked subset (where we are allowed to use gt)
        if np.any(token_has_sequence):  # TODO
            has_sequence = atom_array.token_has_sequence  # (n_atoms,)
            # Expand to atom level
            atom_array_unmasked = atom_array[has_sequence]
            reference_features_unmasked = get_af3_reference_molecule_features(
                atom_array_unmasked, **self.conformer_generation_kwargs,
            )[0]
            # Don't provide reference conformers for unmasked atoms for now.
            # ref_pos[~is_masked] = reference_features_unmasked["ref_pos"]
            ref_atom_name_chars[has_sequence] = reference_features_unmasked["ref_atom_name_chars"]
            ref_mask[has_sequence] = reference_features_unmasked["ref_mask"]
            ref_element[has_sequence] = reference_features_unmasked["ref_element"]
            ref_charge[has_sequence] = reference_features_unmasked["ref_charge"]
            ref_pos_is_ground_truth[has_sequence] = reference_features_unmasked["ref_pos_is_ground_truth"]

        reference_features = {
            "ref_atom_name_chars": ref_atom_name_chars,  # (n_atoms, 4)
            "ref_pos": ref_pos,  # (n_atoms, 3)
            "ref_mask": ref_mask,  # (n_atoms)
            "ref_element": ref_element,  # (n_atoms)
            "ref_charge": ref_charge,  # (n_atoms)
            "ref_space_uid": ref_space_uid,  # (n_atoms)
            "ref_pos_is_ground_truth": ref_pos_is_ground_truth,  # (n_atoms)
            # Conditional masks
            "ref_is_motif_atom": is_motif_atom,  # (n_atoms, 2)
            "ref_is_motif_atom_mask": atom_array.is_motif_atom.copy(),  # (n_atoms)
            "ref_is_motif_token": is_motif_token, # (n_tokens, 2)
            "ref_motif_token_type": motif_token_type,  # (n_tokens, 3)  # 3 types of token
            'ref_motif_atom_type': motif_atom_type,  # (n_atoms, 3)  # 3 types of atom conditions
        }
        if 'feats' not in data:
            data['feats'] = {}
        data['feats'].update(reference_features)
        
        return data

class PadMaskedResiduesWithAtoms(Transform):
    requires_previous_transforms = ['CreateMasks']
    def __init__(self, n_atoms_per_token=14, atom_to_pad_from='CA', mask_atom_names=False, virtual_atom_element_name='X'):
        self.n_atoms_per_token = n_atoms_per_token
        self.atom_to_pad_from = atom_to_pad_from
        self.mask_atom_names = mask_atom_names
        self.virtual_atom_element_name=virtual_atom_element_name

    def check_input(self, data):
        check_contains_keys(data, ["atom_array"])

    @staticmethod
    def _create_pad_array(pad_atoms, n_pad, virtual_atom_element_name):
        '''
        Returns array of n_pad virtual atoms with the same annotations as pad_atoms
        '''
        pad_atoms = pad_atoms[0] if isinstance(pad_atoms, struc.AtomArray) else pad_atoms

        pad_atoms.element = virtual_atom_element_name
        pad_atoms.atom_name = "VX"  # Temporary
        
        # ... Expand to desired number of atoms
        pad_array = struc.array([pad_atoms] * n_pad)
        
        # ... Change occupancy | if any atom in the token has occupancy, set to 1.0
        occ = 1.0 if pad_atoms.occupancy.sum() > 0.0 else 0.0
        pad_array.occupancy = np.full(n_pad, occ)

        # ... Even if the input pad_atoms are all motif, we don't ever want padded atoms to be motif
        pad_array.is_motif_atom = np.zeros(n_pad, dtype=bool)
        return pad_array

    def _mask_names(self, residue_array):
        # ... Atom name masking
        # NB: CB required for downstream identifying which is the central atom (same reason we don't mask sequence)
        backbone_atoms = ['N', 'CA', 'C', 'O', 'CB']
        atom_names = backbone_atoms + [f'V{i}' for i in range(self.n_atoms_per_token - len(backbone_atoms))]
        atom_names = np.array(atom_names, dtype=residue_array.atom_name.dtype)[:len(residue_array)]
        residue_array.set_annotation('atom_name', atom_names)
        return residue_array

    def forward(self, data: dict) -> dict:
        atom_array = data['atom_array']
        token_starts = get_token_starts(atom_array)
        token_level_array = atom_array[token_starts]
        is_motif_token_with_seq = \
            token_level_array.is_motif_token & token_level_array.token_has_sequence

        token_ids = np.unique(atom_array.token_id)
        assert len(token_ids) == len(is_motif_token_with_seq), 'Token ids and token level array have different lengths!'
        
        # Should be protein, masked and not atomized
        is_atomized = atom_array.atomize[token_starts]
        is_protein = atom_array.is_protein[token_starts]
        is_paddable = is_protein \
            & ~is_atomized \
            & ~is_motif_token_with_seq \
            & ~token_level_array.is_motif_atom_without_index  # disallow guideposts from being padded 

        atom_array_padded=None
        for token_id in token_ids:  # zero-indexed
            # ... Create pad array
            residue_array = atom_array[atom_array.token_id == token_id].copy()

            if is_paddable[token_id]:
                n_pad = self.n_atoms_per_token - len(residue_array)

                if n_pad > 0:
                    mask = get_af3_token_representative_masks(residue_array, central_atom=self.atom_to_pad_from)
                    assert np.sum(mask) == 1, 'No representative atom ({}) found for token_id: {} mask: {}'.format(self.atom_to_pad_from, token_id, mask)
                    pad_atoms = residue_array[mask].copy()

                    pad_array = self._create_pad_array(pad_atoms, n_pad, virtual_atom_element_name=self.virtual_atom_element_name)

                    # ... Update residue array
                    residue_array = residue_array + pad_array

                    # ... BUG: chain_iid is not copied over | get_token_starts issues
                    residue_array.set_annotation('chain_iid', np.full(len(residue_array), residue_array.chain_iid[0], residue_array.chain_iid.dtype))
                    # Unsure if these have bugs too, but seems ok
                    # residue_array.set_annotation('atomize', np.full(len(residue_array), False))
                    # residue_array.set_annotation('res_id', np.full(len(residue_array), residue_array.res_id[0], residue_array.res_id.dtype))
                    # if 'atomize' in residue_array.get_annotation_categories():
                    # residue_array.set_annotation('ins_code', np.full(len(residue_array), residue_array.ins_code[0], residue_array.ins_code.dtype))
                    # residue_array.set_annotation('res_name', np.full(len(residue_array), residue_array.res_name[0], residue_array.res_name.dtype))

            # ... Mask atom name information
            if self.mask_atom_names and not is_motif_token_with_seq[token_id]:
                residue_array = self._mask_names(residue_array)
                token_starts_ = get_token_starts(residue_array)
                if len(token_starts_) != 1:
                    raise ValueError("Padded token not recognised as a single token!")

            # ... Update atom array
            atom_array_padded = atom_array_padded + residue_array if atom_array_padded is not None else residue_array

        data['atom_array'] = atom_array_padded
        return data

class AddIsXFeats(Transform):

    def __init__(self, X=[
        'is_backbone', 'is_sidechain', 'is_virtual', 'is_central', 'is_ca'
    ], central_atom='CA', virtual_atom_element_name='Virtual'):
        self.X = X
        self.central_atom = central_atom
        self.virtual_atom_element_name = virtual_atom_element_name
        self.update_atom_array = False

    def check_input(self, data):
        check_contains_keys(data, ["atom_array", "feats"])

    def forward(self, data: dict) -> dict:
        atom_array = data['atom_array']
        atom_array = add_backbone_and_sidechain_annotations(atom_array)
        token_level_array = atom_array[get_token_starts(atom_array)]
        _token_rep_mask = get_af3_token_representative_masks(atom_array, central_atom=self.central_atom)
        _token_rep_idxs = np.where(_token_rep_mask)[0]
        
        # ... Basic features
        if 'is_backbone' in self.X:
            is_backbone = data["atom_array"].get_annotation("is_backbone")
            data["feats"]["is_backbone"] = torch.from_numpy(is_backbone).to(dtype=torch.bool)

        if 'is_sidechain' in self.X:
            is_sidechain = data["atom_array"].get_annotation("is_sidechain")
            data["feats"]["is_sidechain"] = torch.from_numpy(is_sidechain).to(dtype=torch.bool)

        # Virtual atom feats
        if 'is_virtual' in self.X:
            data['feats']['is_virtual'] = atom_array.atom_name == self.virtual_atom_element_name

        for x in [
            'is_motif_token', 'is_motif_atom', 
            'is_motif_atom_with_fixed_pos', 
            'is_motif_atom_without_index'
        ]:
            if x not in self.X:
                continue
            if 'atom' in x:
                mask = atom_array.get_annotation(x).copy().astype(bool)
            else:
                mask = token_level_array.get_annotation(x).copy().astype(bool)
            data['feats'][x] = mask

        # ... Central and CA
        if 'is_central' in self.X:
            data['feats']['is_central'] = _token_rep_mask
        if 'is_ca' in self.X:
            if self.central_atom != 'CA':  # recompute if not CA
                _token_rep_mask = get_af3_token_representative_masks(atom_array, central_atom='CA')
                _token_rep_idxs = np.where(_token_rep_mask)[0]
            data['feats']['is_ca'] = _token_rep_mask
        
        return data

class MotifCenterRandomAugmentation(Transform):
    requires_previous_transforms = ['BatchStructuresForDiffusionNoising']

    def __init__(self, batch_size, sigma_perturb=0.0, decouple_gt_pos_feature_from_gt=False):
        """
        Randomly augments the coordinates of the motif center for diffusion training.
        This is to simulate the uncertainty in the motif center during training.

        decouple_gt_pos_feature_from_gt: 
            False: f(R(x||a) + z, Ra)  \approx R(x||a)  - noised coordinates in same frame as gt_pos
            True : f(R(x||a) + z, R'a) \approx R(x||a)  - noised coordinates in different frame as gt_pos
        """
        # self.decouple_gt_pos_feature_from_gt = decouple_gt_pos_feature_from_gt
        self.scale=1.0
        self.batch_size = batch_size
        self.sigma_perturb = sigma_perturb

    def check_input(self, data: dict):
        pass  # if anything's missing at this point, you're a bit screwed mate

    def forward(self, data):
        '''
        Applies CenterRandomAugmentation 
    
        And supplies the same rotated ground-truth coordinates as the input feature
        '''
        if data["is_inference"]:
            return data  # ori token behaviour set when creating atom array.

        xyz = data["coord_atom_lvl_to_be_noised"]  # (batch_size, n_atoms, 3)
        mask_atom_lvl = data["ground_truth"]["mask_atom_lvl"]
        mask_atom_lvl = mask_atom_lvl & ~data['feats']['is_motif_atom_without_index']  # Avoid double weighting
        mask_atom_lvl_expanded = mask_atom_lvl.expand(
            xyz.shape[0], -1
        )
        
        # Masked center during training (nb not motif mask - just non-zero occupancy)
        xyz = masked_center(xyz, mask_atom_lvl_expanded)

        # Small perturbation to prevent exact COM leakage
        xyz = xyz + torch.randn((self.batch_size, 3,), device=xyz.device)[:, None, :] * self.sigma_perturb

        # Apply random augmentation to the centered coordinates
        # RB: What's the point of the centering operation if we're going to add a random translation later?
        xyz = random_rigid_augmentation(
            xyz, batch_size=self.batch_size, s=self.scale
        )
        data["coord_atom_lvl_to_be_noised"] = xyz

        return data

class CleanupMaskedAtomArray(Transform):
    '''Used for saving outputs properly during inference'''

    def check_input(self, data: dict):
        check_contains_keys(data, ["atom_array"])
    
    def forward(self, data):
        atom_array = data['atom_array'].copy()
        atom_array.bonds = None
        atom_array.res_name[~atom_array.is_motif_token] = "UNK"  # Set non-motif residues to UNK
        if np.any(atom_array.is_motif_atom_without_index):  
            # HACK: Since res_ids are the same, we should save them with a different chain index.
            atom_array.chain_id[atom_array.is_motif_atom_without_index] = 'X'
        data['atom_array'] = atom_array
        return data

class RemoveNoiseFromUnmaskedAtoms(Transform):
    requires_previous_transforms = ['SampleEDMNoise']
    def check_input(self, data: dict):
        check_contains_keys(data, ["coord_atom_lvl_to_be_noised"])
        check_contains_keys(data, ["feats"])
    def forward(self, data: dict) -> dict:
        is_motif_atom_with_fixed_pos = data['feats']['is_motif_atom_with_fixed_pos']
        data['noise'][..., is_motif_atom_with_fixed_pos, :] = 0.0
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
        data["ground_truth"]["seq_token_lvl"] = torch.from_numpy(restype)

        return data