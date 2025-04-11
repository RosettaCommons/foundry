from collections import defaultdict
from typing import Any, Literal

import numpy as np
import toolz
import torch
from biotite.structure import AtomArray
from cifutils.constants import ELEMENT_NAME_TO_ATOMIC_NUMBER
from cifutils.tools.rdkit import atom_array_from_rdkit
from cifutils.utils.selection import get_residue_starts
from datahub.enums import GroundTruthConformerPolicy
from datahub.transforms._checks import (
    check_atom_array_annotation,
    check_contains_keys,
    check_is_instance,
    check_nonzero_length,
)
from datahub.transforms.af3_reference_molecule import (
    KNOWN_CCD_CODES,
    _encode_atom_names_like_af3,
    _get_rdkit_mols_with_conformers,
    _map_reference_conformer_to_residue,
    logger,
)
from datahub.transforms.base import Transform
from datahub.transforms.chirals import get_rf2aa_chiral_features
from datahub.transforms.rdkit_utils import (
    find_automorphisms_with_rdkit,
    get_chiral_centers,
    sample_rdkit_conformer_for_atom_array,
)
from datahub.utils.geometry import masked_center, random_rigid_augmentation


def get_af3_reference_molecule_features_ch(
    atom_array: AtomArray,
    conformer_generation_timeout: float = 10.0,
    should_generate_automorphisms_with_rdkit: bool = True,
    apply_random_rotation_and_translation: bool = True,
    use_element_for_atom_names_of_atomized_tokens: bool = False,
    timeout_strategy: Literal["signal", "subprocess"] = "subprocess",
    **generate_conformers_kwargs,
) -> dict[str, Any]:
    """Get AF3 reference features for each residue in the atom array.

    Args:
        - atom_array (AtomArray): The input atom array.
        - conformer_generation_timeout (float, optional): Maximum time allowed for conformer generation per residue.
            Defaults to 10.0 seconds. If None, no timeout is applied and the timeout strategy is ignored (no subprocesses will be spawned).
        - should_generate_automorphisms_with_rdkit (bool, optional): Whether to generate automorphisms using RDKit. For example,
            we may want to generate automorphisms directly with networkx instead. Defaults to True.
        - apply_random_rotation_and_translation (bool, optional): Whether to apply a random rotation and translation to each conformer (AF-3-style)
        - timeout_strategy (Literal["signal", "subprocess"]): The strategy to use for the timeout.
            Defaults to "subprocess".
        - **generate_conformers_kwargs: Additional keyword arguments to pass to the generate_conformers function.

    Returns:
        dict[str, Any]: A dictionary containing the generated reference features.

    This function generates the following reference features for AF3:
        - ref_pos: [N_atoms, 3] Atom positions in the reference conformer, with a random rotation and
            translation applied. Atom positions are given in Å.
        - ref_mask: [N_atoms] Mask indicating which atom slots are used in the reference conformer.
        - ref_element: [N_atoms, 128] One-hot encoding of the element atomic number for each atom in the
            reference conformer, up to atomic number 128.
        - ref_charge: [N_atoms] Charge for each atom in the reference conformer.
        - ref_atom_name_chars: [N_atoms, 4, 64] One-hot encoding of the unique atom names in the reference conformer.
            Each character is encoded as ord(c) - 32, and names are padded to length 4.
        - ref_space_uid: [N_atoms] Numerical encoding of the chain id and residue index associated with
            this reference conformer. Each (chain id, residue index) tuple is assigned an integer on first appearance.
        - ref_automorphs: A dictionary mapping the `ref_space_uid` to the automorphisms
            of the reference conformer.
        - ref_pos_is_ground_truth (optional): [N_atoms] Whether the reference conformer is the ground-truth conformer.
            Determined by the `ground_truth_conformer_policy` annotation.

    Reference:
        - Section 2.8 of the AF3 supplementary information
          https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf
    """
    # Generate reference conformers for each residue (if cropped, each residue that has tokens in the crop)
    # ... get residue-level stochiometry
    _res_start_ends = get_residue_starts(atom_array, add_exclusive_stop=True)
    _res_starts, _res_ends = _res_start_ends[:-1], _res_start_ends[1:]
    _res_names = atom_array.res_name[_res_starts]
    res_stochiometry = dict(zip(*np.unique(_res_names, return_counts=True)))
    _has_explicit_hydrogens = "H" in atom_array.element

    # ... get reference molecules with conformers for each residue
    # (We do not generate conformers for unknown CCD codes here, as we will do that later)
    ref_mols = _get_rdkit_mols_with_conformers(
        res_stochiometry=res_stochiometry,
        hydrogen_policy="auto" if _has_explicit_hydrogens else "remove",
        timeout=conformer_generation_timeout,
        timeout_strategy=timeout_strategy,
        **generate_conformers_kwargs,
    )

    # ... generate conformers for CCD codes that are unknown (including UNL)
    unknown_ccd_conformers = defaultdict(list)
    ref_mols_with_unk = ref_mols.copy()
    if not all(res_name in KNOWN_CCD_CODES for res_name in res_stochiometry):
        res_indices_with_unknown = np.where(
            ~np.isin(_res_names, list(KNOWN_CCD_CODES))
        )[0]
        for res_index in res_indices_with_unknown:
            res_name = _res_names[res_index]

            conf_i = sample_rdkit_conformer_for_atom_array(
                atom_array[_res_starts[res_index] : _res_ends[res_index]],
                timeout=conformer_generation_timeout,
                timeout_strategy=timeout_strategy,
                **generate_conformers_kwargs,
            )
            unknown_ccd_conformers[res_name].append(conf_i)
            ref_mols_with_unk[res_name] = conf_i

    # ... initialize automorphism-related variables (which we may or may not be needed)
    ref_mol_automorphs = None
    ref_automorphs = None
    ref_automorphs_mask = None

    if should_generate_automorphisms_with_rdkit:
        # ... get automorphisms
        ref_mol_automorphs = toolz.valmap(find_automorphisms_with_rdkit, ref_mols)
        _max_automorphs = max(map(len, ref_mol_automorphs.values()))
        # ...initialize tensors to store automorphisms and masks
        ref_automorphs = np.zeros((_max_automorphs, len(atom_array), 2), dtype=int)
        ref_automorphs_mask = np.zeros((_max_automorphs, len(atom_array)), dtype=bool)

    # ... initialize reference features
    ref_pos = np.zeros((len(atom_array), 3), dtype=np.float32)
    ref_mask = np.zeros(len(atom_array), dtype=bool)
    ref_pos_is_ground_truth = np.zeros(len(atom_array), dtype=bool)

    # Fill `ref_pos` and `ref_mask` arrays
    # ... helper variable to keep track of the next conformer to use for each residue type
    _next_conf_idx = {res_name: 0 for res_name in ref_mols}

    # ... iterate over all residues in the atom array and fill the `ref_pos` and `ref_mask` arrays using the next reference conformer for each residue type
    # We also check the `ground_truth_conformer_policy` annotation to see if we should use the ground-truth conformer
    max_automorphs = 1
    for res_start, res_end in zip(_res_starts, _res_ends):
        res_name = atom_array.res_name[res_start]

        # ... turn conformer into an atom array
        if res_name not in KNOWN_CCD_CODES:
            # (conformers for unknown CCD codes are already atom arrays, since we generated them directly)
            conformer = unknown_ccd_conformers[res_name][_next_conf_idx[res_name]]
        else:
            conformer = atom_array_from_rdkit(
                ref_mols[res_name],
                conformer_id=_next_conf_idx[res_name],
                remove_hydrogens=True,
            )

        if "ground_truth_conformer_policy" in atom_array.get_annotation_categories():
            # We replace the generated conformer with the ground-truth conformer if either:
            # (a) the ground-truth conformer policy is set to "replace" for all atoms in the residue
            # (b) the current conformer is all 0's/NaN's (i.e., the conformer generation failed), and the policy is set to "fallback" for all atoms in the residue
            if np.all(
                atom_array.ground_truth_conformer_policy[res_start:res_end]
                == GroundTruthConformerPolicy.REPLACE
            ) or (
                np.all(np.nan_to_num(conformer.coord) == 0)
                and np.all(
                    atom_array.ground_truth_conformer_policy[res_start:res_end]
                    == GroundTruthConformerPolicy.FALLBACK
                )
            ):
                # NOTE: Inefficient since we generate with RDKit, and then discard, the conformer; however, this replacement-based approach is more interpretable and thus preferred
                if np.isnan(atom_array.coord[res_start:res_end]).any():
                    logger.warning(
                        "Ground-truth conformer requested, but NaNs found in the atom array. Conformer will not be replaced with ground truth."
                    )
                else:
                    # ... use the ground-truth AtomArray (e.g., during inference if we provide a SDF, or if we want to leak ligand geometry)
                    conformer = atom_array[res_start:res_end]
                    # (Center around the origin to avoid leaking 1D information)
                    conformer.coord = masked_center(conformer.coord)
                    ref_pos_is_ground_truth[res_start:res_end] = True

        # ... map the reference conformer information to the given residue
        _ref_pos, _ref_mask, _ref_automorphs = _map_reference_conformer_to_residue(
            res_name=res_name,
            atom_names=atom_array.atom_name[res_start:res_end],
            conformer=conformer,
            automorphs=ref_mol_automorphs[res_name] if ref_mol_automorphs else None,
        )

        # ... apply a random rotation and translation to the reference conformer, if requested
        if apply_random_rotation_and_translation:
            # TODO: Implement more elegantly directly in numpy
            _ref_pos = random_rigid_augmentation(
                torch.from_numpy(_ref_pos[np.newaxis, :]), batch_size=1
            ).numpy()

        # ... fill the reference features for this residue
        ref_pos[res_start:res_end] = _ref_pos
        ref_mask[res_start:res_end] = _ref_mask

        # ... fill the automorphisms for this residue, generating automorphisms from RDKit
        if _ref_automorphs is not None:
            ref_automorphs[: len(_ref_automorphs), res_start:res_end] = _ref_automorphs
            ref_automorphs_mask[: len(_ref_automorphs), res_start:res_end] = True
            max_automorphs = max(max_automorphs, len(_ref_automorphs))

        # ... update to the next conformer index
        _next_conf_idx[res_name] += 1

    # ... resize the reference automorphism arrays to the maximum number of automorphisms
    if ref_automorphs is not None:
        ref_automorphs = ref_automorphs[:max_automorphs]
        ref_automorphs_mask = ref_automorphs_mask[:max_automorphs]

    # Generate remaining reference features
    # ... element
    ref_element = (
        atom_array.atomic_number
        if "atomic_number" in atom_array.get_annotation_categories()
        else np.vectorize(ELEMENT_NAME_TO_ATOMIC_NUMBER.get)(atom_array.element)
    )
    # ... charge
    ref_charge = atom_array.charge

    # ... atom name
    ref_atom_name_chars = _encode_atom_names_like_af3(atom_array.atom_name)

    if use_element_for_atom_names_of_atomized_tokens:
        assert (
            "atomize" in atom_array.get_annotation_categories()
        ), "Atomize annotation is required when using element for atom names of atomized tokens."
        ref_atom_name_chars[atom_array.atomize] = _encode_atom_names_like_af3(
            atom_array.element[atom_array.atomize]
        )

    # ... space uid (type conversion needed for some older torch versions)
    ref_space_uid = atom_array.token_id.astype(np.int64)
    ref_conformer = {
        "ref_pos": ref_pos,  # (n_atoms, 3)
        "ref_mask": ref_mask,  # (n_atoms)
        "ref_element": ref_element,  # (n_atoms)
        "ref_charge": ref_charge,  # (n_atoms)
        "ref_atom_name_chars": ref_atom_name_chars,  # (n_atoms, 4)
        "ref_space_uid": ref_space_uid,  # (n_atoms)
        "ref_automorphs": ref_automorphs,  # (max_automorphs, n_atoms, 2), residue-local indices
        "ref_automorphs_mask": ref_automorphs_mask,  # (max_automorphs, n_atoms)
        "ref_pos_is_ground_truth": ref_pos_is_ground_truth,  # (n_atoms)
    }
    return ref_conformer, ref_mols_with_unk


class GetAF3ReferenceMoleculeFeatures(Transform):
    """
    Generate AF3 reference molecule features for each residue in the atom array.

    This transform adds the following features to the data dictionary under the 'feats' key:
        - ref_pos: [N_atoms, 3] Atom positions in the reference conformer, with a random rotation and
          translation applied. Atom positions are given in Å.
        - ref_mask: [N_atoms] Mask indicating which atom slots are used in the reference conformer.
        - ref_element: [N_atoms] One-hot encoding of the element atomic number for each atom in the
          reference conformer, up to atomic number 128.
        - ref_charge: [N_atoms] Charge for each atom in the reference conformer.
        - ref_atom_name_chars: [N_atoms, 4, 64] One-hot encoding of the unique atom names in the reference conformer.
          Each character is encoded as ord(c) - 32, and names are padded to length 4.
        - ref_space_uid: [N_atoms] Numerical encoding of the chain id and residue index associated with
          this reference conformer. Each (chain id, residue index) tuple is assigned an integer on first appearance.

    Optionally, the following features can be added as well:
        - ref_automorphs: [N_automorphs, N_atoms, 2] Automorphisms of the reference conformer.
          Each automorphism is a mapping from one atom to another. The first column is the source atom index,
          and the second column is the target atom index. The automorphisms are given in residue-local indices.
        - ref_automorphs_mask: [N_automorphs, N_atoms] Mask indicating which atom slots are used in the automorphisms.

    Note: This transform should be applied after cropping.

    Reference:
        - Section 2.8 of the AF3 supplementary information
          https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf
    """

    requires_previous_transforms = ["AddGlobalTokenIdAnnotation"]

    def __init__(
        self,
        conformer_generation_timeout: float = 10.0,
        should_generate_automorphisms_with_rdkit: bool = True,
        save_rdkit_mols: bool = True,
        use_element_for_atom_names_of_atomized_tokens: bool = False,
        apply_random_rotation_and_translation: bool = True,
        **generate_conformers_kwargs,
    ):
        self.conformer_generation_timeout = conformer_generation_timeout
        self.should_generate_automorphisms_with_rdkit = (
            should_generate_automorphisms_with_rdkit
        )
        self.generate_conformers_kwargs = generate_conformers_kwargs
        self.save_rdkit_mols = save_rdkit_mols
        self.use_element_for_atom_names_of_atomized_tokens = (
            use_element_for_atom_names_of_atomized_tokens
        )
        self.apply_random_rotation_and_translation = (
            apply_random_rotation_and_translation
        )
        self.generate_conformers_kwargs = generate_conformers_kwargs

        if self.use_element_for_atom_names_of_atomized_tokens:
            logger.warning("Using element type for atom names of atomized tokens.")

    def check_input(self, data: dict):
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(
            data, ["res_name", "element", "charge", "atom_name", "token_id"]
        )

        if self.use_element_for_atom_names_of_atomized_tokens:
            check_atom_array_annotation(data, ["atomize"])

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        # Generate reference features
        reference_features, rdkit_mols = get_af3_reference_molecule_features_ch(
            atom_array,
            conformer_generation_timeout=self.conformer_generation_timeout,
            should_generate_automorphisms_with_rdkit=self.should_generate_automorphisms_with_rdkit,
            use_element_for_atom_names_of_atomized_tokens=self.use_element_for_atom_names_of_atomized_tokens,
            apply_random_rotation_and_translation=self.apply_random_rotation_and_translation,
            **self.generate_conformers_kwargs,
        )

        # Add reference features to the 'feats' dictionary
        if "feats" not in data:
            data["feats"] = {}
        data["feats"].update(reference_features)

        if self.save_rdkit_mols:
            if "rdkit" not in data:
                data["rdkit"] = {}
            data["rdkit"].update(rdkit_mols)

        return data


def _get_reference_conformer_to_residue_mapping(
    atom_names: np.ndarray, conformer: AtomArray
) -> tuple[np.ndarray]:
    """
    Maps atom indices from a reference conformer (as an AtomArray) to the specified residue, dropping all atoms that are not in the residue.
    Args:
        - atom_names (np.ndarray): Array of atom names in the residue to map to
        - conformer (AtomArray): The reference conformer, as an AtomArray (containing the atom_name annotation)
    Returns:
        - ref_map (np.ndarray): Index in atom of reference positions (-1 if masked)
    """

    # ... mark the atoms that are in the residue (keep) and where
    keep = np.zeros(len(conformer), dtype=bool)  # [n_atoms_in_conformer]
    to_within_res_idx = -np.ones(len(conformer), dtype=int)  # [n_atoms_in_conformer]

    for i, atom_name in enumerate(atom_names):
        matching_atom_idx = np.where(conformer.atom_name == atom_name)[0]
        if len(matching_atom_idx) == 0:
            logger.warning(f"Atom {atom_name} not found in conformer.")
            continue
        matching_atom_idx = matching_atom_idx.item()
        keep[matching_atom_idx] = True
        to_within_res_idx[matching_atom_idx] = i

    return to_within_res_idx  # [n_atoms_in_conf]


class AddAF3ChiralFeatures(Transform):
    """
    AddAF3ChiralFeatures adds chiral features.

    This transform adds the following features to the data dictionary under the 'feats' key:
        - chiral_feats: [N_chiral_centers, 5] A listing of chiral centers of the format:
                  tensor([[ 5.,  1.,  2.,  3.,  0.61546...],...])
          Here, the first 4 columns define atom indices of chiral center; the 5th is target dihedral

    Metadata from GetRDKitChiralCenters is needed for this transform.

    Args:
        data (dict[str, Any]): A dictionary containing the input data, including the atom array and chiral centers.

    Returns:
        dict[str, Any]: The updated `data` dictionary with the added chiral features under the `feats` key.
    """

    requires_previous_transforms = ["GetRDKitChiralCenters"]

    def check_input(self, data: dict[str, Any]):
        check_contains_keys(data, ["atom_array", "chiral_centers", "rdkit", "feats"])
        check_is_instance(data, "atom_array", AtomArray)
        check_nonzero_length(data, "atom_array")

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array: AtomArray = data["atom_array"]

        # We're going to use the same logic we do in GetAF3ReferenceMoleculeFeatures
        # ... get residue-level stochiometry
        _res_start_ends = get_residue_starts(atom_array, add_exclusive_stop=True)
        _res_starts, _res_ends = _res_start_ends[:-1], _res_start_ends[1:]

        all_chirals = []
        for res_start, res_end in zip(_res_starts, _res_ends):
            res_name = atom_array.res_name[res_start]

            chirals = data["chiral_centers"][res_name]
            if len(chirals) == 0:
                continue

            # get rdkit->atomarray mapping
            conformer = atom_array_from_rdkit(
                data["rdkit"][res_name],
                conformer_id=0,
                remove_hydrogens=True,
            )
            _ref_to_conf_map = _get_reference_conformer_to_residue_mapping(
                atom_names=atom_array.atom_name[res_start:res_end], conformer=conformer
            )

            # calculate chirals from reference conformer
            chirals = get_rf2aa_chiral_features(chirals, torch.tensor(conformer.coord))

            # remap reference conformer to native index
            chirals[:, :4] = torch.tensor(_ref_to_conf_map)[chirals[:, :4].long()]

            # remove unmasked chirals
            mask = (chirals[:, :4] >= 0).all(dim=1)
            chirals = chirals[mask]

            # add atom offset
            chirals[:, :4] = chirals[:, :4] + res_start

            all_chirals.append(chirals)

        all_chirals = torch.cat(all_chirals, dim=0)
        data["feats"]["chiral_feats"] = all_chirals  # [n_chirals, 5]

        return data


class GetRDKitChiralCenters(Transform):
    """
    Identify chiral centers in the RDKit molecules stored in the `data["rdkit"]` dictionary.
    Returns a dictionary mapping each residue name to a list of chiral centers, e.g:
      data["chiral_centers"] = {
          ...
          "ILE": [
              {'chiral_center_idx': 1, 'bonded_explicit_atom_idxs': [0, 2, 4], 'chirality': 'S'},
              {'chiral_center_idx': 4, 'bonded_explicit_atom_idxs': [1, 5, 6], 'chirality': 'S'}
          ],
          ...
      }
    Each chiral center is a dict with a center atom index, 3 or 4 bonded atom indices, and the
    RDKit-determined chirality.

    Uses RDKit molecules first computed in GetAF3ReferenceMoleculeFeatures.

    Args:
        data (dict[str, Any]): A dictionary containing the input data, including RDKit molecules
            under the `"rdkit"` key.

    Returns:
        dict[str, Any]: The updated `data` dictionary with `chiral_centers` containing chiral
            centers for each molecule.
    """

    requires_previous_transforms = ["GetAF3ReferenceMoleculeFeatures"]

    def check_input(self, data: dict[str, Any]):
        check_contains_keys(data, ["rdkit"])
        check_is_instance(data, "rdkit", dict)
        check_nonzero_length(data, "rdkit")

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        data["chiral_centers"] = {}
        # Get chiral centers for all rdkit mols
        for resname, rdmol in data["rdkit"].items():
            try:
                # Get the chiral centers (returned are the indices of the chiral center atoms
                #  within the `obmol` object)
                data["chiral_centers"][resname] = get_chiral_centers(rdmol)

            except Exception as e:
                logger.warning(
                    f"Failed to find chiral centers for molecule {resname}: {e}"
                )

        return data
