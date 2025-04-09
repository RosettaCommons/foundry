# from typing import Any

# import biotite.structure as struc
# import numpy as np
# import torch
# import torch.nn.functional as F
# from biotite.structure import AtomArray
# from biotite.structure.filter import filter_amino_acids
# from biotite.structure.info import vdw_radius_protor
# from datahub.transforms._checks import check_contains_keys
# from datahub.transforms.base import Transform


# def calculate_atomwise_sasa_and_rasa(
#     atom_array: AtomArray, probe_radius: float = 1.4, atom_radii: str | np.ndarray = "ProtOr", point_number: int = 100
# ) -> np.ndarray:
#     """
#     Calculate the SASA and RASA for each atom in `atom_array`, excluding those
#     with NaN coordinates. The output will have the same length as the
#     input AtomArray, with NaN values for excluded (invalid) atoms and atoms without defined vdW radius.

#      Args:
#         probe_radius (float, optional): Van-der-Waals radius of the probe in Angstrom. Defaults to 1.4 (for water).
#         atom_radii (str | np.ndarray, optional): Atom radii set to use for calculation. Defaults to "ProtOr". "ProtOr" will not get sasa's for hydrogen atoms and some other atoms, like ions or certain atoms with charges
#         point_number (int, optional): Number of points in the Shrake-Rupley algorithm to sample for calculating SASA. Defaults to 100.

#     """
#     # 1) Create a boolean vector for valid atoms (no NaNs in their coordinates)
#     has_resolved_coordinates = ~np.isnan(atom_array.coord).any(axis=-1)

#     # 2) Slice the array to keep only valid atoms
#     valid_atom_array = atom_array[has_resolved_coordinates]

#     # 3) Compute SASA on only the valid atoms
#     valid_sasa = struc.sasa(
#         valid_atom_array, probe_radius=probe_radius, vdw_radii=atom_radii, point_number=point_number
#     )

#     # 4) Create full-length arrays (with NaNs for invalid atoms)
#     full_sasa = np.full(atom_array.array_length(), np.nan, dtype=float)

#     # 5) Place the valid SASA values back in their original positions
#     full_sasa[has_resolved_coordinates] = valid_sasa

#     # 6) Get maximum SASA's for the atoms with SASA's
#     max_sasa = []
#     for atom in valid_atom_array:
#         try:
#             vdw_radius = vdw_radius_protor(atom.res_name, atom.atom_name)
#             if vdw_radius is None:
#                 vdw_radius = 1.8  ### default radii for non-hydrogen atoms
#             max_sasa_valid = 4.0 * np.pi * (vdw_radius + probe_radius) ** 2
#             max_sasa += [max_sasa_valid]
#         except (ValueError, KeyError):
#             max_sasa += [np.nan]

#     # 7) Calculate RASA for the atoms
#     valid_rasa = valid_sasa / max_sasa

#     # 8) Create full-length arrays (with NaNs for invalid atoms)
#     full_rasa = np.full(atom_array.array_length(), np.nan, dtype=float)

#     # 9) Place the valid RASA values back in their original positions
#     full_rasa[has_resolved_coordinates] = valid_rasa

#     return full_sasa, full_rasa


# def discretize_rasa(atom_array: AtomArray, low: float = 0, high: float = 0.2, n_bins: int = 3) -> np.ndarray:
#     """
#     Discretize the RASA for each atom in `atom_array`. Bin them into n_bins number of bins with given minimum, maximum and number of bins, excluding the 'no condition' bin. The output will be a torch.Tensor object that contains the one-hot-encoded atomwise RASA values.

#      Args:
#         low (float, optional): Minimum for binning
#         high (float, optional): Maximum for binning
#         n_bins (int, optional): Number of bins wanted, does not count the 'no condition' bin, so the final total number of bins will be n_bins+1 when we include the 'no condition' bin
#     """

#     # 1) create boolean vector that highlights atoms in an atomarray coming from proteins
#     protein_mask = filter_amino_acids(atom_array)

#     # 2) Extract rasa array and determine step_size for bins
#     rasa = atom_array.rasa
#     step_size = (high - low) / n_bins

#     # 3) Assign each atom to a bin.
#     bins = np.where(
#         np.isnan(rasa) | protein_mask,
#         3,  # assign protein atoms and atoms with nan rasa to the last category
#         np.where(
#             rasa < (0 + step_size),  # 0 to 0 + step_size will be assigned to first category
#             0,
#             np.where(
#                 rasa < (0 + 2 * step_size),  # make this a general function
#                 1,
#                 2,
#             ),
#         ),
#     )

#     return bins


# class CalculateSASAandRASA(Transform):
#     """Transform for calculating Solvent-Accessible Surface Area (SASA) and relative SASA (RASA) for each atom in an AtomArray."""

#     def __init__(
#         self,
#         probability_of_calc: float = 1.0,
#         probe_radius: float = 1.4,
#         atom_radii: str | np.ndarray = "ProtOr",
#         point_number: int = 100,
#     ):
#         """
#         Initialize the CalculateSASAandRASA transform.

#         Args:
#             probe_radius (float, optional): Van-der-Waals radius of the probe in Angstrom. Defaults to 1.4 (for water).
#             atom_radii (str | np.ndarray, optional): Atom radii set to use for calculation. Defaults to "ProtOr". "ProtOr" will not get sasa's for hydrogen atoms and some other atoms, like ions or certain atoms with charges
#             point_number (int, optional): Number of points in the Shrake-Rupley algorithm to sample for calculating SASA. Defaults to 100.
#         """
#         self.probe_radius = probe_radius
#         self.atom_radii = atom_radii
#         self.point_number = point_number
#         self.probability_of_calc = probability_of_calc

#     def check_input(self, data: dict[str, Any]) -> None:
#         check_contains_keys(data, ["atom_array"])

#     def forward(self, data: dict, key_to_add_sasa_and_rasa_to: str = "atom_array") -> dict:
#         """
#         Calculates SASA, RASA and binned RASA and adds them to the data dictionary under the key `atom_array`.
#         Args:
#             data: dict
#                 A dictionary containing the input data atomarray.
#             key_to_add_sasa_and_rasa_to: str
#                 The key in the data dictionary to add the SASA and RASA values to.

#         Returns:
#             dict: The data dictionary with SASA and RASA values added.
#         """
#         if np.random.rand() > self.probability_of_calc:
#             return data

#         atom_array: AtomArray = data[key_to_add_sasa_and_rasa_to]
#         sasa, rasa = calculate_atomwise_sasa_and_rasa(
#             atom_array,
#             self.probe_radius,
#             self.atom_radii,
#             self.point_number,
#         )
#         atom_array.set_annotation("sasa", sasa)
#         atom_array.set_annotation("rasa", rasa)

#         bins = discretize_rasa(atom_array)

#         atom_array.set_annotation("rasa_bins", bins)

#         data[key_to_add_sasa_and_rasa_to] = atom_array

#         if "feats" not in data:
#             data["feats"] = {}
#         data["feats"]["sasa"] = sasa
#         data["feats"]["rasa"] = rasa
#         data["feats"]["rasa_bins"] = bins
#         data["feats"]["ref_rasa"] = F.one_hot(torch.Tensor(data['atom_array'].rasa_bins).long(), num_classes=4)

#         return data