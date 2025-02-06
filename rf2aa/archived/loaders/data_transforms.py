import logging
from functools import lru_cache
from typing import Any

import torch
import torch.nn.functional as F

import rf2aa
from rf2aa.chemical import ChemicalData as ChemData
from rf2aa.chemical import load_pdb_ideal_sdf_strings
from rf2aa.data_new.transforms._checks import check_contains_keys
from rf2aa.data_new.transforms.base import Compose, Transform

logger = logging.getLogger(__name__)


class ComputeResidueIndex(Transform):
    def check_input(self, data: dict):
        check_contains_keys(data, ["res_idxs", "akeys_sm"])

    def forward(self, data: dict) -> dict:
        res_idxs_poly = data["res_idxs"]
        akeys_sm = data["akeys_sm"]
        if len(akeys_sm) == 0:
            data["residue_idx"] = res_idxs_poly
            return data
        res_idx_nonpoly = torch.cat(
            [
                torch.tensor([int(akey[1]) for akey in akeys_per_chain])
                for akeys_per_chain in akeys_sm
            ],
            dim=0,
        )
        res_idxs = torch.cat([res_idxs_poly, res_idx_nonpoly], dim=0)
        data["residue_idx"] = res_idxs
        return data


class ComputeEntityIndex(Transform):
    """Each chain with a distinct sequence is assigned a unique index"""

    def check_input(self, data: dict):
        check_contains_keys(data, ["ch_label"])

    def forward(self, data: dict[str, Any], *args, **kwargs) -> dict[str, Any]:
        data["entity_idx"] = data["ch_label"]
        return data


class ComputeAsymIndex(Transform):
    """Each unique chain in the protein is assigned a unique index"""

    def check_input(self, data: dict):
        check_contains_keys(data, ["Ls_poly", "Ls_sm"])

    def forward(self, data: dict[str, Any], *args, **kwargs) -> dict[str, Any]:
        Ls = data["Ls_poly"] + data["Ls_sm"]
        asym_idx = torch.cat(
            [torch.tensor(i).repeat(L) for i, L in enumerate(Ls)], dim=0
        )
        data["asym_idx"] = asym_idx
        return data


class ComputeSymmIndex(Transform):
    """Identical sequences get different indices"""

    def check_input(self, data: dict[str, Any]) -> None:
        return super().check_input(data)

    def forward(self, data: dict[str, Any], *args, **kwargs) -> dict[str, Any]:
        symm_chain_idx_poly = self.find_symm_chains(
            data["ch_label_poly"], data["Ls_poly"]
        )
        symm_chain_idx_nonpoly = self.find_symm_chains(
            data["ch_label_sm"], data["Ls_sm"]
        )
        symm_chain_idx = torch.cat([symm_chain_idx_poly, symm_chain_idx_nonpoly], dim=0)
        data["symm_idx"] = symm_chain_idx

        return data

    def find_symm_chains(self, ch_label, Ls):
        num_occurrences = {}
        offset = 0
        symm_idx = torch.zeros_like(ch_label)
        for i, L in enumerate(Ls):
            key = ch_label[offset : offset + L].tolist()
            key = tuple(key)
            if key not in num_occurrences:
                num_occurrences[key] = 0

            symm_idx[offset : offset + L] = num_occurrences[key]

            num_occurrences[key] += 1
            offset += L
        return symm_idx


class AddReferencePositions(Transform):
    def check_input(self, data: dict):
        check_contains_keys(data, ["seq_poly", "lig_names", "msa_sm"])

    def forward(self, data: dict) -> dict:
        self.molname2sdf = self._load_ligand_ideal_sdfs()
        seq_poly = data["seq_poly"]
        seq_3letter = [ChemData().num2aa[aa] for aa in seq_poly]
        protein_ref_pos, protein_has_ref_pos = self.generate_reference_positions(
            seq_3letter
        )
        if len(data["lig_names"]) == 0:
            combined_ref_pos = protein_ref_pos
            combined_seq = seq_poly
            combined_has_ref_pos = protein_has_ref_pos
        else:
            ligand_ref_pos, ligand_has_ref_pos = self.generate_reference_positions(
                data["lig_names"]
            )
            ligand_ref_pos, ligand_has_ref_pos = self.check_ligand_reference_conf_dims(
                data, ligand_ref_pos
            )
            combined_ref_pos = torch.cat([protein_ref_pos, ligand_ref_pos], dim=0)
            combined_seq = torch.cat(
                [data["seq_poly"], data["msa_sm"][0]], dim=0
            ).long()
            combined_has_ref_pos = torch.cat(
                [protein_has_ref_pos, ligand_has_ref_pos], dim=0
            )
        is_real_atom = ChemData().heavyatom_mask[combined_seq]

        # make atom36 container for coordinates
        ideal_coords_atom36 = torch.zeros_like(is_real_atom, dtype=torch.float32)
        ideal_coords_atom36 = ideal_coords_atom36[..., None].repeat(1, 1, 3)
        try:
            ideal_coords_atom36[is_real_atom] = combined_ref_pos
        except Exception as e:
            logger.error(f"Error in adding reference positions: {e}")

        has_ref_pos_atom36 = torch.zeros_like(is_real_atom, dtype=torch.bool)
        has_ref_pos_atom36[is_real_atom] = combined_has_ref_pos
        data["ref_pos_atom36"] = ideal_coords_atom36
        data["ref_mask"] = has_ref_pos_atom36
        data["seq_combined"] = combined_seq
        return data

    @lru_cache(maxsize=None)
    def _load_ligand_ideal_sdfs(self) -> dict:
        molecules = load_pdb_ideal_sdf_strings()
        return molecules

    def generate_reference_positions(self, molecule_names: list) -> dict:
        positions = []
        all_has_ref_pos = []
        for molecule in molecule_names:
            for molecule_3letter in molecule.split("_"):
                if molecule_3letter == "UNK":
                    NUM_ATOMS_IN_UNK = 5
                    coordinates = torch.zeros(
                        (NUM_ATOMS_IN_UNK, 3), dtype=torch.float32
                    )
                    has_ref_pos = torch.zeros(NUM_ATOMS_IN_UNK, dtype=torch.bool)
                elif molecule_3letter in self.molname2sdf:
                    coordinates, _ = self.generate_conformer(molecule_3letter)
                    has_ref_pos = torch.ones(coordinates.size(0), dtype=torch.bool)
                else:
                    coordinates = torch.zeros((0, 3), dtype=torch.float32)
                    has_ref_pos = torch.zeros(0, dtype=torch.bool)
                    logger.debug(f"Could not find ideal SDF for {molecule}")
                    # raise ValueError(f"Could not find ideal SDF for {molecule}")
                positions.append(coordinates)
                all_has_ref_pos.append(has_ref_pos)
        return torch.cat(positions, dim=0), torch.cat(all_has_ref_pos, dim=0)

    def check_ligand_reference_conf_dims(self, data, ligand_ref_pos):
        akeys_per_molecule = data["akeys_sm"]
        all_akeys = [akey for akeys in akeys_per_molecule for akey in akeys]
        if len(all_akeys) != ligand_ref_pos.size(0):
            ligand_ref_pos = torch.zeros((len(all_akeys), 3), dtype=torch.float32)
            ligand_has_ref_pos = torch.zeros(len(all_akeys), dtype=torch.bool)
        else:
            ligand_has_ref_pos = torch.ones(ligand_ref_pos.size(0), dtype=torch.bool)
        return ligand_ref_pos, ligand_has_ref_pos

    def generate_conformer(self, molecule) -> dict:
        metadata = self.molname2sdf[molecule]
        sdf = metadata["sdf"]
        kwargs = {
            "filetype": "sdf",
            "string": True,
            "find_automorphs": False,
            "generate_conformer": True,
            "remove_H": False,
        }
        try:
            obmol, _, _, atom_coords, _ = rf2aa.data.parsers.parse_mol(sdf, **kwargs)
        except Exception:
            # fall back to using ideal coordinates in file
            kwargs["generate_conformer"] = False
            obmol, _, _, atom_coords, _ = rf2aa.data.parsers.parse_mol(sdf, **kwargs)
        # remove hydrogens and leaving atoms
        is_h = torch.tensor([atom_id[0] == "H" for atom_id in metadata["atom_id"]])
        is_leaving = torch.tensor(metadata["leaving"])
        is_h_or_leaving = is_h | is_leaving
        atom_coords = atom_coords[0]
        atom_coords = atom_coords[~is_h_or_leaving]
        return atom_coords, obmol


class AddRefAtomNameChars(Transform):
    def check_input(self, data: dict[str, Any]):
        check_contains_keys(data, ["seq_poly", "akeys_sm", "seq_combined"])

    def forward(self, data: dict[str, Any], *args, **kwargs) -> dict[str, Any]:
        seq_poly = data["seq_poly"]
        protein_encoded_names = self._compute_protein_atom_names(seq_poly)
        if len(data["akeys_sm"]) > 0:
            ligand_encoded_names = self._compute_ligand_atom_names(data["akeys_sm"])
            encoded_names = torch.cat(
                [protein_encoded_names, ligand_encoded_names], dim=0
            )
        else:
            encoded_names = protein_encoded_names

        is_real_atom = ChemData().heavyatom_mask[data["seq_combined"]]

        encoded_names_atom36 = torch.zeros(is_real_atom.shape + (4,), dtype=torch.int16)
        encoded_names_atom36[is_real_atom] = encoded_names.to(torch.int16)
        data["ref_atom_name_chars"] = encoded_names_atom36
        return data

    def _compute_protein_atom_names(self, seq_poly: torch.Tensor) -> torch.Tensor:
        atom_names = []
        for res in seq_poly:
            poly_atom_names = ChemData().aa2long[res][: ChemData().NHEAVY]
            for name in poly_atom_names:
                if name is not None:
                    encoded_name = torch.tensor(
                        [ord(char) - 32 for char in name.strip()]
                    )
                    encoded_name = F.pad(encoded_name, (0, 4 - encoded_name.size(0)))
                    atom_names.append(encoded_name[None])

        return torch.cat(atom_names, dim=0)

    def _compute_ligand_atom_names(self, akeys_sm: list) -> torch.Tensor:
        atom_names = []
        for ligand in akeys_sm:
            for key in ligand:
                ch_letter, res_num, res_name, atom_name = key
                encoded_name = torch.tensor(
                    [ord(char) - 32 for char in atom_name.strip()]
                )
                encoded_name = F.pad(encoded_name, (0, 4 - encoded_name.size(0)))
                atom_names.append(encoded_name[None])
        return torch.cat(atom_names, dim=0)


class GetReferenceCharge(AddReferencePositions):
    def check_input(self, data: dict):
        check_contains_keys(data, ["seq_poly", "akeys_sm", "seq_combined"])

    def forward(self, data: dict) -> dict:
        self.molname2sdf = self._load_ligand_ideal_sdfs()
        seq_poly = data["seq_poly"]
        seq_3letter = [ChemData().num2aa[aa] for aa in seq_poly]
        Ls_per_poly_residue = [sum(ChemData().heavyatom_mask[aa]) for aa in seq_poly]
        # get charges for protein and ligands
        protein_charges = self.generate_reference_charge(
            seq_3letter, Ls_per_poly_residue
        )
        ligand_charges = self.generate_reference_charge(
            data["lig_names"], data["Ls_sm"]
        )
        # combine them
        if len(data["lig_names"]) == 0:
            charges = protein_charges
        else:
            charges = torch.cat([protein_charges, ligand_charges], dim=0)

        # reorient to token indexing to allow cropping down the line
        is_real_atom = ChemData().heavyatom_mask[data["seq_combined"]]
        charges_atom36 = torch.zeros(is_real_atom.shape, dtype=charges.dtype)
        try:
            charges_atom36[is_real_atom] = charges
        except Exception as e:
            print(f"Error in adding reference charges: {e}")

        data["ref_charge"] = charges_atom36
        return data

    def generate_reference_charge(self, molecule_names: list, Ls) -> dict:
        if len(molecule_names) == 0:
            return None
        charges = []
        for i, molecule in enumerate(molecule_names):
            for molecule_3letter in molecule.split("_"):
                if molecule_3letter == "UNK":
                    NUM_ATOMS_IN_UNK = 5
                    charge = torch.zeros(NUM_ATOMS_IN_UNK, dtype=torch.float32)
                elif molecule_3letter in self.molname2sdf:
                    charge = self.generate_charge(molecule_3letter)
                else:
                    charge = torch.zeros(Ls[i], dtype=torch.float32)
                    logger.debug(f"Could not find ideal SDF for {molecule}")
                    # raise ValueError(f"Could not find ideal SDF for {molecule}")
                if charge.shape[0] != Ls[i]:
                    charge = torch.zeros(Ls[i], dtype=torch.float32)
                charges.append(charge)
        return torch.cat(charges, dim=0)

    def generate_charge(self, molecule, akeys=None) -> dict:
        metadata = self.molname2sdf[molecule]
        atom_coords, obmol = self.generate_conformer(molecule)
        charges = []
        atom_nums = []
        for atom in range(obmol.NumAtoms()):
            atom_idx = atom + 1
            atom = obmol.GetAtom(atom_idx)
            charges.append(atom.GetFormalCharge())
            atom_nums.append(atom.GetAtomicNum())
        atom_nums = torch.tensor(atom_nums)
        is_h = atom_nums == 1
        is_leaving = torch.tensor(metadata["leaving"])
        is_h_or_leaving = is_h | is_leaving

        charges = torch.tensor(charges)
        charges = charges[~is_h_or_leaving]

        return charges


pipeline = Compose(
    [
        AddReferencePositions(),
        AddRefAtomNameChars(),
        GetReferenceCharge(),
        ComputeResidueIndex(),
        ComputeAsymIndex(),
        ComputeSymmIndex(),
    ]
)
