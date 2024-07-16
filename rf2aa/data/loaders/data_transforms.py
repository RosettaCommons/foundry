import logging
from typing import Any
import torch
import torch.nn.functional as F
import rf2aa
from rf2aa.data_new.transforms.base import Transform, Compose
from rf2aa.data_new.transforms._checks import check_contains_keys
from rf2aa.chemical import load_pdb_ideal_sdf_strings, ChemicalData as ChemData
from functools import lru_cache

logger = logging.getLogger(__name__)


class AddReferencePositions(Transform):
    
    def check_input(self, data: dict):
        check_contains_keys(data, ["seq_poly", "lig_names", "msa_sm"])

    def forward(self, data: dict) -> dict:  
        self.molname2sdf = self._load_ligand_ideal_sdfs() 
        seq_poly = data["seq_poly"]
        seq_3letter = [ChemData().num2aa[aa] for aa in seq_poly]
        protein_ref_pos, protein_has_ref_pos = self.generate_reference_positions(seq_3letter)
        if len(data["lig_names"]) == 0:
            combined_ref_pos = protein_ref_pos
            combined_seq = seq_poly
            combined_has_ref_pos = protein_has_ref_pos
        else:
            ligand_ref_pos, ligand_has_ref_pos = self.generate_reference_positions(data["lig_names"])
            ligand_ref_pos, ligand_has_ref_pos = self.check_ligand_reference_conf_dims(data, ligand_ref_pos)
            combined_ref_pos = torch.cat([protein_ref_pos, ligand_ref_pos], dim=0)
            combined_seq = torch.cat([data["seq_poly"], data["msa_sm"][0]], dim=0).long()
            combined_has_ref_pos = torch.cat([protein_has_ref_pos, ligand_has_ref_pos], dim=0)
        is_real_atom = ChemData().heavyatom_mask[combined_seq]

        # make atom36 container for coordinates
        ideal_coords_atom36 = torch.zeros_like(is_real_atom, dtype=torch.float32)
        ideal_coords_atom36 = ideal_coords_atom36[...,None].repeat(1,1,3)
        try:
            ideal_coords_atom36[is_real_atom] = combined_ref_pos
        except Exception as e:
            logger.error(f"Error in adding reference positions: {e}")

        has_ref_pos_atom36 = torch.zeros_like(is_real_atom, dtype=torch.bool)
        has_ref_pos_atom36[is_real_atom] = combined_has_ref_pos
        data["ref_pos_atom36"]  = ideal_coords_atom36
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
                    coordinates = torch.zeros((NUM_ATOMS_IN_UNK, 3), dtype=torch.float32)
                    has_ref_pos = torch.zeros(NUM_ATOMS_IN_UNK, dtype=torch.bool)
                elif molecule_3letter in self.molname2sdf:
                    coordinates = self.generate_conformer(molecule_3letter)
                    has_ref_pos = torch.ones(coordinates.size(0), dtype=torch.bool)
                else:
                    raise ValueError(f"Could not find ideal SDF for {molecule}")
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
        kwargs = {"filetype": "sdf", "string": True, "find_automorphs": False, "generate_conformer": True, "remove_H":False}
        try:
            _, _, _, atom_coords, _ = rf2aa.data.parsers.parse_mol(sdf, 
                                                                    **kwargs 
                                                                )
        except Exception as e:
            # fall back to using ideal coordinates in file
            kwargs["generate_conformer"] = False
            _, _, _, atom_coords, _ = rf2aa.data.parsers.parse_mol(sdf, 
                                                                    **kwargs 
            ) 
        # remove hydrogens and leaving atoms
        is_h = torch.tensor([atom_id[0]=="H" for atom_id in metadata["atom_id"]])
        is_leaving = torch.tensor(metadata["leaving"])
        is_h_or_leaving = is_h | is_leaving
        atom_coords = atom_coords[0]
        atom_coords = atom_coords[~is_h_or_leaving]
        return atom_coords                                                   


class AddRefAtomNameChars(Transform):

    def check_input(self, data: dict[str, Any]):
        check_contains_keys(data, ["seq_poly", "akeys_sm", "seq_combined"]) 

    def forward(self, data: dict[str, Any], *args, **kwargs) -> dict[str, Any]:
        seq_poly = data["seq_poly"]
        protein_encoded_names = self._compute_protein_atom_names(seq_poly)
        if len(data["akeys_sm"]) > 0:
            ligand_encoded_names = self._compute_ligand_atom_names(data["akeys_sm"])
            encoded_names = torch.cat([protein_encoded_names, ligand_encoded_names], dim=0)
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
            poly_atom_names = ChemData().aa2long[res][:ChemData().NHEAVY]
            for name in poly_atom_names:
                if name is not None:
                    encoded_name = torch.tensor([ord(char) -32 for char in name.strip()])
                    encoded_name = F.pad(encoded_name, (0, 4 - encoded_name.size(0)))
                    atom_names.append(encoded_name[None])

        return torch.cat(atom_names, dim=0)
    
    def _compute_ligand_atom_names(self, akeys_sm: list) -> torch.Tensor:
        atom_names = []
        for ligand in akeys_sm:
            for key in ligand:
                ch_letter, res_num, res_name, atom_name = key
                encoded_name = torch.tensor([ord(char) - 32 for char in atom_name.strip()])
                encoded_name = F.pad(encoded_name, (0, 4 - encoded_name.size(0)))
                atom_names.append(encoded_name[None])
        return torch.cat(atom_names, dim=0)


pipeline = Compose([
                    AddReferencePositions(),
                    AddRefAtomNameChars()
                    ])