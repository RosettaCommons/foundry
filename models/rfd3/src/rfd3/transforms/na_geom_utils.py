import os
import subprocess
from datetime import datetime
from typing import Dict, Optional
import math
import numpy as np
from biotite.structure import AtomArray

from atomworks.constants import (
    STANDARD_DNA,
    STANDARD_RNA,
)

from atomworks.io.utils.sequence import (
    is_purine,
    is_pyrimidine,
)
from atomworks.ml.utils.token import (
    get_token_starts,
    is_glycine,
    is_protein_unknown,
    is_standard_aa_not_glycine,
    is_unknown_nucleotide,
)
from rfd3.transforms.hbonds_hbplus import save_atomarray_to_pdb

from atomworks.ml.encoding_definitions import AF3SequenceEncoding


DEFAULT_NA_SS_FEATURE_INFO: dict[str, int] = {
    "NA_SS_MASK": 0,
    "NA_SS_PAIR": 1,
    "NA_SS_LOOP": 2,
    "num_classes_nucleic_ss": 3,
}


# Move to function scope to avoid module-level memory retention
def _get_sequence_encoding_data():
    """Get sequence encoding data on demand to avoid persistent module-level variables."""
    sequence_encoding = AF3SequenceEncoding()
    return {
        'aa_like_res_names': sequence_encoding.all_res_names[sequence_encoding.is_aa_like],
        'rna_like_res_names': sequence_encoding.all_res_names[sequence_encoding.is_rna_like], 
        'dna_like_res_names': sequence_encoding.all_res_names[sequence_encoding.is_dna_like],
        'sequence_encoding': sequence_encoding
    }



class NucMolInfo:
    """
    Initializes constants and parameters relevant for computing nucleic acid geometry and interactions.
    """    
    def __init__(self,
                cutoff_HA_dist = 2.5,
                cutoff_DA_dist = 3.9,
        ):
        """
        Args:
            kwargs: Optional keyword arguments for customization.
        """


        # Optional parameters with default values
        # self.incl_protein = True
        self.eps =  1e-8
        # self.clamp_pairwise_params = True
        # self.use_eigennormals = kwargs.get('use_eigennormals', True)
        # self.use_all_base_atoms_for_MBD = kwargs.get('use_all_base_atoms_for_MBD', False)
        self.edges_to_compute = ['S'] # list base edges to compute, if we want to analyze WC/Hoog/etc
        self.perp_base_edge = 'S' # edge orthogonal to x- and z-directions in base frames (which is generally the sugar edge)

        self.cutoff_HA_dist = cutoff_HA_dist
        self.cutoff_DA_dist = cutoff_DA_dist
        self.seq_cutoff = 2
        self.gap_length = 200




        # Hbond interaction type inds when counting:
        self.BB_BB = 0
        self.BB_SC = 1
        self.SC_SC = 2

        self.bp_weight_BB_BB = 0.0
        self.bp_weight_BB_SC = 0.5
        self.bp_weight_SC_SC = 1.0

        self.bp_summation_weights = [self.bp_weight_BB_BB,
                                     self.bp_weight_BB_SC,
                                     self.bp_weight_SC_SC]

        self.min_hbonds_for_bp = 2.0
        self.bp_hbond_coeff    = 9.8 # determined heuristically
        self.bp_val_cutoff     = 0.5 # minimum basepairing score for binarizing basepairs when needed

        self.base_geometry_limits = {}
        self.base_geometry_limits['D_ij'] = 20.0
        self.base_geometry_limits['H_ij'] = 1.5
        self.base_geometry_limits['P_ij'] = math.pi/5
        self.base_geometry_limits['B_ij'] = math.pi/5

        # For interaction-edge classification (Watson-Crick, Hoogstein, Sugar, Base-other):
        # self.edge_to_ind = {'W':0 , 'H':1 , 'S':2 ,'B':3}
        self.rep_atom_dict={"protein": "CA", "rna": "C1'", "dna": "C1'"}

        self.has_planar_sc = {
                'ALA':   False,
                'ARG':   True,
                'ASN':   True,
                'ASP':   True,
                'CYS':   False,
                'GLN':   True,
                'GLU':   True,
                'GLY':   False,
                'HIS':   True,
                'ILE':   False,
                'LEU':   False,
                'LYS':   False,
                'MET':   False,
                'PHE':   True,
                'PRO':   False,
                'SER':   False,
                'THR':   False,
                'TRP':   True,
                'TYR':   True,
                'VAL':   False,
                'UNK':   False,
                'MAS':   False,
                'DA':    True,
                'DC':    True,
                'DG':    True,
                'DT':    True,
                'DX':    False,
                'A':     True,
                'C':     True,
                'G':     True,
                'U':     True,
                'X':     False,
                'HIS_D': True,
                }
        

        
        # Make self.planar_atom_list_dict based on known planar atoms for each residue type:
        self.planar_atom_list_dict = {
            'ALA':   [],
            'ARG':   ['NH1', 'NH2', 'CZ', 'NE', 'CD'],
            'ASN':   ['OD1', 'ND2', 'CG', 'CB'],
            'ASP':   ['OD1', 'OD2', 'CG', 'CB'],
            'CYS':   [],
            'GLN':   ['OE1', 'NE2', 'CD', 'CG'],
            'GLU':   ['OE1', 'OE2', 'CD', 'CG'],
            'GLY':   [],
            'HIS':   ['ND1', 'CE1', 'NE2', 'CD2', 'CG', 'CB'],
            'ILE':   [],
            'LEU':   [],
            'LYS':   [],
            'MET':   [],
            'PHE':   ['CZ', 'CE1', 'CE2', 'CD1', 'CD2', 'CG', 'CB'],
            'PRO':   [],
            'SER':   [],
            'THR':   [],
            'TRP':   ['CH2', 'CZ3', 'CZ2', 'CE3', 'CE2', 'CD2', 'NE1', 'CD1', 'CG', 'CB'],
            'TYR':   ['OH',  'CZ',  'CE1', 'CE2', 'CD1', 'CD2', 'CG',  'CB'],
            'VAL':   [],
            'UNK':   [],
            'MAS':   [],
            'DA':    ['N6', 'C6', 'N1', 'C2', 'N3', 'C4', 'C5', 'N7', 'C8', 'N9'],
            'DC':    ['N4', 'C4', 'N3', 'O2', 'C2', 'C5', 'C6', 'N1'],
            'DG':    ['O6', 'C6', 'N1', 'N2', 'C2', 'N3', 'C4', 'C5', 'N7', 'C8', 'N9'],
            'DT':    ['O4', 'O2', 'N3', 'C4', 'C2', 'C5', 'C6', 'N1', 'C7'],
            'DX':    [],
            'A':     ['N6', 'C6', 'N1', 'C2', 'N3', 'C4', 'C5', 'N7', 'C8', 'N9'],
            'C':     ['N4', 'C4', 'N3', 'O2', 'C2', 'C5', 'C6', 'N1'],
            'G':     ['O6', 'C6', 'N1', 'N2', 'C2', 'N3', 'C4', 'C5', 'N7', 'C8', 'N9'],
            'U':     ['O4', 'O2', 'N3', 'C4', 'C2', 'C5', 'C6', 'N1'],
            'X':     [],
            'HIS_D': ['ND1', 'CD2', 'CE1', 'NE2', 'CG', 'CB'],
            }


        # from pdb import set_trace; set_trace()

        self.nuc_resi_3letter = ["DA","DG","DC","DT","A","G","C","U"]
        self.ring_atom_list = ["N1","C2","N3","C4","C6","C5"]

        # go through self.vec_atom_dict and remove spaces from atom names (values in inner dicts), and remove spaces from keys + replace 'R' with '' in outer dict keys
        self.vec_atom_dict = {
                "DA": {"W_start":"N1", "W_stop":"N6", "H_start":"N7", "H_stop":"N6", "S_start":"C1'", "S_stop":"N3", "B_start":"C1'", "B_stop":"N9" },
                "DG": {"W_start":"N1", "W_stop":"O6", "H_start":"N7", "H_stop":"O6", "S_start":"C1'", "S_stop":"N3", "B_start":"C1'", "B_stop":"N9" },
                "DC": {"W_start":"N3", "W_stop":"N4", "H_start":"C5", "H_stop":"N4", "S_start":"C1'", "S_stop":"O2", "B_start":"C1'", "B_stop":"N1" },
                "DT": {"W_start":"N3", "W_stop":"O4", "H_start":"C5", "H_stop":"O4", "S_start":"C1'", "S_stop":"O2", "B_start":"C1'", "B_stop":"N1" },
                "A": {"W_start":"N1", "W_stop":"N6", "H_start":"N7", "H_stop":"N6", "S_start":"C1'", "S_stop":"N3", "B_start":"C1'", "B_stop":"N9" },
                "G": {"W_start":"N1", "W_stop":"O6", "H_start":"N7", "H_stop":"O6", "S_start":"C1'", "S_stop":"N3", "B_start":"C1'", "B_stop":"N9" },
                "C": {"W_start":"N3", "W_stop":"N4", "H_start":"C5", "H_stop":"N4", "S_start":"C1'", "S_stop":"O2", "B_start":"C1'", "B_stop":"N1" },
                "U": {"W_start":"N3", "W_stop":"O4", "H_start":"C5", "H_stop":"O4", "S_start":"C1'", "S_stop":"O2", "B_start":"C1'", "B_stop":"N1" },
            }



        self.atom_region_dict = {
                'ALA': {'bb':('N','CA','C','O'), 
                        'sc':('CB')},
                'ARG': {'bb':('N','CA','C','O'), 
                        'sc':('CB','CG','CD','NE','CZ','NH1','NH2')},
                'ASN': {'bb':('N','CA','C','O'), 
                        'sc':('CB','CG','OD1','ND2')},
                'ASP': {'bb':('N','CA','C','O'), 
                        'sc':('CB','CG','OD1','OD2')},
                'CYS': {'bb':('N','CA','C','O'), 
                        'sc':('CB','SG')},
                'GLN': {'bb':('N','CA','C','O'), 
                        'sc':('CB','CG','CD','OE1','NE2')},
                'GLU': {'bb':('N','CA','C','O'), 
                        'sc':('CB','CG','CD','OE1','OE2')},
                'GLY': {'bb':('N','CA','C','O'), 
                        'sc':()},
                'HIS': {'bb':('N','CA','C','O'), 
                        'sc':('CB','CG','ND1','CD2','CE1','NE2')},
                'ILE': {'bb':('N','CA','C','O'), 
                        'sc':('CB','CG1','CG2','CD1')},
                'LEU': {'bb':('N','CA','C','O'), 
                        'sc':('CB','CG','CD1','CD2')},
                'LYS': {'bb':('N','CA','C','O'), 
                        'sc':('CB','CG','CD','CE','NZ')},
                'MET': {'bb':('N','CA','C','O'), 
                        'sc':('CB','CG','SD','CE')},
                'PHE': {'bb':('N','CA','C','O'), 
                        'sc':('CB','CG','CD1','CD2','CE1','CE2','CZ')},
                'PRO': {'bb':('N','CA','C','O'), 
                        'sc':('CB','CG','CD')},
                'SER': {'bb':('N','CA','C','O'), 
                        'sc':('CB','OG')},
                'THR': {'bb':('N','CA','C','O'), 
                        'sc':('CB','OG1','CG2')},
                'TRP': {'bb':('N','CA','C','O'), 
                        'sc':('CB','CG','CD1','CD2','CE2','CE3','NE1','CZ2','CZ3','CH2')},
                'TYR': {'bb':('N','CA','C','O'), 
                        'sc':('CB','CG','CD1','CD2','CE1','CE2','CZ','OH')},
                'VAL': {'bb':('N','CA','C','O'), 
                        'sc':('CB','CG1','CG2')},
                'UNK': {'bb':('N','CA','C','O'), 
                        'sc':('CB')},
                'MAS': {'bb':('N','CA','C','O'), 
                        'sc':('CB')},
                'DA': {'bb':("O4'", "C1'", "C2'",'OP1','P','OP2', "O5'", "C5'", "C4'", "C3'", "O3'"), 
                        'sc':('N9','C4','N3','C2','N1','C6','C5','N7','C8','N6')},
                'DC': {'bb':("O4'", "C1'", "C2'",'OP1','P','OP2', "O5'", "C5'", "C4'", "C3'", "O3'"), 
                        'sc':('N1','C2','O2','N3','C4','N4','C5','C6')},
                'DG': {'bb':("O4'", "C1'", "C2'",'OP1','P','OP2', "O5'", "C5'", "C4'", "C3'", "O3'"), 
                        'sc':('N9','C4','N3','C2','N1','C6','C5','N7','C8','N2','O6')},
                'DT': {'bb':("O4'", "C1'", "C2'",'OP1','P','OP2', "O5'", "C5'", "C4'", "C3'", "O3'"), 
                        'sc':('N1','C2','O2','N3','C4','O4','C5','C7','C6')},
                'DX': {'bb':("O4'", "C1'", "C2'",'OP1','P','OP2', "O5'", "C5'", "C4'", "C3'", "O3'"), 
                        'sc':()},
                'A': {'bb':("O4'", "C1'", "C2'",'OP1','P','OP2', "O5'", "C5'", "C4'", "C3'", "O3'", "O2'"), 
                    'sc':('N1','C2','N3','C4','C5','C6','N6','N7','C8','N9')},
                'C': {'bb':("O4'", "C1'", "C2'",'OP1','P','OP2', "O5'", "C5'", "C4'", "C3'", "O3'", "O2'"), 
                    'sc':('N1','C2','O2','N3','C4','N4','C5','C6')},
                'G': {'bb':("O4'", "C1'", "C2'",'OP1','P','OP2', "O5'", "C5'", "C4'", "C3'", "O3'", "O2'"), 
                    'sc':('N1','C2','N2','N3','C4','C5','C6','O6','N7','C8','N9')},
                'U': {'bb':("O4'", "C1'", "C2'",'OP1','P','OP2', "O5'", "C5'", "C4'", "C3'", "O3'", "O2'"), 
                    'sc':('N1','C2','O2','N3','C4','O4','C5','C6')},
                'X': {'bb':("O4'", "C1'", "C2'",'OP1','P','OP2', "O5'", "C5'", "C4'", "C3'", "O3'", "O2'"), 
                    'sc':()},
                'HIS_D': {'bb':('N','CA','C','O'), 
                        'sc':('CB','CG','NE2','CD2','CE1','ND1')},
                }


        self.aa_planar_atoms = ['NH1', 'NH2', 'CZ',  'NE',  'OD1', 'ND2', 
                                'OD2', 'OE1', 'NE2', 'CD',  'OE2', 'ND1', 
                                'CD2', 'CE1', 'CD1', 'CE2', 'NE1', 'CZ2', 
                                'CZ3', 'CH2', 'CE3', 'OH',   'CG',  'CB',]

        self.na_planar_atoms = ['C4', 'N3', 'C2', 'C6', 'C5', 'N7', 'C8', 
                                'N6', 'O2', 'N4', 'N2', 'O6', 'O4', 'C7', 
                                'N9', 'N1']

def find_planar_positions(
                atom_array: AtomArray, 
                mol_info: NucMolInfo,
                tol: float = 1e-2,
                ) -> Dict:
    """
    Finds residues with planar sidechains based on four tip-most atoms,
    but also checks for valid atoms to use for this type of calculation.

    Returns:
    dict of planar atom lists
    """
    unique_positions_list = []
    for atm in atom_array:
        pos_id = (atm.chain_iid, atm.res_id, atm.res_name)
        if pos_id not in unique_positions_list:
            unique_positions_list.append(pos_id)

    # Get candidate planar atoms:
    planar_atom_list_dict = {}

    # for chain_iid, res_id in unique_positions_list:
    for chain_iid, res_id, res_name in unique_positions_list:

        mask = (
            (atom_array.chain_iid == chain_iid) &
            (atom_array.res_id == res_id) & 
            (atom_array.res_name == res_name)
        )
        res_atoms = atom_array[mask]

        # If possible, speed up by using known planar atoms for this residue type:
        if res_name in mol_info.planar_atom_list_dict.keys():
            # Shared atoms between residue and known planar atoms for that residue type:
            planar_atom_list = list(
                set([atm.atom_name for atm in res_atoms]) & 
                set(mol_info.planar_atom_list_dict[res_name])
                )
            planar_atom_list_dict[(chain_iid, res_id)] = planar_atom_list

        # If unknown or noncanonical residue, compute planar atoms geometrically:
        else:
            candidate_planar_atm_names = []
            candidate_planar_atm_coords = []

            for atm in res_atoms:
                # Can pre-filter protein planar atoms:
                if atm.is_protein and (atm.atom_name in mol_info.aa_planar_atoms):
                    candidate_planar_atm_names.append(atm.atom_name)
                    candidate_planar_atm_coords.append(atm.coord)
                # Can pre-filter nucleic acid planar atoms:
                elif (atm.is_rna or atm.is_dna) and (atm.atom_name in mol_info.na_planar_atoms):
                    candidate_planar_atm_names.append(atm.atom_name)
                    candidate_planar_atm_coords.append(atm.coord)
                # Otherwise, consider all atoms for plane fitting:
                else:
                    candidate_planar_atm_names.append(atm.atom_name)
                    candidate_planar_atm_coords.append(atm.coord)

            # reverse order to prioritize atoms further away from bb:
            candidate_planar_atm_names = list(reversed(candidate_planar_atm_names))
            candidate_planar_atm_coords = list(reversed(candidate_planar_atm_coords))

            # Use first four candidate atoms only to define the plane:
            if len(candidate_planar_atm_coords) >= 4:
                coords = np.asarray(candidate_planar_atm_coords, dtype=np.float32)

                # compute 4-atom based plane:
                quad_coords = coords[:4, :]

                # fit plane via PCA (use smallest‑variance eigenvector as normal)
                quad_center = quad_coords.mean(axis=0, keepdims=True)
                all_quad_centered = coords - quad_center
                quad_centered = quad_coords - quad_center
                # covariance matrix
                quad_cov = (quad_centered.T @ quad_centered) / max(quad_coords.shape[0] - 1, 1)
                # eigen decomposition
                _, quad_eigvecs = np.linalg.eigh(quad_cov)
                quad_normal = quad_eigvecs[:, 0]  # eigenvector with smallest eigenvalue
                quad_normal = quad_normal / (np.linalg.norm(quad_normal) + 1e-8)
                # compute distances from plane for all candidate atoms
                quad_dists = np.abs(all_quad_centered @ quad_normal)
                # keep only atoms within tolerance
                quad_valid_mask = quad_dists <= tol

                # Filter for if we have a valid plane in the first place:
                valid_plane_filter = (np.nanmax(quad_dists[:4]) < tol)
                # Filter for if we have enough atoms in the plane:
                plane_atom_filter = (int(np.sum(quad_valid_mask)) >= 4)
                if valid_plane_filter and plane_atom_filter:
                    # Set the planar atom list for this position to those that are within tol of the plane: 
                    # using quad_valid_mask and candidate_planar_atm_names:
                    planar_atom_list = [n for n, keep in zip(candidate_planar_atm_names, quad_valid_mask.tolist()) if keep]
                
                # not enough atoms close to a common plane
                else:
                    planar_atom_list = []

            else:

                # need at least 4 atoms to define a robust plane
                planar_atom_list = []
                
            planar_atom_list_dict[(chain_iid, res_id)] = planar_atom_list


    return planar_atom_list_dict




def calculate_hb_counts(
    atom_array: AtomArray,
    token_level_data: dict,
    mol_info: NucMolInfo,
    cutoff_HA_dist: float = 2.5,
    cutoff_DA_dist: float = 3.9,
    ):
    """
    Compute hbond counts between residues and return an (L, L, 3) 
      numpy array where the last dimension encodes:
        0 -> both backbone (BB-BB)
        1 -> one backbone, one sidechain (BB-SC)
        2 -> both sidechain (SC-SC)
    """
    dtstr = datetime.now().strftime("%Y%m%d%H%M%S")
    pdb_path = f"{dtstr}_{np.random.randint(10000)}.pdb"

    atom_array, nan_mask, chain_map = save_atomarray_to_pdb(atom_array, pdb_path)
    subprocess.call(
        [
            "/projects/ml/hbplus",
            "-h",
            str(cutoff_HA_dist),
            "-d",
            str(cutoff_DA_dist),
            pdb_path,
            pdb_path,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


    num_resis_total = len(token_level_data["token_id_list"])

    hbond_count = np.zeros((num_resis_total, num_resis_total, 3), dtype=np.int32)

    hb2_path = pdb_path.replace("pdb", "hb2")
    with open(hb2_path, "r") as hb2_f:
        for i, line in enumerate(hb2_f):
            if i < 8:
                continue
            if len(line) < 28:
                continue

            # Initialize donor/acceptor sidechain/backbone flags:
            # then replace with True if valid for summation
            d_is_sc = False
            d_is_bb = False
            a_is_sc = False
            a_is_bb = False

            d_chain_iid = chain_map[line[0]]
            d_resi = int(line[1:5].strip())
            d_resn = line[6:9].strip()
            d_atom_name = line[9:13].strip()

            d_mask = (
                (atom_array.atom_name == d_atom_name)
                & (atom_array.res_name == d_resn)
                & (atom_array.res_id == d_resi)
                & (atom_array.chain_iid == d_chain_iid)
            )
            d_atm = atom_array[d_mask]
            d_idx = d_atm.token_id

            # Handle standard polymer residues for donor atom:
            if d_resn in mol_info.atom_region_dict.keys():
                d_is_sc = (d_atom_name in mol_info.atom_region_dict[d_resn]['sc'])
                d_is_bb = (d_atom_name in mol_info.atom_region_dict[d_resn]['bb'])
            else:
                # If non-polymer, define any ligand HBonding atom as backbone:
                if d_mask.sum() > 0:
                    d_is_bb = atom_array[d_mask][0].is_ligand

            a_chain_iid = chain_map[line[14]]
            a_resi = int(line[15:19].strip())
            a_resn = line[20:23].strip()
            a_atom_name = line[23:27].strip()

            a_mask = (
                (atom_array.atom_name == a_atom_name)
                & (atom_array.res_name == a_resn)
                & (atom_array.res_id == a_resi)
                & (atom_array.chain_iid == a_chain_iid)
            )
            a_atm = atom_array[a_mask]
            a_idx = a_atm.token_id

            # Handle standard polymer residues for acceptor atom:
            if a_resn in mol_info.atom_region_dict.keys():
                a_is_sc = (a_atom_name in mol_info.atom_region_dict[a_resn]['sc'])
                a_is_bb = (a_atom_name in mol_info.atom_region_dict[a_resn]['bb'])
            else:
                # If non-polymer, define any ligand HBonding atom as backbone:
                if a_mask.sum() > 0:
                    a_is_bb = atom_array[a_mask][0].is_ligand

            # 0 -> both backbone (BB-BB)
            hbond_count[a_idx, d_idx, 0] += (a_is_bb * d_is_bb)
            hbond_count[d_idx, a_idx, 0] += (d_is_bb * a_is_bb)

            # 1 -> one backbone, one sidechain (BB-SC)
            hbond_count[a_idx, d_idx, 1] += (a_is_bb * d_is_sc) | (a_is_sc * d_is_bb)
            hbond_count[d_idx, a_idx, 1] += (d_is_bb * a_is_sc) | (d_is_sc * a_is_bb)

            # 2 -> both sidechain (SC-SC)
            hbond_count[a_idx, d_idx, 2] += (a_is_sc * d_is_sc)
            hbond_count[d_idx, a_idx, 2] += (d_is_sc * a_is_sc)

    os.remove(pdb_path)
    os.remove(hb2_path)

    return hbond_count




def make_coord_list(atom_array: AtomArray, 
                    residue_list: list[str], 
                    chain_list: list[str],  
                    atom_list: list[str],
                    ) -> list[list[str]]:
    """
    Given an atom array, and lists of residues, chains, and atom names,
    return a list of coordinates for the specified atoms in the specified residues and chains.
    If the atom is not found, return [NaN, NaN, NaN] for that atom.
    The the three input lists must be of the same length, and the output list will have the same length as well.
    Args:
        atom_array: BioTite atom_array object
        residue_list: list of residue names to consider
        chain_list: list of chain identifiers to consider
        atom_list: list of atom names to extract coordinates for
    Returns:
        coord_list: list of lists of coordinates for the specified atoms

    """
    coord_list = []
    for res_id, chain_id, atom_name in zip(residue_list, chain_list, atom_list):

        # Check if the residue exists in the atom array
        if atom_name == "atomized":
            # Check for atomized residue, in which case we take the first atom of the residue
            # full mask should be length-1 if atomized
            mask = (
                (atom_array.chain_id == chain_id) & 
                (atom_array.res_id == res_id)
                )
        else:
            # General case for non-atomized residues
            # should have a unique solution, but we take the first entry either way.
            mask = (
                (atom_array.chain_id == chain_id) & 
                (atom_array.res_id == res_id) & 
                (atom_array.atom_name == atom_name)
                )
            
        # Get the coordinates for the masked atoms
        coords = atom_array.coord[mask][0:1]

        if len(coords) < 1:
            coord_list.append([float("nan"), float("nan"), float("nan")])
        else:
            coord_list.append(coords[0].tolist())

    return coord_list


def get_token_level_metadata(
    atom_array: AtomArray,
    mol_info: "NucMolInfo",
    *,
    NA_only: bool = False,
    planar_only: bool = True,
) -> dict:
    """Lightweight token-level metadata.

    This intentionally avoids expensive coordinate-derived computations
    (e.g., planar plane-fitting and geometry coordinate extraction).

    It is sufficient for:
    - SS reconstruction / loop labeling from ``bp_partners``
    - inference-time SS specification parsing

    If you later need geometry keys (``xyz_planar``, ``frame_xyz``, ``M_i``),
    call :func:`add_token_level_geometry_data`.
    """

    token_starts = get_token_starts(atom_array)
    token_level_array = atom_array[token_starts]

    token_index = np.arange(len(token_starts))

    # molecule type flags
    seq_data = _get_sequence_encoding_data()
    is_protein = np.isin(token_level_array.res_name, seq_data["aa_like_res_names"])
    is_rna = np.isin(token_level_array.res_name, seq_data["rna_like_res_names"])
    is_dna = np.isin(token_level_array.res_name, seq_data["dna_like_res_names"])
    del seq_data

    is_na_arr = (is_dna | is_rna).astype(bool)

    chain_list: list[str] = []
    chain_iid_list: list[str] = []
    resi_list: list[int] = []
    ind_list: list[int] = []
    res_name_list: list[str] = []
    token_id_list: list[str] = []

    rep_atom_list: list[str | None] = []
    S_start_atom_list: list[str | None] = []
    S_stop_atom_list: list[str | None] = []
    sc_planarity_list: list[bool] = []

    for i, atm in enumerate(token_level_array):
        chain_list.append(atm.chain_id)
        chain_iid_list.append(atm.chain_iid)
        resi_list.append(int(atm.res_id))
        ind_list.append(int(i))
        res_name_list.append(atm.res_name)
        token_id_list.append(str(atm.token_id))

        if atm.is_polymer and (atm.res_name in mol_info.has_planar_sc.keys()):
            sc_planarity_list.append(bool(mol_info.has_planar_sc[atm.res_name]))
        else:
            sc_planarity_list.append(False)

        # representative & sugar-edge atoms
        if (is_glycine(atm.res_name) | is_protein_unknown(atm.res_name)):
            rep_atom_i = "CA"
            S_start_atom_i = None
            S_stop_atom_i = None
        elif is_standard_aa_not_glycine(atm.res_name):
            rep_atom_i = "CA"
            S_start_atom_i = "CA"
            S_stop_atom_i = "CB"
        elif is_pyrimidine(atm.res_name):
            rep_atom_i = "C1'"
            S_start_atom_i = "C1'"
            S_stop_atom_i = "O2"
        elif is_purine(atm.res_name):
            rep_atom_i = "C1'"
            S_start_atom_i = "C1'"
            S_stop_atom_i = "N3"
        elif is_unknown_nucleotide(atm.res_name):
            rep_atom_i = "C1'"
            S_start_atom_i = None
            S_stop_atom_i = None
        elif getattr(atm, "atomize", False):
            rep_atom_i = atm.atom_name
            S_start_atom_i = None
            S_stop_atom_i = None
        else:
            rep_atom_i = None
            S_start_atom_i = None
            S_stop_atom_i = None

        rep_atom_list.append(rep_atom_i)
        S_start_atom_list.append(S_start_atom_i)
        S_stop_atom_list.append(S_stop_atom_i)

    # residue index <-> token index map
    resi2index = {f"{c}__{r}": i for c, r, i in zip(chain_iid_list, resi_list, ind_list)}

    # relative sequence positions w/ chain gaps
    rel_pos_list: list[int] = []
    current_chain = ""
    chn_bias = -mol_info.gap_length
    for r, c in zip(resi_list, chain_iid_list):
        if c != current_chain:
            chn_bias += mol_info.gap_length
            current_chain = c
        rel_pos_list.append(int(r + chn_bias))

    rel_pos = np.asarray(rel_pos_list, dtype=np.int64)
    seq_neighbors = (
        np.abs(rel_pos[:, None] - rel_pos[None, :]) <= int(mol_info.seq_cutoff)
    )

    na_inds = np.nonzero(is_na_arr)[0].tolist()
    na_tensor_inds = {na_i: i for i, na_i in enumerate(na_inds)}

    # Cheap planarity heuristic from residue name lookup
    is_planar_arr = np.asarray(sc_planarity_list, dtype=bool)

    # filter mask using NA_only / planar_only flags
    if NA_only and planar_only:
        filter_mask = is_na_arr & is_planar_arr
    elif NA_only and (not planar_only):
        filter_mask = is_na_arr.copy()
    elif (not NA_only) and planar_only:
        filter_mask = is_planar_arr.copy()
    else:
        filter_mask = np.ones_like(is_na_arr, dtype=bool)

    return {
        "token_starts": token_starts,
        "token_index": token_index,
        "is_na": is_na_arr,
        "is_planar": is_planar_arr,
        "chain_list": chain_list,
        "chain_iid_list": chain_iid_list,
        "resi_list": resi_list,
        "resn_list": res_name_list,
        "token_id_list": token_id_list,
        "resi2index": resi2index,
        "len_s": int(len(token_level_array)),
        "seq_neighbors": seq_neighbors,
        "na_inds": na_inds,
        "na_tensor_inds": na_tensor_inds,
        "filter_mask": filter_mask,
        "rep_atom_list": rep_atom_list,
        "S_start_atom_list": S_start_atom_list,
        "S_stop_atom_list": S_stop_atom_list,
        "include_geometry": False,
    }


def add_token_level_geometry_data(
    atom_array: AtomArray,
    mol_info: "NucMolInfo",
    token_level_data: dict,
    *,
    NA_only: bool = False,
    planar_only: bool = True,
) -> dict:
    """Augment a metadata-only token_level_data dict with geometry fields.

    Populates:
    - xyz_planar, xyz_S_start, xyz_S_stop
    - frame_xyz, M_i
    - updates is_planar and filter_mask using coordinate-derived planarity
    - sets include_geometry=True
    """

    if bool(token_level_data.get("include_geometry", False)):
        return token_level_data

    # Backward-compatibility: older token_level_data dicts (or user-provided ones)
    # may not contain the metadata keys this function needs.
    required_keys = (
        "chain_iid_list",
        "chain_list",
        "resi_list",
        "rep_atom_list",
        "S_start_atom_list",
        "S_stop_atom_list",
        "is_na",
    )
    if any(k not in token_level_data for k in required_keys):
        token_level_data = get_token_level_metadata(
            atom_array,
            mol_info,
            NA_only=NA_only,
            planar_only=planar_only,
        )

    chain_iid_list: list[str] = token_level_data["chain_iid_list"]
    chain_list: list[str] = token_level_data["chain_list"]
    resi_list: list[int] = token_level_data["resi_list"]
    rep_atom_list: list[str | None] = token_level_data["rep_atom_list"]
    S_start_atom_list: list[str | None] = token_level_data["S_start_atom_list"]
    S_stop_atom_list: list[str | None] = token_level_data["S_stop_atom_list"]

    planar_atom_list_dict = find_planar_positions(atom_array, mol_info)
    has_planar_sc: list[bool] = []

    xyz_planar: list[list[list[float]]] = []
    xyz_S_start: list[list[float]] = []
    xyz_S_stop: list[list[float]] = []

    for c, r, S_start_atm, S_stop_atm in zip(
        chain_iid_list,
        resi_list,
        S_start_atom_list,
        S_stop_atom_list,
    ):
        planar_atoms_i = planar_atom_list_dict[(c, r)]
        has_planar_sc.append(bool(len(planar_atoms_i) >= 4))

        atom_array_i = atom_array[(atom_array.chain_iid == c) & (atom_array.res_id == r)]

        planar_coords_i: list[list[float]] = []
        for pl_atm_name_j in planar_atoms_i:
            pl_atom_array_ij = atom_array_i[atom_array_i.atom_name == pl_atm_name_j]
            if len(pl_atom_array_ij) == 0:
                planar_coords_i.append([float("nan"), float("nan"), float("nan")])
            else:
                planar_coords_i.append(pl_atom_array_ij[0].coord)

        xyz_planar.append(planar_coords_i if len(planar_coords_i) > 3 else [[float("nan")] * 3])

        if S_start_atm is None:
            xyz_S_start.append([float("nan"), float("nan"), float("nan")])
        else:
            S_start_atom_array_i = atom_array_i[atom_array_i.atom_name == S_start_atm]
            xyz_S_start.append(
                [float("nan"), float("nan"), float("nan")]
                if len(S_start_atom_array_i) == 0
                else S_start_atom_array_i[0].coord
            )

        if S_stop_atm is None:
            xyz_S_stop.append([float("nan"), float("nan"), float("nan")])
        else:
            S_stop_atom_array_i = atom_array_i[atom_array_i.atom_name == S_stop_atm]
            xyz_S_stop.append(
                [float("nan"), float("nan"), float("nan")]
                if len(S_stop_atom_array_i) == 0
                else S_stop_atom_array_i[0].coord
            )

        del atom_array_i

    # frame coordinates and backbone direction
    frame_xyz = np.asarray(
        make_coord_list(atom_array, resi_list, chain_list, rep_atom_list),
        dtype=np.float32,
    )

    padded_centers = np.concatenate([frame_xyz[:1], frame_xyz, frame_xyz[-1:]], axis=0)
    M_i = (
        (padded_centers[1:-1] - padded_centers[:-2])
        + (padded_centers[2:] - padded_centers[1:-1])
    ) / 2.0

    is_planar_arr = np.asarray(has_planar_sc, dtype=bool)
    token_level_data["is_planar"] = is_planar_arr

    is_na_arr = np.asarray(token_level_data["is_na"], dtype=bool)
    if NA_only and planar_only:
        filter_mask = is_na_arr & is_planar_arr
    elif NA_only and (not planar_only):
        filter_mask = is_na_arr.copy()
    elif (not NA_only) and planar_only:
        filter_mask = is_planar_arr.copy()
    else:
        filter_mask = np.ones_like(is_na_arr, dtype=bool)
    token_level_data["filter_mask"] = filter_mask

    token_level_data.update(
        {
            "xyz_planar": xyz_planar,
            "xyz_S_start": xyz_S_start,
            "xyz_S_stop": xyz_S_stop,
            "frame_xyz": frame_xyz,
            "M_i": M_i,
            "include_geometry": True,
        }
    )

    del planar_atom_list_dict, padded_centers
    return token_level_data


def _compute_nucleic_ss_impl(
    mol_info, 
    token_level_data,
    hbond_count,
    clamp_pairwise_params=True,
    eps=1e-8,
    *,
    return_local_params: bool = False,
    return_pairwise_geometry: bool = False,
    return_opening_angle: bool = False,
    return_basepairs_only: bool = False,
):
    """
    Compute nucleic secondary structure–related quantities and pairwise base params.

        Notes
        -----
        This function is used in two modes:

        - Fast annotation mode (default): computes only what is needed to derive
            ``basepairs_bool_ij`` and does *not* retain large intermediate pairwise
            geometry arrays (X_ij/Y_ij/Z_ij/O_ij).
        - Diagnostic mode: set ``return_pairwise_geometry=True`` (and optionally
            ``return_local_params=True`` / ``return_opening_angle=True``) to also
            return additional geometry arrays.
    """

    mask_1d = np.asarray(token_level_data["filter_mask"], dtype=bool)
    len_mask = int(mask_1d.sum())
    # len_full = len(mask_1d)
    
    # unpack 1D data from token_level_data and apply filters
    M_i         = np.asarray(token_level_data["M_i"], dtype=np.float32)[mask_1d]
    frame_xyz   = np.asarray(token_level_data["frame_xyz"], dtype=np.float32)[mask_1d]
    
    is_na       = np.asarray(token_level_data["is_na"], dtype=bool)[mask_1d]
    
    xyz_S_start = [xyz_list_i for xyz_list_i, keep_i in zip(token_level_data["xyz_S_start"], mask_1d) if keep_i]
    xyz_S_stop  = [xyz_list_i for xyz_list_i, keep_i in zip(token_level_data["xyz_S_stop"], mask_1d) if keep_i]
    xyz_planar  = [xyz_list_i for xyz_list_i, keep_i in zip(token_level_data["xyz_planar"], mask_1d) if keep_i]
    

    # unpack 2D data from token_level_data and apply filters
    hbond_count = np.asarray(hbond_count)[mask_1d, :][:, mask_1d]
    seq_neighbors  = np.asarray(token_level_data["seq_neighbors"], dtype=bool)[mask_1d, :][:, mask_1d]

    # --- CALC 0: precompute displacement vectors / distances ----
    planar_centers = np.stack(
        [
            np.nanmean(np.asarray(xyz_i, dtype=np.float32), axis=0)
            for xyz_i in xyz_planar
        ],
        axis=0,
    ).astype(np.float32)

        
    frame_D_ij_vec = frame_xyz[None, :, :] - frame_xyz[:, None, :]  # [L, L, 3]
    sc_D_ij_vec = planar_centers[None, :, :] - planar_centers[:, None, :]   # [L, L, 3]
    # D_ij = frame_D_ij_vec.norm(dim=-1)                                # [L, L]


    # --- CALC I: local base params (canonical frames) ------------
    centered_points = [
        np.asarray(xyz_i, dtype=np.float32) - cen_i
        for xyz_i, cen_i in zip(xyz_planar, planar_centers)
    ]

    # eigenvectors per residue: [L, 3, 3] (NaNs where invalid)
    eigenvectors = np.full((len_mask, 3, 3), np.nan, dtype=np.float32)

    for i, xyz_i in enumerate(centered_points):
        xyz_i = xyz_i[~np.isnan(xyz_i).any(axis=1)]
        if xyz_i.shape[0] >= 3:
            cov_matrix = np.einsum("ij,ik->jk", xyz_i, xyz_i) / max(
                xyz_i.shape[0] - 1, 1
            )
            _, eigvecs = np.linalg.eigh(cov_matrix)
            eigenvectors[i] = eigvecs


    # base-normal (principal) direction N_i, then corrected Z_i
    N_i = eigenvectors[:, :, 0]
    N_i = N_i / (np.linalg.norm(N_i, axis=1, keepdims=True) + eps)

    Z_i = N_i * np.sum(M_i * N_i, axis=-1, keepdims=True)
    Z_i = Z_i / (np.linalg.norm(Z_i, axis=-1, keepdims=True) + eps)

    # Only compute full local frames when requested.
    # Basepair filters only need Z_i (via Z_ij) and do not require X_i/Y_i.
    local_base_params = None
    if return_local_params or return_opening_angle:
        # Sugar-edge vectors X_s_i built from S_start/stop
        X_s_i = (
            np.asarray(xyz_S_stop, dtype=np.float32)
            - np.asarray(xyz_S_start, dtype=np.float32)
        )
        X_s_i = X_s_i / (np.linalg.norm(X_s_i, axis=-1, keepdims=True) + eps)

        X_i = np.cross(Z_i, X_s_i)
        X_i = X_i / (np.linalg.norm(X_i, axis=-1, keepdims=True) + eps)

        if return_local_params:
            Y_i = np.cross(X_i, Z_i)
            Y_i = Y_i / (np.linalg.norm(Y_i, axis=-1, keepdims=True) + eps)
            local_base_params = {"X_i": X_i, "Y_i": Y_i, "Z_i": Z_i}
        else:
            # Opening needs X_i but not the local params dict.
            local_base_params = None

    # --- CALC II: pairwise base parameters -----------------------

    # stack mean Z-direction vectors for parallel (0) and antiparallel (1)
    Z_sum = Z_i[:, None, :] + Z_i[None, :, :]
    Z_diff = Z_i[:, None, :] - Z_i[None, :, :]
    Z_ij_oris = 0.5 * np.stack((Z_sum, Z_diff), axis=0)  # [2, L, L, 3]

    base_ori_ij = (
        np.linalg.norm(Z_ij_oris[1], axis=-1) > np.linalg.norm(Z_ij_oris[0], axis=-1)
    ).astype(np.int64)  # [L, L]

    Z_ij = np.where(base_ori_ij[..., None] == 0, Z_ij_oris[0], Z_ij_oris[1])
    Z_ij = Z_ij / (np.linalg.norm(Z_ij, axis=-1, keepdims=True) + eps)

    Y_ij = frame_D_ij_vec / (np.linalg.norm(frame_D_ij_vec, axis=-1, keepdims=True) + eps)
    X_ij = np.cross(Z_ij, Y_ij)
    X_ij = X_ij / (np.linalg.norm(X_ij, axis=-1, keepdims=True) + eps)

    # vertical displacement using sidechain centroids
    H_ij = np.sum(sc_D_ij_vec * Z_ij, axis=-1)
    # H_ij_vec = H_ij[..., None] * Z_ij

    # Opening (O_ij) is purely diagnostic; compute only if requested.
    O_ij = None
    if return_opening_angle:
        if not (return_local_params or return_opening_angle):
            raise RuntimeError("Internal error: opening angle requested without local frame")

        proj_X_i_XY = (
            np.sum(X_i[:, None, :] * X_ij, axis=-1, keepdims=True) * X_ij
            + np.sum(X_i[:, None, :] * Y_ij, axis=-1, keepdims=True) * Y_ij
        )
        proj_X_i_XY_norm = proj_X_i_XY / (
            np.linalg.norm(proj_X_i_XY, axis=-1, keepdims=True) + eps
        )
        cos_opening = np.sum(
            proj_X_i_XY_norm * proj_X_i_XY_norm.swapaxes(0, 1),
            axis=-1,
        )
        if clamp_pairwise_params:
            cos_opening = np.clip(cos_opening, -1.0, 1.0)
        O_ij = np.arccos(cos_opening)

    # Buckle (B_ij)
    proj_Z_i_YZ = (
        np.sum(Z_i[:, None, :] * Y_ij, axis=-1, keepdims=True) * Y_ij
        + np.sum(Z_i[:, None, :] * Z_ij, axis=-1, keepdims=True) * Z_ij
    )
    proj_Z_i_YZ_norm = proj_Z_i_YZ / (
        np.linalg.norm(proj_Z_i_YZ, axis=-1, keepdims=True) + eps
    )
    cos_buckle = np.sum(
        proj_Z_i_YZ_norm * (-proj_Z_i_YZ_norm.swapaxes(0, 1)),
        axis=-1,
    )

    # Propeller (P_ij)
    proj_Z_i_ZX = (
        np.sum(Z_i[:, None, :] * Z_ij, axis=-1, keepdims=True) * Z_ij
        + np.sum(Z_i[:, None, :] * X_ij, axis=-1, keepdims=True) * X_ij
    )
    proj_Z_i_ZX_norm = proj_Z_i_ZX / (
        np.linalg.norm(proj_Z_i_ZX, axis=-1, keepdims=True) + eps
    )
    cos_propeller = np.sum(
        proj_Z_i_ZX_norm * (-proj_Z_i_ZX_norm.swapaxes(0, 1)),
        axis=-1,
    )

    if clamp_pairwise_params:
        cos_buckle = np.clip(cos_buckle, -1.0, 1.0)
        cos_propeller = np.clip(cos_propeller, -1.0, 1.0)

    B_ij = np.arccos(cos_buckle)
    P_ij = np.arccos(cos_propeller)

    pair_params: dict | None
    if return_basepairs_only:
        pair_params = None
    else:
        pair_params = {
            "H_ij": H_ij,
            "B_ij": B_ij,
            "P_ij": P_ij,
            "base_ori_ij": base_ori_ij,
        }

        if return_opening_angle and O_ij is not None:
            pair_params["O_ij"] = O_ij

        if return_pairwise_geometry:
            pair_params["X_ij"] = X_ij
            pair_params["Y_ij"] = Y_ij
            pair_params["Z_ij"] = Z_ij

    # --- CALC III: basepair filters / probabilities --------------
    hbond_summation = np.tensordot(
        hbond_count.astype(np.float32),
        np.asarray(mol_info.bp_summation_weights, dtype=np.float32),
        axes=([2], [0]),
    )  # [L, L]

    logits = mol_info.bp_hbond_coeff * (
        hbond_summation - (mol_info.min_hbonds_for_bp - 1)
    )
    bp_preds = (1.0 / (1.0 + np.exp(-logits))) + eps

    # Filter Height geometry
    H_ij_filter = (H_ij >= -mol_info.base_geometry_limits["H_ij"]) & (
        H_ij <= mol_info.base_geometry_limits["H_ij"]
    )
    # Filter Buckle geometry
    B_ij_filter = (B_ij <= mol_info.base_geometry_limits["B_ij"]) | (
        B_ij >= math.pi - mol_info.base_geometry_limits["B_ij"]
    )
    # Filter Propeller geometry
    P_ij_filter = (P_ij <= mol_info.base_geometry_limits["P_ij"]) | (
        P_ij >= math.pi - mol_info.base_geometry_limits["P_ij"]
    )
    
    bp_geom_filter = (H_ij_filter & B_ij_filter & P_ij_filter)

    if return_basepairs_only:
        # Avoid allocating basepairs_ij float matrix when only the boolean mask is needed.
        basepairs_bool_ij = (~seq_neighbors) & bp_geom_filter & (
            bp_preds >= float(mol_info.bp_val_cutoff)
        )
        basepairs_ij = None
    else:
        basepairs_ij = (~seq_neighbors).astype(np.float32) * (
            bp_geom_filter.astype(np.float32) * bp_preds.astype(np.float32)
        )
        basepairs_bool_ij = basepairs_ij >= mol_info.bp_val_cutoff

    if return_basepairs_only:
        # Cleanup intermediate tensors to free memory
        del frame_D_ij_vec, sc_D_ij_vec
        del hbond_summation, bp_preds
        del H_ij_filter, B_ij_filter, P_ij_filter, bp_geom_filter

        # Explicitly drop the largest pairwise arrays.
        del X_ij, Y_ij, Z_ij
        if O_ij is not None:
            del O_ij
        if local_base_params is not None:
            del local_base_params
        return basepairs_bool_ij

    assert pair_params is not None

    pair_params["basepairs_bool_ij"] = basepairs_bool_ij
    pair_params["hbond_summation"] = hbond_summation
    pair_params["basepairs_ij"] = basepairs_ij

    nucleic_ss_data = {"pair_params": pair_params}
    if return_local_params and local_base_params is not None:
        nucleic_ss_data["local_params"] = local_base_params

    # Cleanup intermediate tensors to free memory
    del frame_D_ij_vec, sc_D_ij_vec
    del hbond_summation, bp_preds
    del H_ij_filter, B_ij_filter, P_ij_filter, bp_geom_filter

    # If not returning, explicitly drop the largest pairwise arrays.
    if not return_pairwise_geometry:
        del X_ij, Y_ij, Z_ij
    if not return_opening_angle and O_ij is not None:
        del O_ij
    if not return_local_params and local_base_params is not None:
        del local_base_params

    return nucleic_ss_data

def annotate_na_ss(
    atom_array: AtomArray,
    *,
    NA_only: bool = False,
    planar_only: bool = True,
    p_canonical_bp_filter: float = 0.0,
    mol_info: Optional[NucMolInfo] = None,
    overwrite: bool = True,
    token_level_data: Optional[dict] = None,
) -> AtomArray:
    """Annotate base-pair partners directly onto the AtomArray.

    This computes nucleic-acid base pairing similarly to
    :func:`get_gt_nucleic_geom_feats` but instead of returning an integer
    secondary-structure matrix, it writes an AtomArray annotation
    ``bp_partners``.

    The annotation is stored on the *full* ``atom_array`` (length N atoms),
    but only nucleic-acid token-representative atoms (indices ``token_starts``
    from :func:`get_token_starts`) that are included in this call's
    ``annotation_mask`` get a list value.

    Semantics:
    - ``[]`` (empty list): explicitly unpaired nucleic-acid loop
    - ``[token_id, ...]``: paired nucleic-acid token(s)
    - ``None``: unannotated/masked (non-NA tokens, or tokens filtered out)

    Each list element is the partner token identifier (``token_id`` as int)
    for the paired residue. This is sufficient to recover the partner's
    token-representative atom via ``token_starts`` + token_id mapping.

        Notes
        -----
        - ``token_level_data`` may be metadata-only; this function will augment it
            with geometry as needed.
        - If ``p_canonical_bp_filter > 0``, then with that probability we discard
            any non-canonical NA basepairs (keeps only A-U, A-T, G-C).
    """

    if mol_info is None:
        mol_info = NucMolInfo()

    # Token representatives (0..L-1) and their corresponding atom indices (into atom_array)
    token_starts = get_token_starts(atom_array)
    token_level_array = atom_array[token_starts]
    # token_id is assigned token-wise and matches get_token_starts() segmentation.
    token_ids: list[int] = [int(t) for t in list(token_level_array.token_id)]
    token_res_names: list[str] = [str(rn) for rn in list(token_level_array.res_name)]

    # Compute basepairs on the token graph (respecting NA_only/planar_only filtering)
    if token_level_data is None:
        token_level_data = get_token_level_metadata(
            atom_array,
            mol_info,
            NA_only=NA_only,
            planar_only=planar_only,
        )
    token_level_data = add_token_level_geometry_data(
        atom_array,
        mol_info,
        token_level_data,
        NA_only=NA_only,
        planar_only=planar_only,
    )
    # Note: this mask gives positions that are *chemically valid* for forming 
    # base pairs, which is different from custom mask-generation for features
    mask_1d = np.asarray(token_level_data["filter_mask"], dtype=bool)
    
    subset_idxs = np.nonzero(mask_1d)[0]

    is_na_full = np.asarray(token_level_data["is_na"], dtype=bool)

    hbond_count = calculate_hb_counts(
        atom_array,
        token_level_data,
        mol_info,
        cutoff_HA_dist=mol_info.cutoff_HA_dist,
        cutoff_DA_dist=mol_info.cutoff_DA_dist,
    )
    bp_bool = np.asarray(
        _compute_nucleic_ss_impl(
            mol_info,
            token_level_data,
            hbond_count,
            clamp_pairwise_params=True,
            eps=1e-8,
            return_local_params=False,
            return_pairwise_geometry=False,
            return_opening_angle=False,
            return_basepairs_only=True,
        ),
        dtype=bool,
    )

    # Optional: filter to canonical Watson-Crick basepairs only.
    # Sampled probabilistically to allow mixed supervision during training.
    do_canonical_filter = bool(p_canonical_bp_filter and (np.random.rand() < float(p_canonical_bp_filter)))
    if do_canonical_filter:
        def _base_letter(res_name: str) -> str | None:
            rn = str(res_name).strip().upper()
            if rn in STANDARD_RNA:
                return rn
            if rn in STANDARD_DNA:
                return rn[1]  # DA/DC/DG/DT -> A/C/G/T
            return None

        allowed_pairs = {
            ("A", "U"), ("U", "A"),
            ("A", "T"), ("T", "A"),
            ("G", "C"), ("C", "G"),
        }
        base_letters_full: list[str | None] = [_base_letter(rn) for rn in token_res_names]

        bp_bool = np.asarray(bp_bool, dtype=bool)
        bp_rows_tmp, bp_cols_tmp = np.nonzero(bp_bool)
        for r, c in zip(bp_rows_tmp.tolist(), bp_cols_tmp.tolist()):
            full_i = int(subset_idxs[int(r)])
            full_j = int(subset_idxs[int(c)])
            bi = base_letters_full[full_i]
            bj = base_letters_full[full_j]
            if bi is None or bj is None or (bi, bj) not in allowed_pairs:
                bp_bool[int(r), int(c)] = False
                bp_bool[int(c), int(r)] = False

    bp_bool = np.asarray(bp_bool, dtype=bool)
    bp_rows, bp_cols = np.nonzero(bp_bool)

    # Prepare/overwrite annotation array
    if (not overwrite) and ("bp_partners" in atom_array.get_annotation_categories()):
        bp_partners_ann = atom_array.bp_partners
        if len(bp_partners_ann) != len(atom_array):
            raise ValueError(
                "Existing bp_partners annotation has wrong length"
            )
    else:
        bp_partners_ann = np.empty(len(atom_array), dtype=object)
        bp_partners_ann[:] = None

    # Explicit-loop semantics:
    # - Only nucleic-acid token-start atoms *within subset_idxs* get a list container.
    # - [] means explicitly unpaired loop.
    # - None means unannotated/masked.
    for full_i in subset_idxs.tolist():
        if not bool(is_na_full[int(full_i)]):
            continue
        atom_i = int(token_starts[int(full_i)])
        if bp_partners_ann[atom_i] is None:
            bp_partners_ann[atom_i] = []

    # Populate partners using token_id ints
    # We only process each unordered pair once to avoid duplicates.
    for r, c in zip(bp_rows.tolist(), bp_cols.tolist()):
        if r == c:
            continue

        full_i = int(subset_idxs[int(r)])
        full_j = int(subset_idxs[int(c)])
        if full_i == full_j:
            continue

        # Only annotate NA-NA basepairs as nucleic secondary structure.
        if (not bool(is_na_full[int(full_i)])) or (not bool(is_na_full[int(full_j)])):
            continue

        # Enforce uniqueness: only handle (i,j) where i < j
        if full_j < full_i:
            continue

        atom_i = int(token_starts[full_i])
        atom_j = int(token_starts[full_j])
        partner_i = int(token_ids[full_j])
        partner_j = int(token_ids[full_i])

        if bp_partners_ann[atom_i] is None:
            bp_partners_ann[atom_i] = []
        if bp_partners_ann[atom_j] is None:
            bp_partners_ann[atom_j] = []

        # Add if not present
        if partner_i not in bp_partners_ann[atom_i]:
            bp_partners_ann[atom_i].append(partner_i)
        if partner_j not in bp_partners_ann[atom_j]:
            bp_partners_ann[atom_j].append(partner_j)

    atom_array.set_annotation("bp_partners", bp_partners_ann)
    return atom_array


def bp_partner_to_ss_matrix(
    atom_array: AtomArray,
    *,
    feature_info: Optional[dict] = None,
    mol_info: Optional[NucMolInfo] = None,
    include_loops: bool = True,
    token_level_data: Optional[dict] = None,
) -> np.ndarray:
    """Reconstruct an integer NA secondary-structure matrix from annotations.

    Requires that ``atom_array`` has a ``bp_partners`` annotation created by
    :func:`annotate_na_ss`.

    Returns
    -------
    ss_matrix : np.ndarray
        Shape (L, L) with values from ``feature_info``.

        Loop semantics:
        - Only nucleic-acid tokens can be loops.
        - Only tokens with an explicit empty list ``bp_partners == []`` are loops.
            Unannotated tokens (``bp_partners is None``) remain masked.
    """

    if mol_info is None:
        mol_info = NucMolInfo()

    if feature_info is None:
        feature_info = DEFAULT_NA_SS_FEATURE_INFO

    if "bp_partners" not in atom_array.get_annotation_categories():
        raise ValueError(
            "atom_array is missing bp_partners annotation; run annotate_na_ss() first"
        )

    token_starts = get_token_starts(atom_array)
    token_level_array = atom_array[token_starts]
    token_ids_int: list[int] = [int(t) for t in list(token_level_array.token_id)]
    token_id_to_index_int = {int(tid): i for i, tid in enumerate(token_ids_int)}
    L = len(token_starts)

    ss_matrix = feature_info["NA_SS_MASK"] * np.ones((L, L), dtype=np.int64)

    if token_level_data is None:
        token_level_data = get_token_level_metadata(
            atom_array,
            mol_info,
            # NA_only=NA_only,
            # planar_only=planar_only,
        )

    mask_1d = np.asarray(token_level_data["filter_mask"], dtype=bool)
    subset_idxs = np.nonzero(mask_1d)[0]
    subset_set = set(int(x) for x in subset_idxs.tolist())
    is_na = np.asarray(token_level_data["is_na"], dtype=bool)
    subset_na_idxs = subset_idxs[np.asarray(is_na[subset_idxs], dtype=bool)]
    subset_na_set = set(int(x) for x in subset_na_idxs.tolist())

    # Fill base-pair edges (only within subset, and only NA-NA)
    bp_partners_ann = atom_array.bp_partners
    for i in subset_idxs.tolist():
        if not bool(is_na[int(i)]):
            continue
        atom_i = int(token_starts[int(i)])
        partners = bp_partners_ann[atom_i]
        if partners is None:
            continue
        if not isinstance(partners, (list, tuple, np.ndarray)):
            continue
        for partner_token_id in partners:
            # Support int, numpy scalar, and legacy stringified token_id.
            try:
                partner_tid_int = int(partner_token_id)
            except Exception:
                partner_tid_int = None
            j = token_id_to_index_int.get(partner_tid_int) if partner_tid_int is not None else None
            if j is None or j == i:
                continue
            if int(j) not in subset_set:
                continue
            if not bool(is_na[int(j)]):
                continue
            ss_matrix[i, j] = feature_info["NA_SS_PAIR"]
            ss_matrix[j, i] = feature_info["NA_SS_PAIR"]

    if not include_loops:
        return ss_matrix

    # Loop labeling is explicit and NA-only:
    # - only nucleic tokens can be loops
    # - only tokens with an explicit empty list annotation are loops
    loop_idxs_list: list[int] = []
    for i in subset_idxs.tolist():
        if not bool(is_na[int(i)]):
            continue
        atom_i = int(token_starts[int(i)])
        partners = bp_partners_ann[atom_i]
        if not isinstance(partners, (list, tuple, np.ndarray)):
            continue
        if len(partners) == 0:
            loop_idxs_list.append(int(i))

    loop_idxs = np.asarray(loop_idxs_list, dtype=np.int64)
    if loop_idxs.size > 0:
        ss_matrix[loop_idxs[:, None], subset_na_idxs[None, :]] = feature_info["NA_SS_LOOP"]
        ss_matrix[subset_na_idxs[:, None], loop_idxs[None, :]] = feature_info["NA_SS_LOOP"]

    return ss_matrix


def parse_dot_bracket(dot_bracket: str) -> tuple[list[tuple[int, int]], list[int]]:
    """Parse a dot-bracket string into base pairs and unpaired positions.

    Supports (), [], {}, <>, and A..E / a..e bracket pairs.

    Returns
    -------
    pairs : list of (i, j)
        0-based indices in the string for paired positions.
    unpaired : list of int
        0-based indices that are '.' (unpaired).
    """

    stack: dict[str, list[int]] = {}
    pairs: list[tuple[int, int]] = []
    unpaired: list[int] = []

    opener_for = {
        ")": "(",
        "]": "[",
        "}": "{",
        ">": "<",
        "a": "A",
        "b": "B",
        "c": "C",
        "d": "D",
        "e": "E",
    }

    for i, ch in enumerate(str(dot_bracket)):
        if ch == ".":
            unpaired.append(i)
        elif ch in "([{<ABCDE":
            stack.setdefault(ch, []).append(i)
        elif ch in ")]}>abcde":
            o = opener_for.get(ch)
            if o is None or o not in stack or not stack[o]:
                continue
            j = stack[o].pop()
            pairs.append((j, i))
        else:
            continue

    return pairs, unpaired


def annotate_na_ss_from_specification(
    atom_array: AtomArray,
    specification: dict,
    *,
    overwrite: bool = True,
) -> AtomArray:
    """Annotate ``bp_partners`` from an inference-time specification.

    This is the inference analogue of :func:`annotate_na_ss`, except instead
    of computing base pairs from geometry/H-bonds, it interprets a user-provided
    specification (dot-bracket strings and/or residue ranges/positions) and
    writes the same ``bp_partners`` annotation on token-representative atoms.

    Supported spec keys (all optional):
      - ``ss_dbn``: global dot-bracket string applied to the first L tokens.
      - ``ss_dbn_dict``: mapping like {"A5-15": dbn_str, ...}.
      - ``paired_region_list``: list of "A5-15,B1-11" entries.
      - ``paired_position_list``: list of "A19,A61,A20" groups.
      - ``loop_region_list``: list of "A5-10" regions forced unpaired.
    """

    spec = specification or {}
    token_starts = get_token_starts(atom_array)
    token_level_array = atom_array[token_starts]
    token_ids: list[int] = [int(t) for t in list(token_level_array.token_id)]
    n_tokens = len(token_starts)

    # Explicit loops are only meaningful for nucleic-acid tokens.
    seq_data = _get_sequence_encoding_data()
    is_rna_like = np.isin(token_level_array.res_name, seq_data["rna_like_res_names"])
    is_dna_like = np.isin(token_level_array.res_name, seq_data["dna_like_res_names"])
    is_na_token = np.asarray(is_rna_like | is_dna_like, dtype=bool)
    del seq_data

    # Prepare/overwrite annotation array
    if (not overwrite) and ("bp_partners" in atom_array.get_annotation_categories()):
        bp_partners_ann = atom_array.bp_partners
        if len(bp_partners_ann) != len(atom_array):
            raise ValueError("Existing bp_partners annotation has wrong length")
    else:
        bp_partners_ann = np.empty(len(atom_array), dtype=object)
        bp_partners_ann[:] = None

    # Build chain/res -> token index map for region/position specs
    chain_iid_list: list[str] = [str(atm.chain_iid) for atm in token_level_array]
    resi_list: list[int] = [int(atm.res_id) for atm in token_level_array]
    chain_res_to_tok: dict[tuple[str, int], int] = {
        (c, r): i for i, (c, r) in enumerate(zip(chain_iid_list, resi_list))
    }

    def _parse_region(region_str: str) -> tuple[str, int, int] | None:
        region_str = str(region_str).strip()
        if not region_str:
            return None
        chain_id = region_str[0]
        rest = region_str[1:]
        if "-" not in rest:
            return None
        start_s, end_s = rest.split("-", 1)
        try:
            start_res = int(start_s)
            end_res = int(end_s)
        except ValueError:
            return None
        if start_res > end_res:
            start_res, end_res = end_res, start_res
        return chain_id, start_res, end_res

    def _parse_single_pos(pos_str: str) -> tuple[str, int] | None:
        pos_str = str(pos_str).strip()
        if not pos_str:
            return None
        chain_id = pos_str[0]
        rest = pos_str[1:]
        try:
            res_id = int(rest)
        except ValueError:
            return None
        return chain_id, res_id

    def _region_to_token_indices(region_str: str) -> list[int]:
        parsed = _parse_region(region_str)
        if parsed is None:
            return []
        chain_id, start_res, end_res = parsed
        token_indices: list[int] = []
        for res_id in range(start_res, end_res + 1):
            idx = chain_res_to_tok.get((chain_id, int(res_id)))
            if idx is not None:
                token_indices.append(int(idx))
        return token_indices

    def _pos_to_token_index(pos_str: str) -> int | None:
        parsed = _parse_single_pos(pos_str)
        if parsed is None:
            return None
        chain_id, res_id = parsed
        return chain_res_to_tok.get((chain_id, int(res_id)))

    # Accumulate partners as token-index sets
    partners: list[set[int]] = [set() for _ in range(n_tokens)]
    loop_token_idxs: set[int] = set()

    def _add_pair(i: int, j: int) -> None:
        if not (0 <= i < n_tokens and 0 <= j < n_tokens):
            return
        if i == j:
            return
        if (not bool(is_na_token[int(i)])) or (not bool(is_na_token[int(j)])):
            return
        if i in loop_token_idxs or j in loop_token_idxs:
            return
        partners[i].add(j)
        partners[j].add(i)

    # Case 1: global ss_dbn
    ss_dbn = spec.get("ss_dbn")
    if isinstance(ss_dbn, str) and ss_dbn.strip():
        pairs, unpaired = parse_dot_bracket(ss_dbn.strip())
        L = min(len(ss_dbn), n_tokens)
        for i_local, j_local in pairs:
            if 0 <= i_local < L and 0 <= j_local < L:
                _add_pair(int(i_local), int(j_local))
        for i_local in unpaired:
            if 0 <= int(i_local) < L:
                loop_token_idxs.add(int(i_local))

    # Case 1b: ss_dbn_dict
    ss_dbn_dict = spec.get("ss_dbn_dict", {}) or {}
    if isinstance(ss_dbn_dict, dict):
        for region_str, dbn_str in ss_dbn_dict.items():
            if not isinstance(region_str, str) or not isinstance(dbn_str, str):
                continue
            dbn_str = dbn_str.strip()
            if not dbn_str:
                continue
            toks = _region_to_token_indices(region_str)
            if not toks or len(toks) != len(dbn_str):
                continue
            pairs, unpaired = parse_dot_bracket(dbn_str)
            for i_local, j_local in pairs:
                if 0 <= i_local < len(toks) and 0 <= j_local < len(toks):
                    _add_pair(int(toks[int(i_local)]), int(toks[int(j_local)]))
            for i_local in unpaired:
                if 0 <= i_local < len(toks):
                    loop_token_idxs.add(int(toks[int(i_local)]))

    # Case 2: paired_region_list
    paired_region_list = spec.get("paired_region_list", [])
    if isinstance(paired_region_list, str):
        paired_region_list = [paired_region_list]
    if isinstance(paired_region_list, list):
        for region_entry in paired_region_list:
            if not isinstance(region_entry, str) or not region_entry.strip():
                continue
            region_parts = [p.strip() for p in region_entry.split(",") if p.strip()]
            if len(region_parts) != 2:
                continue
            toks1 = _region_to_token_indices(region_parts[0])
            toks2 = _region_to_token_indices(region_parts[1])
            if not toks1 or not toks2:
                continue
            for ti in toks1:
                for tj in toks2:
                    _add_pair(int(ti), int(tj))

    # Case 3: paired_position_list
    paired_position_list = spec.get("paired_position_list", [])
    if isinstance(paired_position_list, str):
        paired_position_list = [paired_position_list]
    if isinstance(paired_position_list, list):
        for group_str in paired_position_list:
            if not isinstance(group_str, str) or not group_str.strip():
                continue
            pos_parts = [p.strip() for p in group_str.split(",") if p.strip()]
            tok_indices: list[int] = []
            for pos_str in pos_parts:
                tok = _pos_to_token_index(pos_str)
                if tok is not None:
                    tok_indices.append(int(tok))
            for i in range(len(tok_indices)):
                for j in range(i + 1, len(tok_indices)):
                    _add_pair(tok_indices[i], tok_indices[j])

    # Case 4: loop_region_list
    loop_region_list = spec.get("loop_region_list", [])
    if isinstance(loop_region_list, str):
        loop_region_list = [loop_region_list]
    if isinstance(loop_region_list, list):
        for region_str in loop_region_list:
            if not isinstance(region_str, str) or not region_str.strip():
                continue
            for tok in _region_to_token_indices(region_str):
                loop_token_idxs.add(int(tok))

    # Enforce loop tokens as unpaired: remove any pairs involving them
    for i in list(loop_token_idxs):
        if not (0 <= i < n_tokens):
            continue
        if not bool(is_na_token[int(i)]):
            loop_token_idxs.discard(int(i))
            continue
        for j in list(partners[i]):
            partners[j].discard(i)
        partners[i].clear()

    # Write lists of partner token_ids onto token-start atoms.
    # Unspecified tokens remain unannotated (None) -> NA_SS_MASK.
    for i in range(n_tokens):
        atom_i = int(token_starts[i])
        if not bool(is_na_token[int(i)]):
            continue
        if len(partners[i]) > 0:
            bp_partners_ann[atom_i] = []
            for j in sorted(partners[i]):
                partner_token_id = int(token_ids[int(j)])
                bp_partners_ann[atom_i].append(partner_token_id)
        elif int(i) in loop_token_idxs:
            bp_partners_ann[atom_i] = []

    atom_array.set_annotation("bp_partners", bp_partners_ann)
    return atom_array


def annotate_na_ss_from_data_specification(
    data: dict,
    *,
    overwrite: bool = True,
) -> AtomArray:
    """Convenience wrapper: annotate bp partners from ``data['specification']``."""
    atom_array = data["atom_array"]
    spec = data.get("specification", {}) or {}
    return annotate_na_ss_from_specification(atom_array, spec, overwrite=overwrite)

