import ast
import os
import random
import re

import numpy as np
from biotite.structure import Atom, array
from cifutils.parser import parse
from cifutils.utils.io_utils import to_cif_file
from datahub.utils.token import get_token_starts, spread_token_wise


def split_contig(x):
    try:
        chain = str(x[0])
        idx = x[1:]
        idx = int(idx)
        assert idx > 0, "Residue index must be a positive integer."
    except Exception as e:
        print(f"Invalid contig format: '{x}'. Expected format is 'ChainIDResID' (e.g. 'A20').")
        raise e
    return [chain, idx]
    
def extract_pn_unit_info(contig):
    """
    Convert substring like A20-21 to separate terms: A, 20, 21.
    """
    pattern = r'([A-Za-z])(\d+)-(\d+)'

    # Extract components
    match = re.match(pattern, contig)
    if match:
        pn_unit_id = match.group(1)  # The letter, e.g., 'A'
        start = int(match.group(2))  # The starting number, e.g., 194
        end = int(match.group(3))  # The ending number, e.g., 195
        
    return pn_unit_id, start, end


def get_design_pattern_with_constraints(contig, length=None):
    """
    Convert the contig string to separate modules.
    e.g. '1-5,A20-21,1-5,A25-25,1-5,A30-30,/0,1-5' with length = 10-10 may be converted to [2, A20, A21, 2, A25, 3, A30, /0, 3]
    Integers represent number of free residues to put there.
    """
    contig_parts = contig.split(',')

    # Separate fixed segments (e.g., "A1051-1051") and variable ranges (e.g., "0-40")
    variable_ranges = []
    fixed_parts = []
    pos_to_put_motif = []
    
    for part in contig_parts:
        if any(c.isalpha() for c in part):  # Detect parts containing letters as fixed
            pn_unit_id, pn_unit_start, pn_unit_end = extract_pn_unit_info(part)
            fixed_parts.append([pn_unit_id, pn_unit_start, pn_unit_end])
            pos_to_put_motif.append(1)
        elif part == "/0":
            pos_to_put_motif.append(2)
        else:
            variable_ranges.append(list(map(int, part.split('-'))))
            pos_to_put_motif.append(0)
            
    # adjust the total length to solely for free residues
    num_motif_residues = sum([i[2] - i[1] + 1 for i in fixed_parts])

    if length is None:
        length_min, length_max = 0, 9999
    else:
        length_min, length_max = map(int, length.split('-'))
    length_min -= num_motif_residues
    length_max -= num_motif_residues
    
    remaining_length_min = length_min
    remaining_length_max = length_max

    num_free_atoms = []
    for range_limits in variable_ranges:
        min_value = range_limits[0]
        max_value = range_limits[1]

        # Calculate the valid range for the current segment
        valid_min = max(min_value, remaining_length_min - sum(r[1] for r in variable_ranges[len(num_free_atoms) + 1:]))
        valid_max = min(max_value, remaining_length_max - sum(r[0] for r in variable_ranges[len(num_free_atoms) + 1:]))

        if valid_min > valid_max and length is not None:
            raise ValueError("No valid selections possible with the given constraints.")

        # Randomly select a value for the current segment
        selected_value = random.randint(valid_min, valid_max)
        num_free_atoms.append(selected_value)

        # Update remaining lengths
        remaining_length_min -= selected_value
        remaining_length_max -= selected_value

    atoms_with_motif = []
    for idx in range(len(pos_to_put_motif)):
        if pos_to_put_motif[idx] == 1:
            motif = fixed_parts.pop(0)
            pn_unit_id, pn_unit_start, pn_unit_end = motif[0], motif[1], motif[2]
            for index in range(pn_unit_start, pn_unit_end+1):
                atoms_with_motif.append(f"{pn_unit_id}{index}")
        elif pos_to_put_motif[idx] == 0:
            free_atom = num_free_atoms.pop(0)
            atoms_with_motif.append(free_atom)
        elif pos_to_put_motif[idx] == 2:
            atoms_with_motif.append("/0")

    return atoms_with_motif

def get_atom_array_from_contig_map(contig, contig_atom, length, pdb, cache_dir='/home/bqiang/contig_cache/'):
    """
    Generate atom array from input contig map.
    The format of provided contig information is 
    e.g. 
    contigs = '1-5,A20-21,1-5,A25-25,1-5,A30-30,/0,1-5', contig_atoms = {'A20':'CB,CG', 'A25':'OG1,CG2','A30':'CG,CD'}, length = 10-10
    means we want residue 20, 21, 25, 30 on chain A as motif, the specific motif atoms are CB, CG on A20, OG1, CG2 on A25, CG, CD on A30. 
    Also, in chain A, put 1-5 free residues before A20, 1-5 free residues between A21 and A25, 1-5 free residues between A25 and A30,
    In chain B, we want 1-5 free residues. There should be in total 10 atoms.

    Args:
      - contig: contig strings.
      - contig_atom: atoms chosen as guidepost in the motif residue in the config.
      - length: total number of free atoms.
      - pdb: path of pdb file to chosen motif from.
    """
    # convert str to dict
    if isinstance(contig_atom, str):
        contig_atom = ast.literal_eval(contig_atom)
    
    # parse the pdb into the original atom array
    data = parse(pdb, add_missing_atoms=False)
    atom_array = data["asym_unit"][0]
    
    # parse the contig map
    atoms_with_motif = get_design_pattern_with_constraints(contig, length)
    
    pn_unit_id = "A" # same as chain_id, chain_entity, pn_unit_entity here
    molecule_id = 0
    res_id = 1
    
    all_atoms = []
    all_pn_unit_id = []
    all_pn_unit_entity = []
    all_molecule_id = []
    all_res_id = []
    all_is_motif = []
    common_args = {
                    "chain_id": "A",
                    "res_id": 0,
                    "ins_code": "",
                    "res_name": "ALA",
                    "hetero": False,
                    "is_backbone_atom": True,
                    "is_aromatic": False,
                    "occupancy": 1.0,
                    "uses_alt_atom_id": False,
                    "b_factor": 55.0,
                    "charge": 0,
                    "chain_type": 6,
                    "is_polymer": True,
                    "pn_unit_id": "A",
                    "molecule_id": 1,
                    "chain_entity": 0,
                    "pn_unit_entity": 0,
                    "molecule_entity": 1,
                    "atom_id": 0,
                    "label_entity_id": '?',
                    "auth_seq_id": 0,
                }
    specfic_args = {
        'N': {'atom_name': 'N', 'element': 'N', 'alt_atom_id': 'N', 'stereo': 'N', 'atomic_number': 7},
        'CA': {'atom_name': 'CA', 'element': 'C', 'alt_atom_id': 'CA', 'stereo': 'S', 'atomic_number': 6},
        'C': {'atom_name': 'C', 'element': 'C', 'alt_atom_id': 'C', 'stereo': 'N', 'atomic_number': 6},
        'O': {'atom_name': 'O', 'element': 'O', 'alt_atom_id': 'O', 'stereo': 'N', 'atomic_number': 8},
        'CB': {'atom_name': 'CB', 'element': 'C', 'alt_atom_id': 'CB', 'stereo': 'N', 'atomic_number': 6},
        'H': {'atom_name': 'H', 'element': 'H', 'alt_atom_id': 'H', 'stereo': 'N', 'atomic_number': 1},
    }
    common_args = {key: value for key, value in common_args.items() if key in atom_array[0]._annot.keys()}
    for key, value in specfic_args.items():
        specfic_args[key] = {key: value for key, value in specfic_args[key].items() if key in atom_array[0]._annot.keys()}
    
    
    for module in atoms_with_motif:
        if module == "/0":
            pn_unit_id = chr(ord(pn_unit_id) + 1)
            molecule_id += 1
            res_id = 1
            continue
        
        if str(module)[0].isalpha(): # if is user-selected motif
            # if all atoms are chosen
            selected_chain_id, selected_res_id = module.split("-")
            if contig_atom is None or f"{selected_chain_id}{selected_res_id}" not in contig_atom.keys():
                atoms_cur_module = atom_array[np.array(atom_array.pn_unit_id == selected_chain_id) & np.array(atom_array.res_id == int(selected_res_id))]
                all_atoms.extend([atom for atom in atoms_cur_module])
                all_pn_unit_id.extend([pn_unit_id] * len(atoms_cur_module))
                all_pn_unit_entity.extend([ord(pn_unit_id) - ord('A')] * len(atoms_cur_module))
                all_molecule_id.extend([molecule_id] * len(atoms_cur_module))
                all_res_id.extend([res_id] * len(atoms_cur_module))
                all_is_motif.extend([True] * len(atoms_cur_module))
                res_id += 1
            else:
                # choose corresponding atoms from that token
                atoms_cur_module = atom_array[np.array(atom_array.pn_unit_id == selected_chain_id) & np.array(atom_array.res_id == int(selected_res_id))]
                atoms_name_cur_module = atoms_cur_module.atom_name
                chosen_as_motif = np.array(contig_atom[f"{selected_chain_id}{selected_res_id}"].split(","))
                chosen_atom_idx_in_residue = np.isin(atoms_name_cur_module, chosen_as_motif)
                
                assert np.sum(chosen_atom_idx_in_residue) == len(chosen_as_motif), "Some given motif atoms are not in the pdb."
                
                all_atoms.extend([atom for atom in atoms_cur_module])
                all_pn_unit_id.extend([pn_unit_id] * len(atoms_cur_module))
                all_pn_unit_entity.extend([ord(pn_unit_id) - ord('A')] * len(atoms_cur_module))
                all_molecule_id.extend([molecule_id] * len(atoms_cur_module))
                all_res_id.extend([res_id] * len(atoms_cur_module))
                all_is_motif.extend(list(chosen_atom_idx_in_residue))
                res_id += 1
        else: # add free atoms
            # Generate atoms for ALA's 5 heavy atoms
            pseudo_backbone_atom_array_single_residue = [
                Atom(np.array([0.0, 0.0, 0.0], dtype=np.float32), **common_args,
                    **specfic_args['N']),
                Atom(np.array([1.5, 0.0, 0.0], dtype=np.float32), **common_args,
                    **specfic_args['CA']),
                Atom(np.array([2.5, 1.0, 0.0], dtype=np.float32), **common_args,
                    **specfic_args['C']),
                Atom(np.array([3.5, 1.0, 1.0], dtype=np.float32), **common_args,
                    **specfic_args['O']),
                Atom(np.array([0.5, 1.5, -1.0], dtype=np.float32), **common_args,
                    **specfic_args['CB'])
            ]
        
            all_atoms.extend(pseudo_backbone_atom_array_single_residue * int(module))
            all_pn_unit_id.extend([pn_unit_id] * 5 * int(module))
            all_pn_unit_entity.extend([ord(pn_unit_id) - ord('A')] * 5 * int(module))
            all_molecule_id.extend([molecule_id] * 5 * int(module))
            all_res_id.extend([i for i in range(res_id, int(module) + res_id) for _ in range(5)])
            all_is_motif.extend([False] * 5 * int(module))
            res_id += int(module)
    
    # adjust positional id
    combined_atom_array = array(all_atoms)
    
    # re-assign the res_id in each chain
    combined_atom_array.chain_id = np.array(all_pn_unit_id)
    combined_atom_array.pn_unit_id = np.array(all_pn_unit_id)
    combined_atom_array.pn_unit_entity = np.array(all_pn_unit_entity)
    combined_atom_array.molecule_id = np.array(all_molecule_id)
    combined_atom_array.set_annotation('molecule_iid', np.array(all_molecule_id))
    combined_atom_array.res_id = np.array(all_res_id)

    # generate atom array with annotation is_motif and can_be_gp
    if "is_motif_atom" not in atom_array.get_annotation_categories():
        combined_atom_array.set_annotation("is_motif_atom", all_is_motif)
    if "can_be_gp" not in atom_array.get_annotation_categories():
        combined_atom_array.set_annotation("can_be_gp", np.array([True] * len(combined_atom_array)))
    
    # record the chain_id, res_id and atom_id of is_motif atoms and can_be_gp atoms
    is_motif_dict, can_be_gp_dict = {}, {}
    for idx, atom in enumerate(combined_atom_array[combined_atom_array.is_motif_atom]):
            if atom.chain_id not in is_motif_dict:
                is_motif_dict[atom.chain_id] = {}
            if atom.res_id not in is_motif_dict[atom.chain_id]:
                is_motif_dict[atom.chain_id][atom.res_id] = []
            is_motif_dict[atom.chain_id][atom.res_id].append(atom.atom_name)
    for idx, atom in enumerate(combined_atom_array[combined_atom_array.can_be_gp]):
        if atom.chain_id not in can_be_gp_dict:
            can_be_gp_dict[atom.chain_id] = {}
        if atom.res_id not in can_be_gp_dict[atom.chain_id]:
            can_be_gp_dict[atom.chain_id][atom.res_id] = []
        can_be_gp_dict[atom.chain_id][atom.res_id].append(atom.atom_name)

    chain_iid = []
    seen_chain = {}
    last_chain = None
    for chain_id in combined_atom_array.chain_id:
        if chain_id != last_chain:
            if chain_id in seen_chain:
                seen_chain[chain_id] += 1
            else:
                seen_chain[chain_id] = 1
            last_chain = chain_id
        chain_iid.append(chain_id + '_' + str(seen_chain[chain_id]))
    
    combined_atom_array.set_annotation("pn_unit_iid", chain_iid)
    combined_atom_array.set_annotation("chain_iid", chain_iid)
    
    # get the token wise annotations 
    chain_iid_token = combined_atom_array[get_token_starts(combined_atom_array)].chain_iid
        #convert to cif format and reparse
    cache_id = "-".join([contig, length, pdb.split("/")[-1].split(".")[0]])
    tmpfile=os.path.join( cache_dir, cache_id + ".cif")
    
    # Write and load the atomarray i guess?
    to_cif_file(combined_atom_array, tmpfile)
    parse_out = parse(tmpfile, remove_hydrogens=True)
    combined_atom_array = parse_out["asym_unit"][0] if "assemblies" in parse_out else parse_out["asym_unit"][0]
    
    chain_iid = spread_token_wise(combined_atom_array, chain_iid_token)
    combined_atom_array.set_annotation("pn_unit_iid", chain_iid)
    combined_atom_array.set_annotation("chain_iid", chain_iid)
    
    # add the is_motif and can_be_gp annotations
    is_motif = np.zeros(len(combined_atom_array), dtype=bool)
    can_be_gp = np.zeros(len(combined_atom_array), dtype=bool)
    for idx, atom in enumerate(combined_atom_array):
        if atom.chain_id in is_motif_dict and atom.res_id in is_motif_dict[atom.chain_id]:
            if atom.atom_name in is_motif_dict[atom.chain_id][atom.res_id]:
                is_motif[idx] = True
        if atom.chain_id in can_be_gp_dict and atom.res_id in can_be_gp_dict[atom.chain_id]:
            if atom.atom_name in can_be_gp_dict[atom.chain_id][atom.res_id]:
                can_be_gp[idx] = True

    combined_atom_array.set_annotation("is_motif_atom", is_motif)
    combined_atom_array.set_annotation("can_be_gp", can_be_gp)
    
    return combined_atom_array
        

# for test
# contigs = '5-5,A20-21,1-5,A25-25,1-5,A30-30,/0,1-5'
# contig_atoms = "{'A20':'CB,CG', 'A25':'OG1,CG2','A30':'CG,CD'}"
# length = "10-10"

# design_pdb = '/home/odinz/rf3_design/rf_diffusion_repo/rf_diffusion/benchmark/odin_test/mini_case/run_M0024_1nzy_cond0_0-atomized-bb-False.pdb'
'''
input_pdb = '/home/yanjing/beta_barrels/temp_file/design_run_M0024_1nzy_cond0_0-atomized-bb-False.pdb'
contig='49-49,A64-64,21-21,A86-86,3-3,A90-90,23-23,A114-114,22-22,A137-137,7-7,A145-145,49-49'
contig_atoms="{'A64':'O,C','A86':'CB,CA,N,C','A90':'CE1,ND1,NE2,CG,CD2','A114':'N,CA','A137':'NE1,CD1,CE2,CG,CD2,CZ2','A145':'OD2,CG,CB,OD1'}"
length = "180-180"

data = get_atom_array_from_contig_map(contig, contig_atoms, length, pdb=input_pdb)
print(data)
'''