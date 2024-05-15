import torch
import requests
import numpy as np
import networkx as nx
from typing import Any, Dict, List, Tuple, Set
from openbabel.openbabel import OBMol
from scipy.spatial import Delaunay
from data_constants import AMINO_ACID_POLAR_TIP_ATOMS

import rf2aa.cifutils as cifutils
from rf2aa.chemical import ChemicalData
from rf2aa.util import (
    cif_ligand_to_obmol,
    cif_ligand_to_xyz,
    get_ligand_atoms_bonds,
    cif_poly_to_xyz,
)


def get_chain_name(chain: cifutils.Chain) -> str:
    return next(iter(chain.atoms.keys()))[-2]


def get_ligand_keys(chains: Dict[str, cifutils.Chain]) -> List[str]:
    nonpoly_chain_keys = [
        c
        for c in chains.keys()
        if chains[c].type == "nonpoly" and get_chain_name(chains[c]) != "HOH"
    ]
    return nonpoly_chain_keys


def merge_ligand_keys(
    covale: List[cifutils.Bond], ligand_keys: List[str]
) -> List[Set[str]]:
    ligand_key_sets = [set([ligand_key]) for ligand_key in ligand_keys]
    for covalent_bond in covale:
        key_a = covalent_bond.a[0]
        key_b = covalent_bond.b[0]

        key_set_index_a = None
        key_set_index_b = None
        for key_set_index, key_set in enumerate(ligand_key_sets):
            if key_a in key_set:
                key_set_index_a = key_set_index
            if key_b in key_set:
                key_set_index_b = key_set_index
        if (
            key_set_index_a is None
            or key_set_index_b is None
            or key_set_index_a == key_set_index_b
        ):
            continue

        # Note: we need to pop the larger index before the smaller index,
        # because the larger index would change if we popped the smaller index first.
        key_set_index_larger, key_set_index_smaller = max(
            key_set_index_a, key_set_index_b
        ), min(key_set_index_a, key_set_index_b)

        key_set_larger = ligand_key_sets.pop(key_set_index_larger)
        key_set_smaller = ligand_key_sets.pop(key_set_index_smaller)
        merged_key_set = key_set_larger.union(key_set_smaller)
        ligand_key_sets.append(merged_key_set)
    return ligand_key_sets


def get_ligand_name_from_query_ligand(
    query_ligand: List[Tuple],
) -> str:
    ligand_names = [lig[2] for lig in query_ligand]
    ligand_names = sorted(ligand_names)
    ligand_name = "".join(ligand_names)
    return ligand_name


def get_ligand_name_counts(
    chains: Dict[str, cifutils.Chain], ligand_key_sets: List[Set[str]]
) -> Dict[str, int]:
    ligand_name_counts = {}
    for ligand_key_set in ligand_key_sets:
        # For multi-chain ligands, the name is just the sorted list of names,
        # concatenated.
        ligand_names = []
        for ligand_key in ligand_key_set:
            ligand_names_per_key = [
                atom_key[2] for atom_key in chains[ligand_key].atoms.keys()
            ]
            ligand_names.extend(ligand_names_per_key)
        ligand_names = sorted(ligand_names)
        ligand_name = "".join(ligand_names)

        if ligand_name not in ligand_name_counts:
            ligand_name_counts[ligand_name] = 0
        ligand_name_counts[ligand_name] += 1
    return ligand_name_counts


def get_ligand_name_count_dictionary(
    assembly_chains: Dict[str, cifutils.Chain], covale: List[cifutils.Bond]
) -> Dict[str, int]:
    nonpoly_chain_keys = get_ligand_keys(assembly_chains)
    ligand_key_sets = merge_ligand_keys(covale, nonpoly_chain_keys)
    ligand_name_counts = get_ligand_name_counts(assembly_chains, ligand_key_sets)
    return ligand_name_counts


def create_amino_acid_atom_dictionary() -> List[Dict[str, int]]:
    amino_acid_atoms = []
    for atoms_list in ChemicalData().aa2long:
        residue_dictionary = {}
        for index, atom in enumerate(atoms_list):
            if atom is not None:
                atom = atom.strip()
                residue_dictionary[atom] = index
        amino_acid_atoms.append(residue_dictionary)
    return amino_acid_atoms


def get_protein_tip_atom_polar_mask() -> torch.Tensor:
    is_hbond_donor_mask = torch.zeros(
        (ChemicalData().UNKINDEX + 1, ChemicalData().NTOTAL), dtype=torch.bool
    )
    amino_acid_atoms = create_amino_acid_atom_dictionary()
    for amino_acid, atoms in AMINO_ACID_POLAR_TIP_ATOMS.items():
        for atom in atoms:
            residue_index = ChemicalData().aa2num[amino_acid]
            atom_index = amino_acid_atoms[residue_index][atom]
            is_hbond_donor_mask[residue_index, atom_index] = True
    return is_hbond_donor_mask


def get_ligand_obmol(
    ligand: List[Tuple],
    chains: Dict[str, Any],
    asmb_xfs: List[Tuple[str, np.ndarray]],
    covale: List[Tuple[Tuple[str, ...], Tuple[str, ...]]],
    lig_xf: List[Tuple[str, int]],
) -> Tuple[int, int]:
    lig_atoms, lig_bonds = get_ligand_atoms_bonds(ligand, chains, covale)
    lig_ch2xf = dict(lig_xf)

    xyz_sm, mask_sm, _, _, lig_akeys = cif_ligand_to_xyz(lig_atoms, asmb_xfs, lig_ch2xf)
    obmol, bond_feats = cif_ligand_to_obmol(xyz_sm, lig_akeys, lig_atoms, lig_bonds)
    return obmol, xyz_sm, mask_sm, bond_feats


def get_is_polar_mask(obmol) -> Tuple[torch.Tensor, torch.Tensor]:
    is_polar_mask = torch.zeros((obmol.NumAtoms(),), dtype=torch.bool)

    for atom_index in range(1, obmol.NumAtoms() + 1):
        atom = obmol.GetAtom(atom_index)
        if (
            atom.GetAtomicNum() == 7
            or atom.GetAtomicNum() == 8
            or atom.GetAtomicNum() == 9
        ):
            is_polar_mask[atom_index - 1] = True
    return is_polar_mask


def get_polar_contacts(
    xyz_sm: torch.Tensor,
    mask_sm: torch.Tensor,
    obmol: OBMol,
    xyz_prot: torch.Tensor,
    mask_prot: torch.Tensor,
    seq_prot: torch.Tensor,
    polar_contact_distance_cutoff: float = 3.6,
) -> int:
    polar_tip_atom_mask = get_protein_tip_atom_polar_mask()
    protein_polar_tip_atom_mask = polar_tip_atom_mask[seq_prot]
    protein_polar_tip_atom_mask = protein_polar_tip_atom_mask & mask_prot

    ligand_is_polar_mask = get_is_polar_mask(obmol)
    ligand_is_polar_mask = ligand_is_polar_mask & mask_sm

    polar_polar_distance_tensor = torch.cdist(
        xyz_sm[ligand_is_polar_mask],
        xyz_prot[protein_polar_tip_atom_mask],
    )

    num_polar_contacts = (
        (polar_polar_distance_tensor < polar_contact_distance_cutoff).sum().item()
    )
    return num_polar_contacts


def get_ligand_diameter(bond_feats: torch.Tensor) -> int:
    adj_mat = bond_feats > 0
    graph = nx.from_numpy_array(adj_mat.numpy())
    tree = nx.maximum_spanning_tree(graph)
    return nx.diameter(tree)


def compute_fraction_atoms_in_convex_hull(
    ligand_coordinates: torch.Tensor, protein_coordinates: torch.Tensor
) -> bool:
    protein_ca_coordinates = protein_coordinates[:, 1, :]
    triangulation = Delaunay(protein_ca_coordinates)
    is_in_ca_hull = [
        triangulation.find_simplex(ligand).item() > 0 for ligand in ligand_coordinates
    ]
    return sum(is_in_ca_hull) / len(is_in_ca_hull)


def get_soi_ligands_from_pdb_id(pdb_id: str) -> Set[str]:
    """get_soi_ligands_from_pdb_id This function takes a PDB ID and
    returns a set of ligand names that are annotated as subject of
    investigation (SOI) in the PDB. Such ligands are often considered
    biologically meaningful.

    Args:
        pdb_id (str): The PDB ID to query.

    Returns:
        Set[str]: A set of ligand names that are annotated as SOI in the PDB.
    """
    try:
        pdb_id = pdb_id.lower()
        core_response = requests.get(
            f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
        )
        core_json_dict = core_response.json()
        nonpolymer_ids = core_json_dict["rcsb_entry_container_identifiers"].get(
            "non_polymer_entity_ids", []
        )
        soi_ligand_names = []

        for nonpolymer_id in nonpolymer_ids:
            ligand_response = requests.get(
                f"https://data.rcsb.org/rest/v1/core/nonpolymer_entity/{pdb_id}/{nonpolymer_id}"
            )
            ligand_json_dict = ligand_response.json()
            ligand_name = ligand_json_dict["pdbx_entity_nonpoly"]["comp_id"]

            if "rcsb_nonpolymer_entity_annotation" in ligand_json_dict:
                for annotation_dict in ligand_json_dict[
                    "rcsb_nonpolymer_entity_annotation"
                ]:
                    if (
                        annotation_dict["type"] == "SUBJECT_OF_INVESTIGATION"
                        and annotation_dict["name"] == ligand_name
                    ):
                        soi_ligand_names.append(ligand_name)
                        break
    except Exception:
        return set()
    return set(soi_ligand_names)


def get_is_metal(chain: cifutils.Chain) -> bool:
    if len(chain.atoms) != 1:
        return False
    atom = next(iter(chain.atoms.values()))
    return atom.metal


def get_is_coordinated(
    xyz_sm: torch.Tensor,
    mask_sm: torch.Tensor,
    xyz_prot: torch.Tensor,
    mask_prot: torch.Tensor,
    coordinating_distance: float = 2.6,
    min_coordinating: int = 3,
) -> bool:
    all_dists = torch.cdist(xyz_sm[mask_sm], xyz_prot[mask_prot])
    num_coordinating = (all_dists < coordinating_distance).sum()
    return (num_coordinating >= min_coordinating).item()


def get_criterion(
    ligand: List[Tuple],
    primary_protein_chain: str,
    chains: Dict[str, Any],
    asmb_xfs: List[Tuple[str, np.ndarray]],
    covale: List[Tuple[Tuple[str, ...], Tuple[str, ...]]],
    lig_xf: List[Tuple[str, int]],
    query_ligand_name: str,
    soi_ligand_names: Set[str],
    **kwargs,
) -> Dict[str, float]:
    polar_contact_distance_cutoff = kwargs.get("polar_contact_distance_cutoff", 3.6)
    coordinating_distance = kwargs.get("coordinating_distance", 2.6)
    min_coordinating = kwargs.get("min_coordinating", 3)

    obmol, xyz_sm, mask_sm, bond_feats_sm = get_ligand_obmol(
        ligand, chains, asmb_xfs, covale, lig_xf
    )
    try:
        ligand_diameter = get_ligand_diameter(bond_feats_sm)
    except nx.exception.NetworkXError:
        ligand_diameter = 0.0

    is_soi = query_ligand_name in soi_ligand_names
    is_metal = get_is_metal(chains[lig_xf[0][0]])

    prot_chain_xf = None
    for transform in asmb_xfs:
        if transform[0] == primary_protein_chain:
            prot_chain_xf = transform
            break
    if prot_chain_xf is None:
        return {
            "QLIG_POLAR_CONTACTS": 0,
            "QLIG_FRACTION_HULL": 0.0,
            "QLIG_DIAMETER": ligand_diameter,
            "QLIG_IS_SOI": is_soi,
            "QLIG_IS_COORDINATED": False,
            "QLIG_IS_METAL": is_metal,
        }

    xyz_prot, mask_prot, seq_prot, _, _, _ = cif_poly_to_xyz(
        chains[primary_protein_chain], prot_chain_xf
    )
    mask_sm = mask_sm.bool()
    mask_prot = mask_prot.bool()
    seq_prot = seq_prot.long()
    seq_prot[seq_prot > ChemicalData().UNKINDEX] = ChemicalData().UNKINDEX

    try:
        is_coordinated = get_is_coordinated(
            xyz_sm,
            mask_sm,
            xyz_prot,
            mask_prot,
            coordinating_distance,
            min_coordinating,
        )
    except Exception:
        is_coordinated = False

    try:
        num_polar_contacts = get_polar_contacts(
            xyz_sm,
            mask_sm,
            obmol,
            xyz_prot,
            mask_prot,
            seq_prot,
            polar_contact_distance_cutoff,
        )
    except Exception:
        num_polar_contacts = 0

    try:
        fraction_atoms_in_hull = compute_fraction_atoms_in_convex_hull(xyz_sm, xyz_prot)
    except Exception:
        fraction_atoms_in_hull = 1.0

    return {
        "QLIG_POLAR_CONTACTS": num_polar_contacts,
        "QLIG_FRACTION_HULL": fraction_atoms_in_hull,
        "QLIG_DIAMETER": ligand_diameter,
        "QLIG_IS_SOI": is_soi,
        "QLIG_IS_COORDINATED": is_coordinated,
        "QLIG_IS_METAL": is_metal,
    }
