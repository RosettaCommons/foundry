import os
import string
import subprocess
from datetime import datetime
from typing import Any, Tuple

import numpy as np
from atomworks.ml.transforms._checks import (
    check_atom_array_annotation,
    check_contains_keys,
    check_is_instance,
)
from atomworks.ml.transforms.base import Transform
from biotite.structure import AtomArray
from biotite.structure.io.pdb import PDBFile


def save_atomarray_to_pdb(atom_array, output_path):
    def _handle_nan_coords(atom_array, noise_level=1e-3):
        coords = atom_array.coord
        nan_mask = np.isnan(coords)
        coords[nan_mask] = np.random.uniform(
            -noise_level, noise_level, size=nan_mask.sum()
        )
        atom_array.coord = coords
        return atom_array, nan_mask

    atom_array, nan_mask = _handle_nan_coords(atom_array)

    chain_iids = np.unique(atom_array.chain_iid)
    if len(chain_iids) > 52:
        raise ValueError(
            "Too many chain_iids, cannot convert to PDB", "skipping HBPLUS"
        )

    all_possible_chainIDS = string.ascii_letters
    chain_map = {}
    for item in chain_iids:
        if len(item) == 1:
            chain_map[item] = item
            all_possible_chainIDS = all_possible_chainIDS.replace(item, "")
    for item in chain_iids:
        if len(item) > 1:
            chain_map[item] = all_possible_chainIDS[0]
            all_possible_chainIDS = all_possible_chainIDS.replace(chain_map[item], "")

    new_chain_ids = [chain_map[i] for i in atom_array.chain_iid]
    inverted_chain_map = {v: k for k, v in chain_map.items()}
    atom_array.chain_id = new_chain_ids
    atom_array.b_factor = np.zeros(len(atom_array))

    pdb = PDBFile()
    pdb.set_structure(atom_array)
    pdb.write(output_path)

    return atom_array, nan_mask, inverted_chain_map


def check_atom_array_has_hydrogen(data: dict[str, Any]):
    if not np.any(data["atom_array"].element == "H"):
        raise ValueError("Key `atom_array` in data has no hydrogens.")


def calculate_hbonds(
    atom_array: AtomArray,
    cutoff_HA_dist: float = 3,
    cutoff_DA_distance: float = 3.5,
) -> Tuple[np.ndarray, np.ndarray, AtomArray]:
    dtstr = datetime.now().strftime("%Y%m%d%H%M%S")
    pdb_path = f"{dtstr}_{np.random.randint(10000)}.pdb"
    atom_array, nan_mask, chain_map = save_atomarray_to_pdb(atom_array, pdb_path)

    hbplus_exe = os.environ.get("HBPLUS_PATH")

    if hbplus_exe is None or hbplus_exe == "":
        raise ValueError(
            "HBPLUS_PATH environment variable not set. "
            "Please set it to the path of the hbplus executable in order to calculate hydrogen bonds."
        )

    subprocess.call(
        [
            hbplus_exe,
            "-h",
            str(cutoff_HA_dist),
            "-d",
            str(cutoff_DA_distance),
            pdb_path,
            pdb_path,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    HB = open(pdb_path.replace("pdb", "hb2"), "r").readlines()
    hbonds = []
    for i in range(8, len(HB)):
        d_chain = HB[i][0]
        d_resi = str(int(HB[i][1:5].strip()))
        d_resn = HB[i][6:9].strip()
        d_ins = HB[i][5].replace("-", " ")
        d_atom = HB[i][9:13].strip()
        a_chain = HB[i][14]
        a_resi = str(int(HB[i][15:19].strip()))
        a_ins = HB[i][19].replace("-", " ")
        a_resn = HB[i][20:23].strip()
        a_atom = HB[i][23:27].strip()
        dist = float(HB[i][27:32].strip())

        items = {
            "d_chain": chain_map[d_chain],
            "d_resi": d_resi,
            "d_resn": d_resn,
            "d_ins": d_ins,
            "d_atom": d_atom,
            "a_chain": chain_map[a_chain],
            "a_resi": a_resi,
            "a_resn": a_resn,
            "a_ins": a_ins,
            "a_atom": a_atom,
            "dist": dist,
        }
        hbonds.append(items)

    donor_array = np.zeros(len(atom_array))
    acceptor_array = np.zeros(len(atom_array))
    donor_mask = np.bool_(donor_array)
    acceptor_mask = np.bool_(acceptor_array)

    motif_hbonds = []
    for item in hbonds:
        current_donor_mask = (
            (atom_array.chain_iid == item["d_chain"])
            & (atom_array.res_id == float(item["d_resi"]))
            & (atom_array.atom_name == item["d_atom"])
        )
        current_acceptor_mask = (
            (atom_array.chain_iid == item["a_chain"])
            & (atom_array.res_id == float(item["a_resi"]))
            & (atom_array.atom_name == item["a_atom"])
        )

        # Ensure that we can uniquely identify the donor and acceptor atoms
        if current_donor_mask.sum() != 1:
            raise ValueError(
                f"Unable to uniquely identify a donor atom with chain_iid={item['d_chain']}, res_id={item['d_resi']}, atom_name={item['d_atom']}."
            )
        if current_acceptor_mask.sum() != 1:
            raise ValueError(
                f"Unable to uniquely identify an acceptor atom with chain_iid={item['a_chain']}, res_id={item['a_resi']}, atom_name={item['a_atom']}."
            )

        current_donor_is_motif = atom_array.is_motif_atom[current_donor_mask][0]
        current_acceptor_is_motif = atom_array.is_motif_atom[current_acceptor_mask][0]

        # Only keep hbonds between the motif and diffused regions
        if current_donor_is_motif != current_acceptor_is_motif:
            motif_hbonds.append(item)
            donor_mask |= current_donor_mask
            acceptor_mask |= current_acceptor_mask

    donor_array[donor_mask] = 1
    acceptor_array[acceptor_mask] = 1

    os.remove(pdb_path)
    os.remove(pdb_path.replace("pdb", "hb2"))
    atom_array.set_annotation("active_donor", donor_array)
    atom_array.set_annotation("active_acceptor", acceptor_array)

    return atom_array, motif_hbonds, len(motif_hbonds)


class CalculateHbondsPlus(Transform):
    """Transform for calculating Hbonds, expects an AtomArray containing hydrogens."""

    def __init__(
        self,
        cutoff_HA_dist: float = 3,
        cutoff_DA_distance: float = 3.5,
    ):
        self.cutoff_HA_dist = cutoff_HA_dist
        self.cutoff_DA_distance = cutoff_DA_distance

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["res_name"])
        # check_atom_array_has_hydrogen(data)

    def forward(self, data: dict) -> dict:
        atom_array: AtomArray = data["atom_array"]

        atom_array, hbonds, _ = calculate_hbonds(
            atom_array,
            cutoff_HA_dist=self.cutoff_HA_dist,
            cutoff_DA_distance=self.cutoff_DA_distance,
        )

        data.setdefault("log_dict", {})
        log_dict = data["log_dict"]

        hbond_types = np.vstack((atom_array.active_donor, atom_array.active_acceptor)).T

        final_hbond_types = hbond_types
        final_hbond_types[:, 0] *= np.array(atom_array.is_motif_atom)
        final_hbond_types[:, 1] *= np.array(atom_array.is_motif_atom)
        log_dict["hbond_total_count"] = np.sum(final_hbond_types)

        if data["conditions"]["hbond_subsample"] and np.sum(final_hbond_types) > 3:
            base_fraction = 0.1
            max_fraction = 0.9
            n_hbonds = np.sum(final_hbond_types)
            max_hbonds = 50

            fraction = max_fraction - (max_fraction - base_fraction) * min(
                n_hbonds / max_hbonds, 1.0
            )
            final_hbond_types = subsample_one_hot_np(final_hbond_types, fraction)

        atom_array.set_annotation("active_donor", final_hbond_types[:, 0])
        atom_array.set_annotation("active_acceptor", final_hbond_types[:, 1])
        log_dict["hbond_subsample_atoms"] = np.sum(final_hbond_types)

        data["log_dict"] = log_dict
        data["atom_array"] = atom_array

        return data


def subsample_one_hot_np(array, fraction):
    if not (0 < fraction <= 1):
        raise ValueError("Fraction must be in the range (0, 1].")

    array = array.copy()
    one_indices = np.argwhere(array == 1)
    num_ones = len(one_indices)
    keep_count = int(num_ones * fraction)

    np.random.shuffle(one_indices)
    keep_indices = one_indices[:keep_count]

    new_array = np.zeros_like(array)
    for i, j in keep_indices:
        new_array[i, j] = 1

    return new_array
