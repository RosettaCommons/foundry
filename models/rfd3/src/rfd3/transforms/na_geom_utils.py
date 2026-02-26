import os
import subprocess
from datetime import datetime
from typing import Dict, Optional
import math
import numpy as np
import biotite.structure as struc
from biotite.structure import AtomArray

from atomworks.constants import (
    STANDARD_AA, 
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

from rfd3.constants import (
ATOM_REGION_BY_RESI,
PLANAR_ATOMS_BY_RESI,
)
import tempfile

# Derived: True when the residue has any planar sidechain atoms
HAS_PLANAR_SC = {res: bool(atoms) for res, atoms in PLANAR_ATOMS_BY_RESI.items()}

DEFAULT_NA_SS_FEATURE_INFO: dict[str, int] = {
    "NA_SS_MASK": 0,
    "NA_SS_PAIR": 1,
    "NA_SS_LOOP": 2,
}

AA_PLANAR_ATOMS = sorted(set(
    atom for res in STANDARD_AA if res in PLANAR_ATOMS_BY_RESI
    for atom in PLANAR_ATOMS_BY_RESI[res]
))

NA_PLANAR_ATOMS = sorted(set(
    atom for res in (*STANDARD_RNA, *STANDARD_DNA) if res in PLANAR_ATOMS_BY_RESI
    for atom in PLANAR_ATOMS_BY_RESI[res]
))


class NucMolInfo:
    """Constants and parameters for nucleic-acid geometry and interaction scoring.

    All parameters are set to empirically validated defaults. No constructor
    arguments are currently accepted.
    """

    def __init__(self) -> None:


        # Hbond interaction-class indices of the `hbond_count`` array:
        # `hbond_count`` array is (L, L, 3), where the last dimension 
        # encodes interaction type between tokens i & j
        self.BB_BB = 0 # backbone-backbone hbond interactions
        self.BB_SC = 1 # backbone-sidechain hbond interactions
        self.SC_SC = 2 # sidechain-sidechain hbond interactions

        # We sum over the last dimension of the hbond_count array, scaling 
        # count by the following weights to get the interaction score:
        self.bp_weight_BB_BB = 0.0
        self.bp_weight_BB_SC = 0.5
        self.bp_weight_SC_SC = 1.0
        self.bp_summation_weights = [self.bp_weight_BB_BB,
                                     self.bp_weight_BB_SC,
                                     self.bp_weight_SC_SC]

        # Parameters fo sigmoid function that gives us a continuous step function for 
        # meeting basepair interaction criteria based on hbond counts alone (1st filter).
        # Calibrated such that:
        # >= 2 base-base H-bonds -> ~1.0
        # 1 base-base H-bond + 1 base-backbone H-bond -> ~0.5
        self.min_hbonds_for_bp = 2.0
        self.bp_hbond_coeff    = 9.8 # determined heuristically
        self.bp_val_cutoff     = 0.5 # minimum basepairing score for binarizing basepairs when needed

        self.base_geometry_limits = {}
        self.base_geometry_limits['D_ij'] = 16.0
        self.base_geometry_limits['H_ij'] = 1.5
        self.base_geometry_limits['P_ij'] = math.pi/5
        self.base_geometry_limits['B_ij'] = math.pi/5

        self.rep_atom_dict={"protein": "CA", "rna": "C1'", "dna": "C1'"}

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



def calculate_hb_counts(
    atom_array: AtomArray,
    token_level_data: dict,
    mol_info: NucMolInfo,
    cutoff_HA_dist: float = 2.5,
    cutoff_DA_dist: float = 3.9,
    ):
    """Count hydrogen bonds between residue pairs using HBPLUS.

    Args:
        atom_array: Structure to analyse.
        token_level_data: Token-level metadata dict (must contain
            ``token_id_list`` and ``resi2index``).
        mol_info: Molecular-info object for backbone/sidechain atom lookup.
        cutoff_HA_dist: H–A distance cutoff (Å) passed to HBPLUS.
        cutoff_DA_dist: D–A distance cutoff (Å) passed to HBPLUS.

    Returns:
        np.ndarray of shape ``(I, I, 3)`` (int32) where the last axis
        encodes: 0 = BB–BB, 1 = BB–SC, 2 = SC–SC H-bond counts.
    """
    hbplus_exe = os.environ.get("HBPLUS_PATH")

    if hbplus_exe is None or hbplus_exe == "":
        raise ValueError(
            "HBPLUS_PATH environment variable not set. "
            "Please set it to the path of the hbplus executable in order to calculate hydrogen bonds."
        )
    with tempfile.TemporaryDirectory() as tmpdir:
        dtstr = datetime.now().strftime("%Y%m%d%H%M%S")
        pdb_filename = f"{dtstr}_{np.random.randint(10000)}.pdb"
        pdb_path = os.path.join(tmpdir, pdb_filename)
        atom_array, nan_mask, chain_map = save_atomarray_to_pdb(atom_array, pdb_path)

        subprocess.call(
            [
                hbplus_exe,
                "-h",
                str(cutoff_HA_dist),
                "-d",
                str(cutoff_DA_dist),
                pdb_path,
                pdb_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=tmpdir,
        )

        num_resis_total = len(token_level_data["token_id_list"])

        hbond_count = np.zeros((num_resis_total, num_resis_total, 3), dtype=np.int32)

        hb2_path = pdb_path.replace("pdb", "hb2")
        if not os.path.exists(hb2_path):
            print("WARNING: HB2 file could not be found; skipping NA SS metric")
            return hbond_count
        with open(hb2_path, "r") as hb2_f:
            for i, line in enumerate(hb2_f):
                if i < 8:
                    continue
                if len(line) < 28:
                    continue

                d_chain_iid = chain_map[line[0]]
                d_resi = int(line[1:5].strip())
                d_resn = line[6:9].strip()
                d_atom_name = line[9:13].strip()

                # Initialize donor/acceptor sidechain/backbone flags:
                # then replace with True if valid for summation
                d_is_sc = False
                d_is_bb = False
                a_is_sc = False
                a_is_bb = False

                d_mask = (
                    (atom_array.atom_name == d_atom_name)
                    & (atom_array.res_name == d_resn)
                    & (atom_array.res_id == d_resi)
                    & (atom_array.chain_iid == d_chain_iid)
                )
                # d_atm = atom_array[d_mask]
                # d_idx = d_atm.token_id
                d_idx = token_level_data["resi2index"].get(f"{d_chain_iid}__{d_resi}", None)
                if d_idx is None:
                    continue

                # Handle standard polymer residues for donor atom:
                if d_resn in ATOM_REGION_BY_RESI.keys():
                    d_is_sc = (d_atom_name in ATOM_REGION_BY_RESI[d_resn]['sc'])
                    d_is_bb = (d_atom_name in ATOM_REGION_BY_RESI[d_resn]['bb'])
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
                a_idx = token_level_data["resi2index"].get(f"{a_chain_iid}__{a_resi}", None)
                if a_idx is None:
                    continue

                # Handle standard polymer residues for acceptor atom:
                if a_resn in ATOM_REGION_BY_RESI.keys():
                    a_is_sc = (a_atom_name in ATOM_REGION_BY_RESI[a_resn]['sc'])
                    a_is_bb = (a_atom_name in ATOM_REGION_BY_RESI[a_resn]['bb'])
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
    '''    
    try:
        os.remove(pdb_path)
        os.remove(hb2_path)
    except:
        print("temp pdb/hb already removed or not created to begin with")
    '''
    return hbond_count


def find_planar_positions(
                atom_array: AtomArray, 
                mol_info: NucMolInfo,
                tol: float = 1e-2,
                ) -> Dict:
    """Identify residues with planar sidechains via known atom lists or PCA plane-fitting.

    For canonical residues the planar atoms are looked up from ``mol_info``;
    for non-canonical residues a plane is fitted to the four tip-most sidechain
    atoms, and all atoms within *tol* of that plane are returned.

    Args:
        atom_array: Structure to analyse.
        mol_info: Molecular-info object supplying per-residue planar atom lists.
        tol: Distance tolerance (Å) from the fitted plane for an atom to be
            considered planar.

    Returns:
        Dictionary ``{(chain_iid, res_id): [atom_name, ...]}`` mapping each
        unique residue position to its list of planar sidechain atom names.
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
        if res_name in PLANAR_ATOMS_BY_RESI.keys():
            # Shared atoms between residue and known planar atoms for that residue type:
            planar_atom_list = list(
                set([atm.atom_name for atm in res_atoms]) & 
                set(PLANAR_ATOMS_BY_RESI[res_name])
                )
            planar_atom_list_dict[(chain_iid, res_id)] = planar_atom_list

        # If unknown or noncanonical residue, compute planar atoms geometrically:
        else:
            candidate_planar_atm_names = []
            candidate_planar_atm_coords = []

            for atm in res_atoms:
                # Can pre-filter protein planar atoms:
                if atm.is_protein and (atm.atom_name in AA_PLANAR_ATOMS):
                    candidate_planar_atm_names.append(atm.atom_name)
                    candidate_planar_atm_coords.append(atm.coord)
                # Can pre-filter nucleic acid planar atoms:
                elif (atm.is_rna or atm.is_dna) and (atm.atom_name in NA_PLANAR_ATOMS):
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


def make_coord_list(atom_array: AtomArray, 
                    residue_list: list[str], 
                    chain_list: list[str],  
                    atom_list: list[str],
                    ) -> list[list[str]]:
    """Extract per-residue representative coordinates from an AtomArray.

    All three input lists must have the same length. Missing atoms are
    filled with ``[NaN, NaN, NaN]``.

    Args:
        atom_array: Biotite AtomArray to query.
        residue_list: Residue IDs (one per token).
        chain_list: Chain identifiers (one per token).
        atom_list: Atom names to extract (use ``"atomized"`` to take the
            first atom of the residue).

    Returns:
        List of ``[x, y, z]`` coordinate lists, same length as input.
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
    seq_cutoff = 2,
    gap_length = 200
) -> dict:
    """Build lightweight token-level metadata (no coordinate geometry).

    Sufficient for SS reconstruction, loop labeling from ``bp_partners``,
    and inference-time SS specification parsing. For geometry keys
    (``xyz_planar``, ``frame_xyz``, ``M_i``), follow up with
    :func:`add_token_level_geometry_data`.

    Args:
        atom_array: Structure to analyse.
        mol_info: Molecular-info constants.
        NA_only: If True, restrict filter_mask to nucleic-acid tokens.
        planar_only: If True, restrict filter_mask to tokens with planar
            sidechains.
        seq_cutoff: Sequence-distance threshold for the ``seq_neighbors``
            boolean mask.
        gap_length: Artificial gap inserted between chains for relative
            sequence position computation.

    Returns:
        Dict with keys: ``token_starts``, ``token_index``, ``is_na``,
        ``is_planar``, ``chain_list``, ``chain_iid_list``, ``resi_list``,
        ``resn_list``, ``token_id_list``, ``resi2index``, ``len_s``,
        ``seq_neighbors``, ``na_inds``, ``na_tensor_inds``,
        ``filter_mask``, ``rep_atom_list``, ``S_start_atom_list``,
        ``S_stop_atom_list``, ``include_geometry`` (False).
    """

    # Use residue starts (not token starts) so atomized atoms within one residue
    # map to a single NA-SS position.
    token_starts = struc.get_residue_starts(atom_array)
    token_level_array = atom_array[token_starts]

    token_index = np.arange(len(token_starts))

    # molecule type flags
    # Instantiate encoding locally to avoid retaining large arrays at module scope.
    sequence_encoding = AF3SequenceEncoding()
    is_protein = np.isin(token_level_array.res_name, sequence_encoding.all_res_names[sequence_encoding.is_aa_like])
    is_rna = np.isin(token_level_array.res_name, sequence_encoding.all_res_names[sequence_encoding.is_rna_like])
    is_dna = np.isin(token_level_array.res_name, sequence_encoding.all_res_names[sequence_encoding.is_dna_like])

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

        if atm.is_polymer and (atm.res_name in HAS_PLANAR_SC.keys()):
            sc_planarity_list.append(bool(HAS_PLANAR_SC[atm.res_name]))
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
    chn_bias = -gap_length
    for r, c in zip(resi_list, chain_iid_list):
        if c != current_chain:
            chn_bias += gap_length
            current_chain = c
        rel_pos_list.append(int(r + chn_bias))

    rel_pos = np.asarray(rel_pos_list, dtype=np.int64)
    seq_neighbors = (
        np.abs(rel_pos[:, None] - rel_pos[None, :]) <= int(seq_cutoff)
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
    """Augment token-level metadata with coordinate-derived geometry fields.

    Populates ``xyz_planar``, ``xyz_S_start``, ``xyz_S_stop``,
    ``frame_xyz``, ``M_i`` and updates ``is_planar`` / ``filter_mask``
    using coordinate-derived planarity. Sets ``include_geometry=True``.

    No-ops if geometry was already computed.

    Args:
        atom_array: Structure to extract coordinates from.
        mol_info: Molecular-info constants.
        token_level_data: Dict produced by :func:`get_token_level_metadata`
            (modified in-place and returned).
        NA_only: Restrict filter_mask to nucleic-acid tokens.
        planar_only: Restrict filter_mask to tokens with planar sidechains.

    Returns:
        The same ``token_level_data`` dict, augmented with geometry keys.
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

    planar_atom_list_dict = find_planar_positions(atom_array, mol_info)  # {(chain_iid, res_id): [atom_name, ...]}
    has_planar_sc: list[bool] = []

    xyz_planar: list[list[list[float]]] = []    # list[I] of [K_i, 3]  (K_i varies per residue)
    xyz_S_start: list[list[float]] = []         # list[I] of [3]
    xyz_S_stop: list[list[float]] = []          # list[I] of [3]

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
    frame_xyz = np.asarray(                                                            # [I, 3]  representative-atom coordinates
        make_coord_list(atom_array, resi_list, chain_list, rep_atom_list),
        dtype=np.float32,
    )

    padded_centers = np.concatenate([frame_xyz[:1], frame_xyz, frame_xyz[-1:]], axis=0) # [I+2, 3]
    M_i = (                                                                            # [I, 3]  smoothed backbone-direction vectors
        (padded_centers[1:-1] - padded_centers[:-2])
        + (padded_centers[2:] - padded_centers[1:-1])
    ) / 2.0

    is_planar_arr = np.asarray(has_planar_sc, dtype=bool)                               # [I]
    token_level_data["is_planar"] = is_planar_arr

    is_na_arr = np.asarray(token_level_data["is_na"], dtype=bool)                       # [I]
    if NA_only and planar_only:
        filter_mask = is_na_arr & is_planar_arr
    elif NA_only and (not planar_only):
        filter_mask = is_na_arr.copy()
    elif (not NA_only) and planar_only:
        filter_mask = is_planar_arr.copy()
    else:
        filter_mask = np.ones_like(is_na_arr, dtype=bool)
    token_level_data["filter_mask"] = filter_mask                                      # [I]  bool

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


# ---------------------------------------------------------------------------
# Sub-calculations used by compute_nucleic_ss
# ---------------------------------------------------------------------------


def _compute_local_frames(
    xyz_planar: list[np.ndarray],
    planar_centers: np.ndarray,
    M_i: np.ndarray,
    *,
    xyz_S_start: list | None = None,
    xyz_S_stop: list | None = None,
    compute_full_frame: bool = False,
    eps: float = 1e-8,
) -> dict[str, np.ndarray]:
    """Build per-residue local coordinate frames from planar sidechain atoms.

    The base-normal direction Z_i is always computed via PCA on the planar
    atom cloud, corrected for backbone direction.  When *compute_full_frame*
    is True the sugar-edge vector is used to derive X_i and Y_i as well.

    Args:
        xyz_planar: Per-residue planar-atom coordinates, list[I] of [K_i, 3].
        planar_centers: Sidechain planar-atom centroids, [I, 3].
        M_i: Backbone-direction vectors, [I, 3].
        xyz_S_start: Sugar-edge start coordinates, list[I] of [3].
            Required when *compute_full_frame* is True.
        xyz_S_stop: Sugar-edge stop coordinates, list[I] of [3].
            Required when *compute_full_frame* is True.
        compute_full_frame: If True, also compute X_i and Y_i.
        eps: Small constant for numerical stability.

    Returns:
        Dict with ``"Z_i"`` (always), and ``"X_i"``, ``"Y_i"`` when
        *compute_full_frame* is True.  Each array has shape ``[I, 3]``.
    """
    n_tokens = len(xyz_planar)

    # Mean-centre the planar atoms per residue
    centered_points = [                                # list[I] of [K_i, 3]
        np.asarray(xyz_i, dtype=np.float32) - cen_i
        for xyz_i, cen_i in zip(xyz_planar, planar_centers)
    ]

    # PCA → eigenvectors per residue
    eigenvectors = np.full((n_tokens, 3, 3), np.nan, dtype=np.float32)  # [I, 3, 3]

    for i, xyz_i in enumerate(centered_points):
        xyz_i = xyz_i[~np.isnan(xyz_i).any(axis=1)]
        if xyz_i.shape[0] >= 3:
            cov_matrix = np.einsum("ij,ik->jk", xyz_i, xyz_i) / max(  # [3, 3]
                xyz_i.shape[0] - 1, 1
            )
            _, eigvecs = np.linalg.eigh(cov_matrix)  # [3, 3]
            eigenvectors[i] = eigvecs

    # Base-normal: smallest-eigenvalue direction, corrected for backbone dir
    N_i = eigenvectors[:, :, 0]  # [I, 3]
    N_i = N_i / (np.linalg.norm(N_i, axis=1, keepdims=True) + eps)

    Z_i = N_i * np.sum(M_i * N_i, axis=-1, keepdims=True)  # [I, 3]
    Z_i = Z_i / (np.linalg.norm(Z_i, axis=-1, keepdims=True) + eps)

    result: dict[str, np.ndarray] = {"Z_i": Z_i}

    if compute_full_frame:
        if xyz_S_start is None or xyz_S_stop is None:
            raise ValueError("xyz_S_start and xyz_S_stop are required for full frame")

        X_s_i = (  # [I, 3]  sugar-edge direction
            np.asarray(xyz_S_stop, dtype=np.float32)
            - np.asarray(xyz_S_start, dtype=np.float32)
        )
        X_s_i = X_s_i / (np.linalg.norm(X_s_i, axis=-1, keepdims=True) + eps)

        X_i = np.cross(Z_i, X_s_i)  # [I, 3]
        X_i = X_i / (np.linalg.norm(X_i, axis=-1, keepdims=True) + eps)
        result["X_i"] = X_i

        Y_i = np.cross(X_i, Z_i)  # [I, 3]
        Y_i = Y_i / (np.linalg.norm(Y_i, axis=-1, keepdims=True) + eps)
        result["Y_i"] = Y_i

    return result


def _compute_pairwise_geometry(
    Z_i: np.ndarray,
    frame_D_ij_vec: np.ndarray,
    sc_D_ij_vec: np.ndarray,
    *,
    X_i: np.ndarray | None = None,
    clamp: bool = True,
    compute_opening: bool = False,
    eps: float = 1e-8,
) -> dict[str, np.ndarray]:
    """Compute pairwise base-step geometry between all residue pairs.

    Derives the pairwise coordinate frame (X_ij, Y_ij, Z_ij) and the
    base-pair geometry parameters: rise (H_ij), buckle (B_ij), propeller
    (P_ij), and optionally opening angle (O_ij).

    Args:
        Z_i: Per-residue base-normal vectors, [I, 3].
        frame_D_ij_vec: Pairwise backbone displacement vectors, [I, I, 3].
        sc_D_ij_vec: Pairwise sidechain-centroid displacement vectors, [I, I, 3].
        X_i: Per-residue local X-axis, [I, 3].  Required when
            *compute_opening* is True.
        clamp: Clamp cosines to [-1, 1] before ``arccos``.
        compute_opening: If True, compute opening angle O_ij.
        eps: Small constant for numerical stability.

    Returns:
        Dict with keys ``"H_ij"`` [I, I], ``"B_ij"`` [I, I],
        ``"P_ij"`` [I, I], ``"base_ori_ij"`` [I, I],
        ``"X_ij"`` [I, I, 3], ``"Y_ij"`` [I, I, 3],
        ``"Z_ij"`` [I, I, 3], and optionally ``"O_ij"`` [I, I].
    """
    # Orientation-selected pairwise Z-axis
    Z_sum = Z_i[:, None, :] + Z_i[None, :, :]    # [I, I, 3]
    Z_diff = Z_i[:, None, :] - Z_i[None, :, :]   # [I, I, 3]
    Z_ij_oris = 0.5 * np.stack((Z_sum, Z_diff), axis=0)  # [2, I, I, 3]

    base_ori_ij = (  # [I, I]  0=parallel, 1=antiparallel
        np.linalg.norm(Z_ij_oris[1], axis=-1) > np.linalg.norm(Z_ij_oris[0], axis=-1)
    ).astype(np.int64)

    Z_ij = np.where(base_ori_ij[..., None] == 0, Z_ij_oris[0], Z_ij_oris[1])  # [I, I, 3]
    Z_ij = Z_ij / (np.linalg.norm(Z_ij, axis=-1, keepdims=True) + eps)

    # Pairwise Y (inter-residue direction) and X axes
    Y_ij = frame_D_ij_vec / (np.linalg.norm(frame_D_ij_vec, axis=-1, keepdims=True) + eps)  # [I, I, 3]
    X_ij = np.cross(Z_ij, Y_ij)  # [I, I, 3]
    X_ij = X_ij / (np.linalg.norm(X_ij, axis=-1, keepdims=True) + eps)

    # Rise (H_ij)
    H_ij = np.sum(sc_D_ij_vec * Z_ij, axis=-1)  # [I, I]
    D_ij = np.linalg.norm(sc_D_ij_vec, axis=-1)  # [I, I]

    # Buckle (B_ij)
    proj_Z_i_YZ = (  # [I, I, 3]
        np.sum(Z_i[:, None, :] * Y_ij, axis=-1, keepdims=True) * Y_ij
        + np.sum(Z_i[:, None, :] * Z_ij, axis=-1, keepdims=True) * Z_ij
    )
    proj_Z_i_YZ_norm = proj_Z_i_YZ / (np.linalg.norm(proj_Z_i_YZ, axis=-1, keepdims=True) + eps)
    cos_buckle = np.sum(proj_Z_i_YZ_norm * (-proj_Z_i_YZ_norm.swapaxes(0, 1)), axis=-1)  # [I, I]

    # Propeller (P_ij)
    proj_Z_i_ZX = (  # [I, I, 3]
        np.sum(Z_i[:, None, :] * Z_ij, axis=-1, keepdims=True) * Z_ij
        + np.sum(Z_i[:, None, :] * X_ij, axis=-1, keepdims=True) * X_ij
    )
    proj_Z_i_ZX_norm = proj_Z_i_ZX / (np.linalg.norm(proj_Z_i_ZX, axis=-1, keepdims=True) + eps)
    cos_propeller = np.sum(proj_Z_i_ZX_norm * (-proj_Z_i_ZX_norm.swapaxes(0, 1)), axis=-1)  # [I, I]

    if clamp:
        cos_buckle = np.clip(cos_buckle, -1.0, 1.0)
        cos_propeller = np.clip(cos_propeller, -1.0, 1.0)

    B_ij = np.arccos(cos_buckle)    # [I, I]
    P_ij = np.arccos(cos_propeller)  # [I, I]

    result: dict[str, np.ndarray] = {
        "H_ij": H_ij,
        "B_ij": B_ij,
        "P_ij": P_ij,
        "D_ij": D_ij,
        "base_ori_ij": base_ori_ij,
        "X_ij": X_ij,
        "Y_ij": Y_ij,
        "Z_ij": Z_ij,
    }

    # Opening angle (O_ij) — purely diagnostic
    if compute_opening:
        if X_i is None:
            raise ValueError("X_i is required to compute opening angle")

        proj_X_i_XY = (  # [I, I, 3]
            np.sum(X_i[:, None, :] * X_ij, axis=-1, keepdims=True) * X_ij
            + np.sum(X_i[:, None, :] * Y_ij, axis=-1, keepdims=True) * Y_ij
        )
        proj_X_i_XY_norm = proj_X_i_XY / (np.linalg.norm(proj_X_i_XY, axis=-1, keepdims=True) + eps)
        cos_opening = np.sum(proj_X_i_XY_norm * proj_X_i_XY_norm.swapaxes(0, 1), axis=-1)  # [I, I]
        if clamp:
            cos_opening = np.clip(cos_opening, -1.0, 1.0)
        result["O_ij"] = np.arccos(cos_opening)  # [I, I]

    return result


def _compute_basepair_mask(
    hbond_count: np.ndarray,
    seq_neighbors: np.ndarray,
    H_ij: np.ndarray,
    B_ij: np.ndarray,
    P_ij: np.ndarray,
    D_ij: np.ndarray,
    mol_info,
    *,
    bool_only: bool = False,
    eps: float = 1e-8,
) -> dict[str, np.ndarray] | np.ndarray:
    """Identify base pairs by combining H-bond scores with geometry filters.

    Computes a sigmoid-based base-pair probability from weighted H-bond
    counts and gates it with rise / buckle / propeller geometry limits.

    Args:
        hbond_count: H-bond counts, [I, I, 3] (BB-BB / BB-SC / SC-SC).
        seq_neighbors: Sequence-neighbor boolean mask, [I, I].
        H_ij: Rise displacement, [I, I].
        B_ij: Buckle angle (radians), [I, I].
        P_ij: Propeller angle (radians), [I, I].
        mol_info: Molecular-info object with ``bp_summation_weights``,
            ``bp_hbond_coeff``, ``min_hbonds_for_bp``, ``bp_val_cutoff``,
            and ``base_geometry_limits``.
        bool_only: If True, return only the boolean mask array.
        eps: Small constant for numerical stability.

    Returns:
        If *bool_only*: ``np.ndarray`` of shape ``(I, I)`` (bool).
        Otherwise: dict with ``"basepairs_bool_ij"`` [I, I] (bool),
        ``"basepairs_ij"`` [I, I] (float), and
        ``"hbond_summation"`` [I, I] (float).
    """
    hbond_summation = np.tensordot(  # [I, I]
        hbond_count.astype(np.float32),
        np.asarray(mol_info.bp_summation_weights, dtype=np.float32),
        axes=([2], [0]),
    )

    logits = mol_info.bp_hbond_coeff * (  # [I, I]
        hbond_summation - (mol_info.min_hbonds_for_bp - 1)
    )
    bp_preds = (1.0 / (1.0 + np.exp(-logits))) + eps  # [I, I]

    # Geometry filters
    H_ij_filter = (  # [I, I]
        (H_ij >= -mol_info.base_geometry_limits["H_ij"])
        & (H_ij <= mol_info.base_geometry_limits["H_ij"])
    )
    B_ij_filter = (  # [I, I]
        (B_ij <= mol_info.base_geometry_limits["B_ij"])
        | (B_ij >= math.pi - mol_info.base_geometry_limits["B_ij"])
    )
    P_ij_filter = (  # [I, I]
        (P_ij <= mol_info.base_geometry_limits["P_ij"])
        | (P_ij >= math.pi - mol_info.base_geometry_limits["P_ij"])
    )

    D_ij_filter = (D_ij <= mol_info.base_geometry_limits["D_ij"])

    bp_geom_filter = H_ij_filter & B_ij_filter & P_ij_filter & D_ij_filter  # [I, I]

    if bool_only:
        basepairs_bool_ij = (  # [I, I]
            (~seq_neighbors) & bp_geom_filter
            & (bp_preds >= float(mol_info.bp_val_cutoff))
        )
        return basepairs_bool_ij

    basepairs_ij = (  # [I, I]
        (~seq_neighbors).astype(np.float32)
        * bp_geom_filter.astype(np.float32)
        * bp_preds.astype(np.float32)
    )
    basepairs_bool_ij = basepairs_ij >= mol_info.bp_val_cutoff  # [I, I]

    return {
        "basepairs_bool_ij": basepairs_bool_ij,
        "basepairs_ij": basepairs_ij,
        "hbond_summation": hbond_summation,
    }


def compute_nucleic_ss(
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
    """Compute nucleic-acid pairwise base-pair geometry and filters.

    Operates in two modes:

    * **Fast annotation** (default / ``return_basepairs_only=True``): returns
      only ``basepairs_bool_ij`` and frees intermediate arrays.
    * **Diagnostic**: additionally returns local/pairwise geometry when
      ``return_pairwise_geometry``, ``return_local_params``, or
      ``return_opening_angle`` are set.

    Args:
        mol_info: Molecular-info constants (geometry limits, H-bond weights).
        token_level_data: Token-level dict with geometry (from
            :func:`add_token_level_geometry_data`).
        hbond_count: H-bond count array, shape ``(I_full, I_full, 3)``.
        clamp_pairwise_params: Clamp cosines to [-1, 1] before ``arccos``.
        eps: Small constant for numerical stability.
        return_local_params: Return per-residue X/Y/Z local frames.
        return_pairwise_geometry: Return pairwise X_ij/Y_ij/Z_ij arrays.
        return_opening_angle: Return pairwise opening angle O_ij.
        return_basepairs_only: Return only the boolean base-pair mask
            (fastest path).

    Returns:
        If ``return_basepairs_only``: ``np.ndarray`` of shape ``(I, I)``
        (bool) — the base-pair boolean mask.

        Otherwise: dict ``{"pair_params": {...}, "local_params": {...}}``
        containing the requested geometry arrays (all shape ``(I, I)`` or
        ``(I, 3)``).
    """

    mask_1d = np.asarray(token_level_data["filter_mask"], dtype=bool)  # [I_full]

    # --- Unpack and filter token-level data ----------------------
    M_i         = np.asarray(token_level_data["M_i"], dtype=np.float32)[mask_1d]        # [I, 3]
    frame_xyz   = np.asarray(token_level_data["frame_xyz"], dtype=np.float32)[mask_1d]  # [I, 3]
    xyz_S_start = [v for v, k in zip(token_level_data["xyz_S_start"], mask_1d) if k]    # list[I] of [3]
    xyz_S_stop  = [v for v, k in zip(token_level_data["xyz_S_stop"],  mask_1d) if k]    # list[I] of [3]
    xyz_planar  = [v for v, k in zip(token_level_data["xyz_planar"],  mask_1d) if k]    # list[I] of [K_i, 3]

    hbond_count   = np.asarray(hbond_count)[mask_1d, :][:, mask_1d]                                      # [I, I, 3]
    seq_neighbors = np.asarray(token_level_data["seq_neighbors"], dtype=bool)[mask_1d, :][:, mask_1d]     # [I, I]

    # Nothing passed NA/planar filtering for this structure.
    # Return empty outputs instead of failing downstream on np.stack([]).
    if len(xyz_planar) == 0:
        if return_basepairs_only:
            return np.zeros((0, 0), dtype=bool)

        pair_params: dict[str, np.ndarray] = {
            "H_ij": np.zeros((0, 0), dtype=np.float32),
            "B_ij": np.zeros((0, 0), dtype=np.float32),
            "P_ij": np.zeros((0, 0), dtype=np.float32),
            "D_ij": np.zeros((0, 0), dtype=np.float32),
            "base_ori_ij": np.zeros((0, 0), dtype=np.float32),
            "basepairs_bool_ij": np.zeros((0, 0), dtype=bool),
            "basepairs_ij": np.zeros((0, 0), dtype=np.float32),
            "hbond_summation": np.zeros((0, 0), dtype=np.float32),
        }

        if return_opening_angle:
            pair_params["O_ij"] = np.zeros((0, 0), dtype=np.float32)

        if return_pairwise_geometry:
            pair_params["X_ij"] = np.zeros((0, 0), dtype=np.float32)
            pair_params["Y_ij"] = np.zeros((0, 0), dtype=np.float32)
            pair_params["Z_ij"] = np.zeros((0, 0), dtype=np.float32)

        nucleic_ss_data: dict = {"pair_params": pair_params}
        if return_local_params:
            nucleic_ss_data["local_params"] = {
                "X_i": np.zeros((0, 3), dtype=np.float32),
                "Y_i": np.zeros((0, 3), dtype=np.float32),
                "Z_i": np.zeros((0, 3), dtype=np.float32),
            }

        return nucleic_ss_data

    # --- Precompute centroids and displacement vectors -----------
    planar_centers = np.stack(  # [I, 3]
        [np.nanmean(np.asarray(xyz_i, dtype=np.float32), axis=0) for xyz_i in xyz_planar],
        axis=0,
    ).astype(np.float32)

    frame_D_ij_vec = frame_xyz[None, :, :] - frame_xyz[:, None, :]          # [I, I, 3]
    sc_D_ij_vec    = planar_centers[None, :, :] - planar_centers[:, None, :]  # [I, I, 3]

    # --- CALC I: per-residue local coordinate frames -------------
    need_full_frame = return_local_params or return_opening_angle
    local_frames = _compute_local_frames(
        xyz_planar,
        planar_centers,
        M_i,
        xyz_S_start=xyz_S_start if need_full_frame else None,
        xyz_S_stop=xyz_S_stop if need_full_frame else None,
        compute_full_frame=need_full_frame,
        eps=eps,
    )
    Z_i = local_frames["Z_i"]                           # [I, 3]
    X_i = local_frames.get("X_i")                       # [I, 3] or None

    # --- CALC II: pairwise base-step geometry --------------------
    pw_geom = _compute_pairwise_geometry(
        Z_i,
        frame_D_ij_vec,
        sc_D_ij_vec,
        X_i=X_i,
        clamp=clamp_pairwise_params,
        compute_opening=return_opening_angle,
        eps=eps,
    )

    # --- CALC III: base-pair identification ----------------------
    bp_result = _compute_basepair_mask(
        hbond_count,
        seq_neighbors,
        pw_geom["H_ij"],
        pw_geom["B_ij"],
        pw_geom["P_ij"],
        pw_geom["D_ij"],
        mol_info,
        bool_only=return_basepairs_only,
        eps=eps,
    )

    if return_basepairs_only:
        return bp_result  # np.ndarray [I, I] bool

    # --- Assemble output dict ------------------------------------
    assert isinstance(bp_result, dict)

    pair_params: dict[str, np.ndarray] = {
        "H_ij": pw_geom["H_ij"],
        "B_ij": pw_geom["B_ij"],
        "P_ij": pw_geom["P_ij"],
        "base_ori_ij": pw_geom["base_ori_ij"],
        "basepairs_bool_ij": bp_result["basepairs_bool_ij"],
        "basepairs_ij": bp_result["basepairs_ij"],
        "hbond_summation": bp_result["hbond_summation"],
    }

    if return_opening_angle and "O_ij" in pw_geom:
        pair_params["O_ij"] = pw_geom["O_ij"]

    if return_pairwise_geometry:
        pair_params["X_ij"] = pw_geom["X_ij"]
        pair_params["Y_ij"] = pw_geom["Y_ij"]
        pair_params["Z_ij"] = pw_geom["Z_ij"]

    nucleic_ss_data: dict = {"pair_params": pair_params}
    if return_local_params and "Y_i" in local_frames:
        nucleic_ss_data["local_params"] = {
            "X_i": local_frames["X_i"],
            "Y_i": local_frames["Y_i"],
            "Z_i": local_frames["Z_i"],
        }

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
    cutoff_HA_dist: float = 3.5,
    cutoff_DA_dist: float = 3.5,
) -> AtomArray:
    """Compute base pairs and write a ``bp_partners`` annotation onto *atom_array*.

    Uses H-bond counts and pairwise geometry filters to identify base pairs,
    then stores the result as a per-atom annotation with the following
    semantics:

    * ``[]`` — explicitly unpaired (loop)
    * ``[token_id, ...]`` — paired partner token IDs
    * ``None`` — unannotated / masked (non-NA or filtered-out tokens)

    Args:
        atom_array: Structure to annotate (modified in-place).
        NA_only: Restrict geometry filter to nucleic-acid tokens.
        planar_only: Restrict geometry filter to tokens with planar
            sidechains.
        p_canonical_bp_filter: Probability of discarding non-canonical
            base pairs (keeps only A–U, A–T, G–C).
        mol_info: Molecular-info constants; created if ``None``.
        overwrite: If False, merge with existing ``bp_partners``.
        token_level_data: Pre-computed metadata dict; augmented with
            geometry as needed.
        cutoff_HA_dist: H–A distance cutoff (Å) for HBPLUS.
        cutoff_DA_dist: D–A distance cutoff (Å) for HBPLUS.

    Returns:
        The same *atom_array* with the ``bp_partners`` annotation set.
    """

    if mol_info is None:
        mol_info = NucMolInfo()

    # Residue representatives (0..L-1) and their corresponding atom indices.
    # Keep this aligned with get_token_level_metadata(), which uses residue starts.
    if token_level_data is not None and "token_starts" in token_level_data:
        token_starts = np.asarray(token_level_data["token_starts"], dtype=int)
    else:
        token_starts = struc.get_residue_starts(atom_array)
    residue_start_end = np.concatenate([token_starts, [atom_array.array_length()]])
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
        cutoff_HA_dist=cutoff_HA_dist,
        cutoff_DA_dist=cutoff_DA_dist,
    )
    bp_bool = np.asarray(
        compute_nucleic_ss(
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

    # Apply optional filters
    if NA_only:
        bp_bool &= is_na_full[:, None]
        bp_bool &= is_na_full[None, :]
    if planar_only:
        n_tokens = bp_bool.shape[0]
        has_planar_sc = np.asarray(
            token_level_data.get("has_planar_sc", np.ones(n_tokens, dtype=bool)), dtype=bool
        )
        bp_bool &= has_planar_sc[:, None]
        bp_bool &= has_planar_sc[None, :]

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

    # Build residue-level annotation first, then spread to all atoms in each residue.
    if (not overwrite) and ("bp_partners" in atom_array.get_annotation_categories()):
        existing_ann = atom_array.bp_partners
        if len(existing_ann) != len(atom_array):
            raise ValueError("Existing bp_partners annotation has wrong length")
        residue_bp_partners = np.empty(len(token_starts), dtype=object)
        residue_bp_partners[:] = None
        for i, start in enumerate(token_starts.tolist()):
            residue_bp_partners[i] = existing_ann[int(start)]
    else:
        residue_bp_partners = np.empty(len(token_starts), dtype=object)
        residue_bp_partners[:] = None

    # Explicit-loop semantics:
    # - Only nucleic-acid token-start atoms *within subset_idxs* get a list container.
    # - [] means explicitly unpaired loop.
    # - None means unannotated/masked.
    for full_i in subset_idxs.tolist():
        if not bool(is_na_full[int(full_i)]):
            continue
        if residue_bp_partners[int(full_i)] is None:
            residue_bp_partners[int(full_i)] = []

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

        partner_i = int(token_ids[full_j])
        partner_j = int(token_ids[full_i])

        if residue_bp_partners[full_i] is None:
            residue_bp_partners[full_i] = []
        if residue_bp_partners[full_j] is None:
            residue_bp_partners[full_j] = []

        # Add if not present
        if partner_i not in residue_bp_partners[full_i]:
            residue_bp_partners[full_i].append(partner_i)
        if partner_j not in residue_bp_partners[full_j]:
            residue_bp_partners[full_j].append(partner_j)

    # Project residue-level annotations back to atom-level storage:
    # - atomized residues: spread to all atoms in that residue
    # - non-atomized residues: keep only on token-start representative atom
    bp_partners_ann = np.empty(len(atom_array), dtype=object)
    bp_partners_ann[:] = None
    for i, start in enumerate(token_starts.tolist()):
        stop = int(residue_start_end[i + 1])
        value = residue_bp_partners[i]
        if value is None:
            continue
        # A residue is treated as atomized if any atom in the residue carries atomize=True.
        if "atomize" in atom_array.get_annotation_categories():
            residue_is_atomized = bool(np.any(np.asarray(atom_array.atomize[int(start):stop], dtype=bool)))
        else:
            residue_is_atomized = False
        if residue_is_atomized:
            for atom_idx in range(int(start), stop):
                bp_partners_ann[atom_idx] = list(value)
        else:
            bp_partners_ann[int(start)] = list(value)

    atom_array.set_annotation("bp_partners", bp_partners_ann)
    return atom_array


def parse_dot_bracket(dot_bracket: str) -> tuple[list[tuple[int, int]], list[int]]:
    """Parse a dot-bracket string into base pairs and unpaired positions.

    Supports standard ``()``, ``[]``, ``{}``, ``<>`` and pseudoknot
    brackets ``A``–``E`` / ``a``–``e``.

    Args:
        dot_bracket: Dot-bracket notation string.

    Returns:
        Tuple of ``(pairs, unpaired)`` where *pairs* is a list of 0-based
        ``(i, j)`` index tuples and *unpaired* is a list of 0-based indices
        corresponding to ``.`` characters.
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
    """Write ``bp_partners`` annotation from an inference-time specification.

    Inference analogue of :func:`annotate_na_ss`: interprets user-provided
    dot-bracket strings and/or residue ranges rather than computing base
    pairs from geometry.

    Supported *specification* keys (all optional):

    * ``ss_dbn``: global dot-bracket string (applied to the first *L* tokens).
    * ``ss_dbn_dict``: ``{"<chain><start>-<end>": dbn_str, ...}``.
    * ``paired_region_list``: ``["A5-15,B1-11", ...]``.
    * ``paired_position_list``: ``["A19,A61,A20", ...]``.
    * ``loop_region_list``: ``["A5-10", ...]`` (forced unpaired).

    Args:
        atom_array: Structure to annotate (modified in-place).
        specification: Specification dict as described above.
        overwrite: If False, merge with existing ``bp_partners``.

    Returns:
        The same *atom_array* with the ``bp_partners`` annotation set.
    """

    spec = specification or {}
    token_starts = get_token_starts(atom_array)
    token_level_array = atom_array[token_starts]
    token_ids: list[int] = [int(t) for t in list(token_level_array.token_id)]
    n_tokens = len(token_starts)

    # Prepare/overwrite annotation array
    if (not overwrite) and ("bp_partners" in atom_array.get_annotation_categories()):
        bp_partners_ann = atom_array.bp_partners
        if len(bp_partners_ann) != len(atom_array):
            raise ValueError("Existing bp_partners annotation has wrong length")
    else:
        bp_partners_ann = np.empty(len(atom_array), dtype=object)
        bp_partners_ann[:] = None

    # Build chain/res -> token index map for region/position specs.
    # Accept both chain_iid-like keys (e.g. "A_1") and plain chain IDs (e.g. "A")
    # so CLI/json specs like "A1,B3" work reliably in inference.
    chain_iid_list: list[str] = [str(atm.chain_iid) for atm in token_level_array]
    chain_id_list: list[str] = [str(atm.chain_id) for atm in token_level_array]
    resi_list: list[int] = [int(atm.res_id) for atm in token_level_array]
    chain_res_to_tok: dict[tuple[str, int], int] = {}
    for i, (chain_iid, chain_id, res_id) in enumerate(
        zip(chain_iid_list, chain_id_list, resi_list)
    ):
        key_iid = (chain_iid, int(res_id))
        key_chain = (chain_id, int(res_id))
        chain_res_to_tok.setdefault(key_iid, int(i))
        chain_res_to_tok.setdefault(key_chain, int(i))
        # Also support the short alias from chain_iid (e.g. "A_1" -> "A")
        short_chain = chain_iid.split("_", 1)[0]
        chain_res_to_tok.setdefault((short_chain, int(res_id)), int(i))

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
        for j in list(partners[i]):
            partners[j].discard(i)
        partners[i].clear()

    # Write lists of partner token_ids onto token-start atoms.
    # Unspecified tokens remain unannotated (None) -> NA_SS_MASK.
    for i in range(n_tokens):
        atom_i = int(token_starts[i])
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
    """Annotate ``bp_partners`` from ``data["specification"]``.

    Convenience wrapper around :func:`annotate_na_ss_from_specification`.

    Args:
        data: Pipeline data dict containing ``atom_array`` and optionally
            ``specification``.
        overwrite: If False, merge with existing ``bp_partners``.

    Returns:
        The annotated AtomArray (also stored back in *data*).
    """
    atom_array = data["atom_array"]
    spec = data.get("specification", {}) or {}
    return annotate_na_ss_from_specification(atom_array, spec, overwrite=overwrite)
