import networkx as nx
import numpy as np
from biotite.structure.info import residue
from scipy.spatial.distance import cdist

from foundry.metrics.metric import Metric


def collapsing_virtual_atoms_batched(
    atom_arrays, central_atom, threshold=0.5, return_virtual_index=False
):
    """
    Apply collapsing_virtual_atoms to a batch of atom arrays.

    Parameters:
        atom_arrays (List[AtomArray]): Batch of atom arrays.
        central_atom (str): Atom to compute distance from (e.g., "CA").
        threshold (float): Distance threshold to identify virtual atoms.
        return_virtual_index (bool): Whether to also return the virtual mask.

    Returns:
        List of filtered atom arrays or (atom_array, mask) tuples
    """
    result = []
    for atom_array in atom_arrays:
        virtual_atom_mask = np.zeros(len(atom_array), dtype=bool)

        # We need to select residues by the combination of chain_iid and res_id.
        chain_iid_with_sep = np.char.add(atom_array.chain_iid, "|")
        chain_iid_and_res_id = np.char.add(
            chain_iid_with_sep, atom_array.res_id.astype(str)
        )
        atom_array.set_annotation("chain_iid_and_res_id", chain_iid_and_res_id)
        unique_residue_identifiers = np.unique(chain_iid_and_res_id)

        for res_identifier in unique_residue_identifiers:
            # ... Pick the current residue
            cur_mask = atom_array.chain_iid_and_res_id == res_identifier
            cur_residue = atom_array[cur_mask]
            cur_central_atom = central_atom

            # For Glycine: it doesn't have CB, so set the virtual atom as CA.
            # The current way to handle this is to check if predicted CA and CB are too close, because in the case of glycine and we pad virtual atoms based on CB, CB's coords are set as CA.
            # There might be a better way to do this.
            CA_coord = cur_residue.coord[cur_residue.atom_name == "CA"]
            CB_coord = cur_residue.coord[cur_residue.atom_name == "CB"]
            if np.linalg.norm(CA_coord - CB_coord) < threshold:
                cur_central_atom = "CA"

            central_mask = cur_residue.atom_name == cur_central_atom

            if not np.any(central_mask):
                continue

            # ... Calculate the distance to the central atom
            central_coord = cur_residue.coord[central_mask][
                0
            ]  # Should only have one central atom anyway
            dists = np.linalg.norm(cur_residue.coord - central_coord, axis=-1)

            # ... Select virtual atom by the distance. Shouldn't count the central atom itself. (F)
            is_virtual = (dists < threshold) & ~central_mask

            virtual_atom_mask[np.where(cur_mask)[0][is_virtual]] = True

        filtered = atom_array[~virtual_atom_mask]
        if return_virtual_index:
            result.append((filtered, virtual_atom_mask))
        else:
            result.append(filtered)

    return result


def construct_graph(coords, cutoff_min, cutoff_max):
    """
    Use coordinates to construct a NetworkX graph.
    Nodes = atom indices.
    Edges = distance-based inferred bonds.

    Parameters:
        coords: [n, 3]
        cutoff_min: min distance to consider a bond (avoid self-loops)
        cutoff_max: max distance to consider a bond (e.g., typical covalent bond)

    Returns:
        G: A NetworkX graph
    """
    dists = cdist(coords, coords)  # [N, N]

    G = nx.Graph()
    n_atoms = coords.shape[0]

    # ... Add nodes
    for i in range(n_atoms):
        G.add_node(i)

    # ... Add edges based on distance
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            if cutoff_min < dists[i, j] < cutoff_max:
                G.add_edge(i, j)

    return G


def are_graphs_isomorphic(g1, g2):
    """
    Check if two graphs are topologically isomorphic (ignoring atom/bond types).
    """
    return nx.is_isomorphic(g1, g2)


def check_sidechain_quality(atom_array, dist_threshold_min=1, dist_threshold_max=2):
    """
    Check sidechain quality. This is done by checking:
    (1) if a sidechain can map to a standard amino acid based on the topology;
    (2) if two sidechains has unexpected bond connection
    (3) if a sidechain itself has collapse
    A valid sidechain is defined by satisfying all the three rules.

    Return:
     - matched (dict): all possible standard amino acids that a sidechain can map to.
     - valid_sidechain_percent (float): percentage of valid sidechains.
     - unintended_bonds_percent (float): percentage of sidechains with unintended bonds with other sidechains.
     - clash_percent (float): percentage of sidechains that has collapse in itself.
    """
    # Step 1: Build standard amino acid graphs
    standard_aa = [
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLU",
        "GLN",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    ]

    standard_aa_atom_array = [residue(aa) for aa in standard_aa]

    # ... Remove OXT atoms and hydrogens
    standard_aa_atom_array = [
        aa[(~np.isin(aa.atom_name, np.array(["OXT"]))) & (aa.element != "H")]
        for aa in standard_aa_atom_array
    ]

    # ... Convert standard AA to topology graphs
    standard_aa_graphs = []
    for aa in standard_aa_atom_array:
        try:
            g = construct_graph(
                aa.coord, cutoff_min=dist_threshold_min, cutoff_max=dist_threshold_max
            )
            standard_aa_graphs.append(g)
        except Exception as e:
            print(f"Failed to convert {aa} to graph: {e}")
            standard_aa_graphs.append(None)

    # We need to select residues by the combination of chain_iid and res_id.
    chain_iid_with_sep = np.char.add(atom_array.chain_iid, "|")
    chain_iid_and_res_id = np.char.add(
        chain_iid_with_sep, atom_array.res_id.astype(str)
    )
    atom_array.set_annotation("chain_iid_and_res_id", chain_iid_and_res_id)
    unique_residue_identifiers = np.unique(chain_iid_and_res_id)
    matches = {}

    # ... Map predicted sidechain to any standard amino acids
    for res_identifier in unique_residue_identifiers:
        matches[res_identifier] = []
        cur_res_coords = atom_array.coord[
            atom_array.chain_iid_and_res_id == res_identifier
        ]

        try:
            cur_graph = construct_graph(
                cur_res_coords,
                cutoff_min=dist_threshold_min,
                cutoff_max=dist_threshold_max,
            )
        except Exception as e:
            print(
                f"[WARN] Could not build graph for chain_iid|res_id {res_identifier}: {e}"
            )
            continue

        for aa_idx, aa_graph in enumerate(standard_aa_graphs):
            if aa_graph is None:
                continue
            if are_graphs_isomorphic(cur_graph, aa_graph):
                matches[res_identifier].append(standard_aa[aa_idx])

    # Step 2: Check if the inter and intra-residue quality is good.
    # (1) Check if there are potential bonds between sidechains from different residues.
    # (2) Check if atoms are too close to collapse.

    coords = atom_array.coord
    residue_identifiers = atom_array.chain_iid_and_res_id

    # ... Mask sidechain atoms. Now the sidechain is any atoms except four backbone atoms
    is_sidechain = ~np.isin(atom_array.atom_name, np.array(["N", "CA", "C", "O"]))

    coords_sc = coords[is_sidechain]
    residue_identifiers_sc = residue_identifiers[is_sidechain]

    # ... Calculate pairwise distances
    dists = cdist(coords_sc, coords_sc)

    # ... Check if there are potential bonds between sidechains from different residues.
    unintended_bonds = {
        res_identifier: False for res_identifier in unique_residue_identifiers
    }
    N = dists.shape[0]

    # Only look at the upper triangle (exclude diagonal)
    iu, ju = np.triu_indices(N, k=1)

    # Apply distance threshold to identify any possible bonds
    potential_bonds = (dists[iu, ju] > dist_threshold_min) & (
        dists[iu, ju] < dist_threshold_max
    )

    # Check if atoms are from different residues
    diff_res_mask = residue_identifiers_sc[iu] != residue_identifiers_sc[ju]

    # Combine both masks
    bonds_mask = potential_bonds & diff_res_mask

    # ... Annotate residues with unintended bonds
    for idx in range(len(bonds_mask)):
        if bonds_mask[idx]:
            unintended_bonds[residue_identifiers_sc[iu[idx]]] = True
            unintended_bonds[residue_identifiers_sc[ju[idx]]] = True

    # ... Check if atoms are too close to be real
    clash_residues = {
        res_identifier: False for res_identifier in unique_residue_identifiers
    }
    clash_mask = dists[iu, ju] < dist_threshold_min
    for idx in range(len(clash_mask)):
        if clash_mask[idx]:
            clash_residues[residue_identifiers_sc[iu[idx]]] = True
            clash_residues[residue_identifiers_sc[ju[idx]]] = True

    # ... Output the final valid sidechains
    if_valid_sidechains = [
        (len(matches[res_identifier]) > 0)
        & (~unintended_bonds[res_identifier])
        & (~clash_residues[res_identifier])
        for res_identifier in unique_residue_identifiers
    ]
    if_unintended_bonds = [
        unintended_bonds[res_identifier]
        for res_identifier in unique_residue_identifiers
    ]
    if_clash = [
        clash_residues[res_identifier] for res_identifier in unique_residue_identifiers
    ]

    valid_sidechain_percent = sum(if_valid_sidechains) / len(unique_residue_identifiers)
    unintended_bonds_percent = sum(if_unintended_bonds) / len(
        unique_residue_identifiers
    )
    clash_percent = sum(if_clash) / len(unique_residue_identifiers)

    return matches, valid_sidechain_percent, unintended_bonds_percent, clash_percent


def compute_batched_sidechain_quality(
    predicted_atom_array_stack,
    central_atom,
    dist_threshold_min=1.0,
    dist_threshold_max=2.0,
    already_removed_virtual_atoms=False,
):
    """
    Compute sidechain metrics for each structure in a batch.
    """
    batch_metrics = []

    for atom_array in predicted_atom_array_stack:
        metrics = {}
        matches, valid, unintended, clash = check_sidechain_quality(
            atom_array, dist_threshold_min, dist_threshold_max
        )
        metrics["mapped_restype"] = matches
        metrics["valid_sidechain_percent"] = valid
        metrics["unintended_bonds_percent"] = unintended
        metrics["clash_percent"] = clash
        batch_metrics.append(metrics)
    return batch_metrics


class SidechainMetrics(Metric):
    def __init__(
        self,
        dist_threshold_min,
        dist_threshold_max,
        central_atom,
        already_removed_virtual_atoms=False,
    ):
        super().__init__()
        self.dist_threshold_min = dist_threshold_min
        self.dist_threshold_max = dist_threshold_max
        self.central_atom = central_atom
        self.already_removed_virtual_atoms = already_removed_virtual_atoms

    @property
    def kwargs_to_compute_args(self):
        return {
            "predicted_atom_array_stack": ("predicted_atom_array_stack",),
        }

    def compute(self, predicted_atom_array_stack):
        batch_metrics = compute_batched_sidechain_quality(
            predicted_atom_array_stack,
            self.central_atom,
            self.dist_threshold_min,
            self.dist_threshold_max,
            self.already_removed_virtual_atoms,
        )

        # Aggregate output for batch-level metrics
        o = {
            "mean_valid_sidechain_percent": float(
                np.mean([m["valid_sidechain_percent"] for m in batch_metrics])
            ),
            "mean_unintended_bonds_percent": float(
                np.mean([m["unintended_bonds_percent"] for m in batch_metrics])
            ),
            "mean_clash_percent": float(
                np.mean([m["clash_percent"] for m in batch_metrics])
            ),
            # "mapped_restype": [m["mapped_restype"] for m in batch_metrics],
        }
        return o
