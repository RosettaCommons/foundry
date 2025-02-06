from typing import Dict, List, Optional, Tuple

import pandas as pd
from openbabel import pybel
from tqdm import tqdm

from rf2aa.chemical import load_pdb_ideal_sdf_strings
from rf2aa.data.data_loader import parse_mol


def get_mol(id: str, all_small_molecules: Dict[str, str]) -> Optional[pybel.Molecule]:
    if id not in all_small_molecules:
        return None
    obmol, _, _, _, _ = parse_mol(
        all_small_molecules[id], filetype="sdf", string=True, find_automorphs=False
    )
    mol = pybel.Molecule(obmol)
    return mol


def get_molecule_lengths_and_fingerprints(
    molecule_ids: List[str], all_small_molecules: Dict[str, str]
) -> Dict[str, Tuple[int, pybel.Fingerprint]]:
    id_to_length_and_fingerprint = {}
    for mol_id in tqdm(
        molecule_ids, "Reading molecule smile strings for clustering..."
    ):
        mol = get_mol(mol_id, all_small_molecules)
        if mol is None:
            continue
        fingerprint = mol.calcfp()
        length = len(mol.atoms)
        id_to_length_and_fingerprint[mol_id] = (length, fingerprint)
    return id_to_length_and_fingerprint


def get_sim(
    id1: str,
    id2: str,
    id_to_length_and_fingerprint: Dict[str, Tuple[int, pybel.Fingerprint]],
    min_length: int = 5,
) -> float:
    if (
        id1 not in id_to_length_and_fingerprint
        or id2 not in id_to_length_and_fingerprint
    ):
        return 0.0

    length1, fingerprint1 = id_to_length_and_fingerprint[id1]
    length2, fingerprint2 = id_to_length_and_fingerprint[id2]

    if length1 < min_length and length2 < min_length:
        return 1.0
    else:
        return fingerprint1 | fingerprint2


def cluster_for_each_protein_cluster_ligand_tanimoto(
    df: pd.DataFrame, sim_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Given a DataFrame with an existing "CLUSTER" column, this function will split
    existing clusters into sub-clusters based on ligand similarity. The algorithm
    is as follows:
    1. Sort the DataFrame by "CLUSTER" and "LIGAND_ID".
    2. For each cluster, start a new sub-cluster with the first ligand.
    3. For each subsequent ligand, compare it to all ligand "representatives" in the
        in the current cluster.
    4. If the ligand is similar to any of the representatives, assign it to the same
        sub-cluster.
    5. If the ligand is not similar to any of the representatives, start a new sub-cluster.
    6. Update the "CLUSTER" column with the new sub-clusters.

    Args:
        df: DataFrame with columns "CLUSTER", "LIGAND" and "LIGAND_ID".
        sim_threshold: Similarity threshold for clustering ligands.
    Returns:
        DataFrame with updated "CLUSTER" column.
    """
    df["LIGAND_ID"] = df["LIGAND"].apply(lambda x: x[0][-1])

    all_small_molecules_dict = load_pdb_ideal_sdf_strings(return_only_sdf_strings=True)
    id_to_length_and_fingerprint = get_molecule_lengths_and_fingerprints(
        df["LIGAND_ID"].unique(), all_small_molecules_dict
    )

    sorted_df = df.sort_values(by=["CLUSTER", "LIGAND_ID"])
    current_cluster = sorted_df.iloc[0]["CLUSTER"]
    new_cluster_index = 0
    ligand_cluster_dict = {}
    new_clusters = []

    for _, row in tqdm(
        sorted_df.iterrows(), total=len(sorted_df), desc="Clustering by ligands..."
    ):
        if row["CLUSTER"] != current_cluster:
            current_cluster = row["CLUSTER"]
            new_cluster_index += 1
            new_clusters.append(new_cluster_index)
            ligand_cluster_dict = {}
        else:
            ligand_id = row["LIGAND_ID"]
            found_match = False
            for rep_id, rep_cluster in ligand_cluster_dict.items():
                if (
                    get_sim(rep_id, ligand_id, id_to_length_and_fingerprint)
                    > sim_threshold
                ):
                    new_clusters.append(rep_cluster)
                    found_match = True
                    break

            if not found_match:
                new_cluster_index += 1
                new_clusters.append(new_cluster_index)
                ligand_cluster_dict[ligand_id] = new_cluster_index

    sorted_df["CLUSTER"] = new_clusters
    return sorted_df


# Each function should take in a data frame and optional kwargs and return a new data frame
# with updated cluster assignments in the "CLUSTER" column.
cluster_factory = {
    "by_protein_sequence": lambda x: x,
    "cluster_for_each_protein_cluster_ligand_tanimoto": cluster_for_each_protein_cluster_ligand_tanimoto,
}
