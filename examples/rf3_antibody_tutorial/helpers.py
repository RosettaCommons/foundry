"""Helper utilities for the RF3 Antibody/Antigen Tutorial."""

import numpy as np
import matplotlib.pyplot as plt
import py3Dmol

import biotite.structure as struc
from biotite.structure import superimpose, rmsd
from atomworks.io.parser import parse
from atomworks.io.utils.io_utils import to_pdb_string, to_cif_file
from atomworks.io.utils.sequence import aa_chem_comp_3to1

AA3_TO_1 = aa_chem_comp_3to1(standard_only=True)



def load_structure(path: str) -> struc.AtomArray:
    """Load a structure file and return the first model as an AtomArray."""
    s = parse(path, model=1)["asym_unit"][0]
    return s[~np.any(np.isnan(s.coord), axis=1)]


def extract_sequence(atom_array: struc.AtomArray, chain_id: str | None = None) -> str:
    """Extract one-letter amino acid sequence from CA atoms."""
    if chain_id is not None:
        atom_array = atom_array[atom_array.chain_id == chain_id]
    ca = atom_array[atom_array.atom_name == "CA"]
    return "".join(AA3_TO_1.get(str(r), "X") for r in ca.res_name)



def _align_ca(reference: struc.AtomArray, mobile: struc.AtomArray):
    """Extract CA atoms matched by common residue IDs."""
    ref_ca = reference[reference.atom_name == "CA"]
    mob_ca = mobile[mobile.atom_name == "CA"]
    common = sorted(set(ref_ca.res_id.tolist()) & set(mob_ca.res_id.tolist()))
    if not common:
        n = min(len(ref_ca), len(mob_ca))
        return ref_ca[:n], mob_ca[:n]
    return ref_ca[np.isin(ref_ca.res_id, common)], mob_ca[np.isin(mob_ca.res_id, common)]


def superimpose_and_rmsd(
    reference: struc.AtomArray,
    mobile: struc.AtomArray,
) -> tuple[struc.AtomArray, float]:
    """Superimpose mobile onto reference using CA atoms, return (fitted_mobile, CA RMSD)."""
    ref_ca, mob_ca = _align_ca(reference, mobile)
    fitted_ca, transform = superimpose(ref_ca, mob_ca)
    fitted_all = transform.apply(mobile)
    return fitted_all, float(rmsd(ref_ca, fitted_ca))


def per_residue_rmsd(
    reference: struc.AtomArray,
    mobile: struc.AtomArray,
) -> np.ndarray:
    """Per-residue CA distance after superimposition."""
    ref_ca, mob_ca = _align_ca(reference, mobile)
    fitted_ca, _ = superimpose(ref_ca, mob_ca)
    return np.sqrt(np.sum((ref_ca.coord - fitted_ca.coord) ** 2, axis=1))




def print_confidence_summary(conf: dict, chain_labels: list[str]) -> None:
    """Print pLDDT, iPTM, and pairwise min iPAE."""
    s = conf["summary"]
    print(f"pLDDT: {s['overall_plddt']:.3f}  |  iPTM: {s['iptm']:.3f}")
    pae_min = s["chain_pair_pae_min"]
    for i in range(len(chain_labels)):
        for j in range(i + 1, len(chain_labels)):
            if pae_min[i][j] is not None:
                print(f"  min iPAE {chain_labels[i]} – {chain_labels[j]}: {pae_min[i][j]:.1f}")


def best_ab_ag_ipae(pae_min: list[list[float | None]], labels: list[str]) -> float | None:
    """Best (lowest) iPAE between antibody and antigen chains, skipping VH–VL."""
    vals = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if pae_min[i][j] is not None:
                pair = f"{labels[i]}-{labels[j]}"
                if not ("VH" in pair and "VL" in pair):
                    vals.append(pae_min[i][j])
    return min(vals) if vals else None



def view_by_plddt(atom_array: struc.AtomArray):
    """Visualize structure colored by B-factor (pLDDT) using py3Dmol."""
    pdb_str = to_pdb_string(atom_array)
    viewer = py3Dmol.view(width=800, height=600)
    viewer.addModel(pdb_str, "pdb")
    viewer.setStyle({"cartoon": {"colorscheme": {"prop": "b", "gradient": "roygb", "min": 0.5, "max": 1.0}}})
    viewer.zoomTo()
    return viewer


def view_overlay(struct1: struc.AtomArray, struct2: struc.AtomArray,
                 label1: str = "Prediction", label2: str = "Reference"):
    """Overlay two structures for visual comparison."""
    viewer = py3Dmol.view(width=800, height=600)
    viewer.addModel(to_pdb_string(struct1), "pdb")
    viewer.addModel(to_pdb_string(struct2), "pdb")
    viewer.setStyle({"model": 0}, {"cartoon": {"color": "#1f77b4"}})
    viewer.setStyle({"model": 1}, {"cartoon": {"color": "#ff7f0e", "opacity": 0.7}})
    viewer.zoomTo()
    return viewer



def plot_pae(
    pae_matrix: np.ndarray,
    chain_breaks: list[int] | None = None,
    chain_labels: list[str] | None = None,
    title: str = "Predicted Aligned Error (PAE)",
    vmax: float = 30.0,
):
    """Plot a PAE heatmap with optional chain boundary annotations."""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(pae_matrix, cmap="Greens_r", vmin=0, vmax=vmax, aspect="equal")

    if chain_breaks:
        for brk in chain_breaks:
            ax.axhline(y=brk - 0.5, color="red", linewidth=1, linestyle="--")
            ax.axvline(x=brk - 0.5, color="red", linewidth=1, linestyle="--")

    if chain_labels and chain_breaks:
        boundaries = [0] + chain_breaks + [pae_matrix.shape[0]]
        for i, label in enumerate(chain_labels):
            mid = (boundaries[i] + boundaries[i + 1]) / 2
            ax.text(-3, mid, label, ha="right", va="center", fontsize=10, fontweight="bold")
            ax.text(mid, -3, label, ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xlabel("Scored residue")
    ax.set_ylabel("Aligned residue")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Expected position error (Angstroms)", shrink=0.8)
    plt.tight_layout()
    return fig, ax



def compute_contacts(
    atom_array: struc.AtomArray, chain1: str, chain2: str, cutoff: float = 8.0,
) -> set[tuple[int, int]]:
    """Compute residue-level contacts between two chains."""
    c1 = atom_array[(atom_array.chain_id == chain1) & (atom_array.element != "H")]
    c2 = atom_array[(atom_array.chain_id == chain2) & (atom_array.element != "H")]
    cell_list = struc.CellList(c2, cutoff)
    contacts = set()
    for i in range(len(c1)):
        neighbors = cell_list.get_atoms(c1.coord[i], cutoff)
        if len(neighbors) > 0:
            for j in neighbors:
                contacts.add((int(c1.res_id[i]), int(c2.res_id[j])))
    return contacts


def compare_contacts(
    pred_contacts: set[tuple[int, int]], ref_contacts: set[tuple[int, int]],
) -> dict:
    """Compare predicted vs reference contacts. Returns precision, recall, F1."""
    tp = len(pred_contacts & ref_contacts)
    fp = len(pred_contacts - ref_contacts)
    fn = len(ref_contacts - pred_contacts)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"true_positives": tp, "false_positives": fp, "false_negatives": fn,
            "precision": precision, "recall": recall, "f1": f1}
