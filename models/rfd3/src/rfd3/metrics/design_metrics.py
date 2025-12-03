import numpy as np
import torch
from atomworks.ml.utils.token import (
    get_token_starts,
)
from beartype.typing import Any
from rfd3.metrics.metrics_utils import (
    _flatten_dict,
    get_hotspot_contacts,
    get_ss_metrics_and_rg,
)

from foundry.common import exists
from foundry.metrics.metric import Metric

STANDARD_CACA_DIST = 3.8


def get_clash_metrics(
    atom_array,
    clash_threshold=1.5,
    ligand_clash_threshold=1.5,
    chainbreak_threshold=0.75,
):
    # HACK: For now, ligands are treated as any atomized residues
    is_ligand = np.logical_and(
        atom_array.is_ligand, ~atom_array.is_motif_atom_unindexed
    )

    def get_chainbreaks():
        ca_atoms = atom_array[atom_array.atom_name == "CA"]
        xyz = ca_atoms.coord
        xyz = torch.from_numpy(xyz)
        ca_dists = torch.norm(xyz[1:] - xyz[:-1], dim=-1)
        deviation = torch.abs(ca_dists - STANDARD_CACA_DIST)

        # Allow leniency for expected chain breaks (e.g. PPI)
        chain_breaks = ca_atoms.chain_iid[1:] != ca_atoms.chain_iid[:-1]
        deviation[chain_breaks] = 0

        is_chainbreak = deviation > chainbreak_threshold
        return {
            "max_ca_deviation": float(deviation.max(-1).values.mean()),
            "n_chainbreaks": int(is_chainbreak.sum()),
        }

    def get_interresidue_clashes(backbone_only=False):
        protein_array = atom_array[atom_array.is_protein]
        resid = protein_array.res_id - protein_array.res_id.min()
        xyz = protein_array.coord
        dists = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)  # N_atoms x N_atoms

        # Block out intra-residue distances
        mask = np.triu(np.ones_like(dists), k=1).astype(bool)
        block_mask = np.abs(resid[:, None] - resid[None, :]) <= 1
        mask[block_mask] = False
        dists[~mask] = 999

        if backbone_only:
            # Block out non-backbone atoms
            backbone_mask = np.isin(protein_array.atom_name, ["N", "CA", "C"])
            mask = backbone_mask[:, None] & backbone_mask[None, :]
            dists[~mask] = 999

        num_clashes_L = dists.min(axis=-1) < clash_threshold
        return int(num_clashes_L.sum())

    def get_ligand_clash_metrics():
        if not is_ligand.any():
            return {}

        # Clashes are any non-motif atom against any ligand atom
        xyz_ligand = atom_array[is_ligand].coord
        backbone_mask = np.isin(atom_array.atom_name, ["N", "CA", "C"]) & ~is_ligand
        xyz_diffused = atom_array[
            backbone_mask
            & ~atom_array.is_motif_atom_unindexed
            & ~atom_array.is_motif_atom_with_fixed_coord
        ].coord

        # If we have no diffused backbone atoms, return empty
        if xyz_diffused.shape[0] == 0:
            return {}

        diff = (
            xyz_diffused[:, None, :] - xyz_ligand[None, :, :]
        )  # (n_diffused, n_ligand, 3)
        dists_ligand = np.linalg.norm(diff, axis=-1)  # (n_diffused, n_ligand)
        dists = np.min(dists_ligand, axis=0)
        return {
            "n_clashing.ligand_clashes": int(np.sum(dists < ligand_clash_threshold)),
            "n_clashing.ligand_min_distance": float(np.min(dists)),
        }

    # Accumulate metrics
    o = {}
    o = o | get_chainbreaks()
    o["n_clashing.interresidue_clashes_w_sidechain"] = get_interresidue_clashes()
    o["n_clashing.interresidue_clashes_w_backbone"] = get_interresidue_clashes(
        backbone_only=True
    )
    o |= get_ligand_clash_metrics()
    return {k: v for k, v in o.items() if exists(v)}


def convert_to_float_or_str(o):
    """
    Converts elements of a dictionary to ensure all components are saveable with JSONs
    """
    for k, v in o.items():
        if not isinstance(v, (int, float, str, list)):
            try:
                o[k] = float(v)
            except Exception as e:
                raise ValueError(f"Unsupported type for key {k}: {type(v)}. Error: {e}")
    return o


def get_all_backbone_metrics(
    atom_array,
    verbose=True,
    compute_non_clash_metrics_for_diffused_region_only: bool = False,
):
    """
    Calculate metrics for the AtomArray

    The atom array coming in will be a cleaned atom array (no virtual atoms and corrected atom names)
    without guideposts
    """
    o = {}

    # ... Clash metrics
    o = o | get_clash_metrics(
        atom_array,
    )

    if verbose:
        if compute_non_clash_metrics_for_diffused_region_only:
            # Subset to diffused region only
            atom_array = atom_array[~atom_array.is_motif_atom_with_fixed_coord]

        # ... Add additional metrics
        o |= get_ss_metrics_and_rg(
            atom_array[~atom_array.is_motif_atom_with_fixed_coord]
        )

        # Basic compositional statistics
        starts = get_token_starts(atom_array)
        protein_starts = starts[atom_array.is_protein[starts]]
        o["alanine_content"] = np.mean(atom_array[protein_starts].res_name == "ALA")
        o["glycine_content"] = np.mean(atom_array[protein_starts].res_name == "GLY")
        o["num_residues"] = len(protein_starts)

        fixed = atom_array.is_motif_atom_with_fixed_coord
        o["diffused_com"] = np.mean(atom_array.coord[~fixed, :], axis=0).tolist()
        if np.any(fixed):
            o["fixed_com"] = np.mean(atom_array.coord[fixed, :], axis=0).tolist()

        # if "b_factor" in token_array.get_annotation_categories():
        #     m["sequence_entropy_mean"] = np.mean(token_array.b_factor)
        #     m["sequence_entropy_max"] = np.max(token_array.b_factor)
        #     m["sequence_entropy_min"] = np.min(token_array.b_factor)
        #     m["sequence_entropy_std"] = np.std(token_array.b_factor)

    # ... Ensure JSON-saveable
    o = convert_to_float_or_str(o)
    return o


class AtomArrayMetrics(Metric):
    """General metrics for the predicted atom array."""

    def __init__(
        self,
        compute_for_diffused_region_only: bool = False,
        compute_ss_adherence_if_possible: bool = False,
    ):
        super().__init__()
        self.clash_threshold = 1.2
        self.float_threshold = (
            3.0  # maximum closest-neighbour distance before considered a floating atom
        )
        self.standard_ca_dist = 3.8
        self.compute_for_diffused_region_only = compute_for_diffused_region_only
        self.compute_ss_adherence_if_possible = compute_ss_adherence_if_possible

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {
            "atom_array_stack": ("predicted_atom_array_stack"),
            "feats": ("network_input", "f"),
        }

    def compute(self, atom_array_stack, feats):
        o = {}

        for atom_array in atom_array_stack:
            # Subset to indexed tokens only
            atom_array = atom_array[~atom_array.is_motif_atom_unindexed]

            if self.compute_for_diffused_region_only:
                atom_array = atom_array[~atom_array.is_motif_atom_with_fixed_coord]

            # SS content and ROG
            if (
                self.compute_ss_adherence_if_possible
                and (
                    "is_helix_conditioning" in feats
                    and "is_sheet_conditioning" in feats
                    and "is_loop_conditioning" in feats
                )
                and (
                    feats["is_helix_conditioning"].sum() > 0
                    or feats["is_sheet_conditioning"].sum() > 0
                    or feats["is_loop_conditioning"].sum() > 0
                )
            ):
                ss_conditioning = {
                    "helix": feats["is_helix_conditioning"].cpu().numpy(),
                    "sheet": feats["is_sheet_conditioning"].cpu().numpy(),
                    "loop": feats["is_loop_conditioning"].cpu().numpy(),
                }
            else:
                ss_conditioning = None
            m = get_ss_metrics_and_rg(atom_array, ss_conditioning=ss_conditioning)

            # Subset to token level array for consistency
            token_array = atom_array[get_token_starts(atom_array)]

            # Basic compositional statistics
            m["alanine_content"] = np.mean(token_array.res_name == "ALA")
            m["glycine_content"] = np.mean(token_array.res_name == "GLY")

            # Sequence Confidence
            if "b_factor" in token_array.get_annotation_categories():
                m["sequence_entropy_mean"] = np.mean(token_array.b_factor)
                m["sequence_entropy_max"] = np.max(token_array.b_factor)
                m["sequence_entropy_min"] = np.min(token_array.b_factor)
                m["sequence_entropy_std"] = np.std(token_array.b_factor)

            # Write to o
            for k, v in m.items():
                if k not in o:
                    o[k] = []
                o[k].append(v)

        # Summarize stats
        for k, v in o.items():
            o[k] = float(np.mean(v))
        return o


class MetadataMetrics(Metric):
    """
    Fetches all floating point values from the prediction metadata
    """

    @property
    def kwargs_to_compute_args(self):
        return {
            "prediction_metadata": ("prediction_metadata",),
        }

    def compute(self, prediction_metadata):
        """ """
        if not prediction_metadata:
            return {}

        o = {}
        for idx, metadata in prediction_metadata.items():
            # Flatten dictionary
            metadata = _flatten_dict(metadata)

            # Update output dictionary
            for key, value in metadata.items():
                if isinstance(value, (int, float)):
                    if key not in o:
                        o[key] = []
                    o[key].append(value)

        # Reduce via mean
        o = {k: float(np.mean(v)) for k, v in o.items()}
        return o


class BackboneMetrics(Metric):
    def __init__(self, compute_for_diffused_region_only: bool = False):
        super().__init__()
        self.clash_threshold = 1.2
        self.float_threshold = (
            3.0  # maximum closest-neighbour distance before considered a floating atom
        )
        self.standard_ca_dist = 3.8
        self.compute_for_diffused_region_only = compute_for_diffused_region_only

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {
            "X_L": ("network_output", "X_L"),  # [D, L, 3]
            "tok_idx": ("network_input", "f", "atom_to_token_map"),
            "f": ("network_input", "f"),
        }

    def compute(self, X_L, tok_idx, f):
        o = {}
        xyz = X_L.detach().cpu().numpy()
        tok_idx = tok_idx.cpu().numpy()
        dists = np.linalg.norm(
            xyz[..., :, None, :] - xyz[..., None, :, :], axis=-1
        )  # N_atoms x N_atoms

        is_protein = f["is_protein"][tok_idx].cpu().numpy()  # n_atoms

        mask = np.zeros_like(dists, dtype=bool)
        mask = mask | (np.eye(dists.shape[-1], dtype=bool))[None]
        mask = mask | (tok_idx[:, None] == tok_idx[None, :])[None]
        mask = mask | ~(is_protein[:, None] & is_protein[None, :])[None]
        dists[mask] = 999

        num_clashes_L = (dists.min(axis=-1) < self.clash_threshold).astype(
            float
        )  # B, L
        o["frac_clashing"] = float(num_clashes_L.mean(-1).mean())
        o["n_clashing"] = float(num_clashes_L.sum(-1).mean())

        if "is_backbone" in f:
            is_backbone = f["is_backbone"].cpu().numpy()
            mask = np.zeros_like(dists, dtype=bool)
            mask = mask | (tok_idx[:, None] == tok_idx[None, :])[None]
            mask = mask | ~(is_backbone[:, None] & is_backbone[None, :])[None]
            dists[mask] = 999
            o["frac_backbone_clashing"] = float(
                (dists.min(axis=-1) < self.clash_threshold)
                .astype(float)
                .mean(-1)
                .mean()
            )
            o["n_backbone_clashing"] = float(
                (dists.min(axis=-1) < self.clash_threshold).astype(float).sum(-1).mean()
            )

        # We do this after clash detection, since that should consider both chains
        if self.compute_for_diffused_region_only:
            diffused_region = ~(f["is_motif_atom_with_fixed_coord"].cpu().numpy())
            xyz = xyz[:, diffused_region]
            tok_idx = tok_idx[diffused_region]

        # Num floating
        dists = np.linalg.norm(
            xyz[..., :, None, :] - xyz[..., None, :, :], axis=-1
        )  # N_atoms x N_atoms
        mask = np.zeros_like(dists, dtype=bool)
        mask = mask & np.eye(dists.shape[-1], dtype=bool)[None]
        dists[mask] = 999

        is_floating = dists.min(axis=-1) > self.float_threshold
        o["frac_floating"] = float(is_floating.mean(-1).mean())

        if "is_ca" in f:
            # Calculate CA
            is_ca = f["is_ca"].cpu().numpy()
            if self.compute_for_diffused_region_only:
                is_ca = is_ca[diffused_region]
                is_protein = is_protein[diffused_region]
            idx_mask = is_ca & is_protein
            if self.compute_for_diffused_region_only:
                xyz = X_L.cpu()[:, diffused_region][:, idx_mask]
            else:
                xyz = X_L.cpu()[:, idx_mask]

            ca_dists = torch.norm(xyz[:, 1:] - xyz[:, :-1], dim=-1)
            deviation = torch.abs(ca_dists - self.standard_ca_dist)  # B, (I-1)
            is_chainbreak = deviation > 0.75

            o["max_ca_deviation"] = float(deviation.max(-1).values.mean())
            o["fraction_chainbreaks"] = float(is_chainbreak.float().mean(-1).mean())
            o["n_chainbreaks"] = float(is_chainbreak.float().sum(-1).mean())

        return o


class PPIMetrics(Metric):
    """PPI-specific metrics"""

    def __init__(self, distance_cutoff: float = 4.5):
        super().__init__()
        self.distance_cutoff = distance_cutoff  # Distance cutoff for hotspot contacts

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {
            "atom_array_stack": ("predicted_atom_array_stack"),
            # "ppi_hotspots_mask": ("network_input", "f", "is_atom_level_hotspot"),
        }

    def compute(self, atom_array_stack):
        # Get the number of hotspots for which a diffused atom is within the distance cutoff
        metrics_dict = {"fraction_hotspots_contacted": []}
        for atom_array in atom_array_stack:
            ppi_hotspots_mask = atom_array.get_annotation(
                "is_atom_level_hotspot"
            ).astype(bool)
            if ppi_hotspots_mask.sum() == 0:
                continue

            fraction_contacted = get_hotspot_contacts(
                atom_array,
                hotspot_mask=ppi_hotspots_mask,
                distance_cutoff=self.distance_cutoff,
            )

            metrics_dict["fraction_hotspots_contacted"].append(fraction_contacted)

        fraction_contacted_array = np.array(metrics_dict["fraction_hotspots_contacted"])

        if fraction_contacted_array.size == 0:
            return {}

        return {"fraction_hotspots_contacted": float(np.mean(fraction_contacted_array))}


class SequenceMetrics(Metric):
    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {
            "S_I": ("network_output", "sequence_logits_I"),  # [D, I, K]
            "S_gt_I": ("extra_info", "seq_token_lvl"),  # [D, I]
        }

    def compute(self, S_I, S_gt_I):
        o = {}
        seq_head_pred = S_I.argmax(dim=-1)  # B, I
        seq_head_recovery = seq_head_pred == S_gt_I

        # Filter out unresolved residues
        seq_head_recovery = seq_head_recovery.float().mean()
        o["seq_head_recovery"] = float(seq_head_recovery.mean())

        # Calculate the confusion matrix
        seq_head_gt = S_gt_I[None].expand(seq_head_pred.shape[0], -1)  # B, I

        # One-hot encode predictions and ground truth
        seq_head_pred = S_I.clone()
        seq_head_pred = torch.nn.functional.softmax(seq_head_pred, dim=-1)  # (B, I, C)

        # Set any unresolve residues to be 31
        seq_head_gt = torch.nn.functional.one_hot(
            seq_head_gt, num_classes=S_I.shape[-1]
        ).float()  # (B, I, C)

        # Permute predictions to shape (B, C, I) for matmul
        seq_head_pred = seq_head_pred.permute(0, 2, 1)  # (B, C, I)

        # Compute confusion matrix per batch (B, C, C)
        confusion_matrix = torch.matmul(seq_head_pred, seq_head_gt)

        # Sum over batch to get (C, C)
        confusion_matrix = confusion_matrix.sum(dim=0)
        confusion_matrix = confusion_matrix.cpu().numpy().astype(np.float32)

        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                o[f"confusion_matrix_{i}_{j}"] = confusion_matrix[i, j]

        return o
