import numpy as np
import torch
from atomworks.io.transforms.atom_array import ensure_atom_array_stack
from atomworks.ml.transforms.atom_array import AddGlobalTokenIdAnnotation
from atomworks.ml.transforms.atomize import AtomizeByCCDName
from atomworks.ml.transforms.base import Compose
from atomworks.ml.utils.token import get_token_starts
from beartype.typing import Any
from biotite.structure import AtomArray, AtomArrayStack, stack
from jaxtyping import Bool, Float, Int

from foundry.metrics.metric import Metric
from foundry.utils.ddp import RankedLogger

ranked_logger = RankedLogger(__name__, rank_zero_only=True)


def calc_lddt(
    X_L: Float[torch.Tensor, "D L 3"],
    X_gt_L: Float[torch.Tensor, "D L 3"],
    crd_mask_L: Bool[torch.Tensor, "D L"],
    tok_idx: Int[torch.Tensor, "L"],
    pairs_to_score: Bool[torch.Tensor, "L L"] | None = None,
    distance_cutoff: float = 15.0,
    eps: float = 1e-6,
) -> Float[torch.Tensor, "D"]:
    """Calculates LDDT scores for each model in the batch.

    Args:
        X_L: Predicted coordinates (D, L, 3).
        X_gt_L: Ground truth coordinates (D, L, 3).
        crd_mask_L: Coordinate mask indicating valid atoms (D, L).
        tok_idx: Token index of each atom (L,). Used to exclude same-token pairs.
        pairs_to_score: Boolean mask for pairs to score (L, L). If None, scores all valid pairs.
        distance_cutoff: Distance cutoff for scoring pairs.
        eps: Small epsilon to prevent division by zero.

    Returns:
        LDDT scores for each model (D,).
    """
    D, L = X_L.shape[:2]

    # Create pairs to score mask - if not provided, use upper triangular (includes diagonal)
    if pairs_to_score is None:
        pairs_to_score = torch.ones((L, L), dtype=torch.bool).triu(0).to(X_L.device)
    else:
        assert pairs_to_score.shape == (L, L)
        pairs_to_score = pairs_to_score.triu(0).to(X_L.device)

    # Get indices of atom pairs to evaluate
    first_index: Int[torch.Tensor, "n_pairs"]
    second_index: Int[torch.Tensor, "n_pairs"]
    first_index, second_index = torch.nonzero(pairs_to_score, as_tuple=True)

    # Compute LDDT score for each model in the batch
    lddt_scores = []
    for d in range(D):
        # Calculate pairwise distances in ground truth structure
        ground_truth_distances = torch.linalg.norm(
            X_gt_L[d, first_index] - X_gt_L[d, second_index], dim=-1
        )

        # Create mask for valid pairs to score:
        # 1. Ground truth distance > 0 (atoms not at same position)
        # 2. Ground truth distance < cutoff (within interaction range)
        pair_mask = torch.logical_and(
            ground_truth_distances > 0, ground_truth_distances < distance_cutoff
        )

        # Only score pairs that are resolved in the ground truth
        pair_mask *= crd_mask_L[d, first_index] * crd_mask_L[d, second_index]

        # Don't score pairs that are in the same token (e.g., same residue)
        pair_mask *= tok_idx[first_index] != tok_idx[second_index]

        # Filter to only "valid" pairs
        valid_pairs = pair_mask.nonzero(as_tuple=True)

        pair_mask_valid = pair_mask[valid_pairs].to(X_L.dtype)
        ground_truth_distances_valid = ground_truth_distances[valid_pairs]

        first_index_valid: Int[torch.Tensor, "n_valid_pairs"] = first_index[valid_pairs]
        second_index_valid: Int[torch.Tensor, "n_valid_pairs"] = second_index[
            valid_pairs
        ]

        # Calculate pairwise distances in predicted structure
        predicted_distances = torch.linalg.norm(
            X_L[d, first_index_valid] - X_L[d, second_index_valid], dim=-1
        )

        # Compute absolute distance differences (with small eps to avoid numerical issues)
        delta_distances = torch.abs(
            predicted_distances - ground_truth_distances_valid + eps
        )
        del predicted_distances, ground_truth_distances_valid

        # Calculate LDDT score using standard thresholds (0.5Å, 1.0Å, 2.0Å, 4.0Å)
        # LDDT is the average fraction of distances preserved within each threshold
        lddt_score = (
            0.25
            * (
                torch.sum((delta_distances < 0.5) * pair_mask_valid)  # 0.5Å threshold
                + torch.sum((delta_distances < 1.0) * pair_mask_valid)  # 1.0Å threshold
                + torch.sum((delta_distances < 2.0) * pair_mask_valid)  # 2.0Å threshold
                + torch.sum((delta_distances < 4.0) * pair_mask_valid)  # 4.0Å threshold
            )
            / (torch.sum(pair_mask_valid) + eps)  # Normalize by number of valid pairs
        )

        lddt_scores.append(lddt_score)

    return torch.tensor(lddt_scores, device=X_L.device)


def extract_lddt_features_from_atom_arrays(
    predicted_atom_array_stack: AtomArrayStack | AtomArray,
    ground_truth_atom_array_stack: AtomArrayStack | AtomArray,
) -> dict[str, Any]:
    """Extract all features needed for LDDT computation from AtomArrays.

    Args:
        predicted_atom_array_stack: Predicted coordinates as AtomArray(Stack)
        ground_truth_atom_array_stack: Ground truth coordinates as AtomArray(Stack)

    Returns:
        Dictionary containing:
        - X_L: Predicted coordinates tensor (D, L, 3)
        - X_gt_L: Ground truth coordinates tensor (D, L, 3)
        - crd_mask_L: Coordinate validity mask (D, L)
        - tok_idx: Token indices for each atom (L,)
        - chain_iid_token_lvl: Chain identification at token level
    """
    predicted_atom_array_stack = ensure_atom_array_stack(predicted_atom_array_stack)
    ground_truth_atom_array_stack = ensure_atom_array_stack(
        ground_truth_atom_array_stack
    )

    if (
        ground_truth_atom_array_stack.stack_depth() == 1
        and predicted_atom_array_stack.stack_depth() > 1
    ):
        # If the ground truth is a single model, and the predicted is a stack, we need to expand the ground truth to the same length as the predicted
        ground_truth_atom_array_stack = stack(
            [ground_truth_atom_array_stack[0]]
            * predicted_atom_array_stack.stack_depth()
        )

    # Compute coordinates - convert AtomArrays to tensors
    X_L: Float[torch.Tensor, "D L 3"] = torch.from_numpy(
        predicted_atom_array_stack.coord
    ).float()
    X_gt_L: Float[torch.Tensor, "D L 3"] = torch.from_numpy(
        ground_truth_atom_array_stack.coord
    ).float()

    # For the remaining feature generation, we can directly use the first model in the stack (only coordinates are different)
    ground_truth_atom_array = ground_truth_atom_array_stack[0]

    # Create coordinate mask using occupancy if available, fallback to coordinate validity
    if "occupancy" in ground_truth_atom_array.get_annotation_categories():
        # Use occupancy annotation (broadcast to all models in stack)if present (occupancy > 0 means atom is present)
        occupancy_mask = ground_truth_atom_array.occupancy > 0
        crd_mask_L: Bool[torch.Tensor, "D L"] = (
            torch.from_numpy(occupancy_mask)
            .bool()
            .unsqueeze(0)
            .expand(X_gt_L.shape[0], -1)
        )
    else:
        # Fallback to coordinate validity (not NaN)
        crd_mask_L: Bool[torch.Tensor, "D L"] = ~torch.isnan(X_gt_L).any(dim=-1)

    # Get token indices using the same logic as ComputeAtomToTokenMap
    if "token_id" in ground_truth_atom_array.get_annotation_categories():
        # Use the existing token_id annotation (matches ComputeAtomToTokenMap exactly)
        tok_idx = ground_truth_atom_array.token_id.astype(np.int32)
    else:
        # Generate annotations with Transform pipeline
        pipe = Compose(
            [AtomizeByCCDName(atomize_by_default=True), AddGlobalTokenIdAnnotation()]
        )
        data = pipe({"atom_array": ground_truth_atom_array})
        tok_idx = data["atom_array"].token_id.astype(np.int32)

    # Compute chain identification at the token-level
    token_starts = get_token_starts(ground_truth_atom_array)

    if "chain_iid" in ground_truth_atom_array.get_annotation_categories():
        chain_iid_token_lvl = ground_truth_atom_array.chain_iid[token_starts]
    else:
        # Use the chain_id annotation instead (e.g., for AF-3 outputs, where the chain_id is ostensibly the chain_iid)
        chain_iid_token_lvl = ground_truth_atom_array.chain_id[token_starts]

    return {
        "X_L": X_L,
        "X_gt_L": X_gt_L,
        "crd_mask_L": crd_mask_L,
        "tok_idx": tok_idx,
        "chain_iid_token_lvl": chain_iid_token_lvl,
    }


class AllAtomLDDT(Metric):
    """Computes all-atom LDDT scores from AtomArrays."""

    def __init__(self, log_lddt_for_every_batch: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.log_lddt_for_every_batch = log_lddt_for_every_batch

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {
            "predicted_atom_array_stack": "predicted_atom_array_stack",
            "ground_truth_atom_array_stack": "ground_truth_atom_array_stack",
        }

    def compute(
        self,
        predicted_atom_array_stack: AtomArrayStack | AtomArray,
        ground_truth_atom_array_stack: AtomArrayStack | AtomArray,
    ) -> dict[str, Any]:
        """Calculates all-atom LDDT between all pairs of atoms.

        Args:
            predicted_atom_array_stack: Predicted coordinates as AtomArray(Stack)
            ground_truth_atom_array_stack: Ground truth coordinates as AtomArray(Stack)

        Returns:
            A dictionary with all-atom LDDT scores:
            - lddt_scores: Raw LDDT scores for each model (torch.Tensor)
            - best_of_1_lddt: LDDT score for the first model
            - best_of_{N}_lddt: Best LDDT score across all N models
        """
        lddt_features = extract_lddt_features_from_atom_arrays(
            predicted_atom_array_stack, ground_truth_atom_array_stack
        )
        tok_idx = torch.tensor(lddt_features["tok_idx"]).to(lddt_features["X_L"].device)

        all_atom_lddt = calc_lddt(
            X_L=lddt_features["X_L"],
            X_gt_L=lddt_features["X_gt_L"],
            crd_mask_L=lddt_features["crd_mask_L"],
            tok_idx=tok_idx,
            pairs_to_score=None,  # By default, score all pairs, except those within the same token
            distance_cutoff=15.0,
        )

        result = {
            "best_of_1_lddt": all_atom_lddt[0].item(),
            f"best_of_{len(all_atom_lddt)}_lddt": all_atom_lddt.max().item(),
        }

        if self.log_lddt_for_every_batch:
            lddt_by_batch = {
                f"all_atom_lddt_{i}": all_atom_lddt[i].item()
                for i in range(len(all_atom_lddt))
            }
            result.update(lddt_by_batch)

        return result


class InterfaceLDDTByType(Metric):
    """Computes interface LDDT, grouped by interface type"""

    def __init__(self, log_lddt_for_every_batch: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.log_lddt_for_every_batch = log_lddt_for_every_batch

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {
            "predicted_atom_array_stack": "predicted_atom_array_stack",
            "ground_truth_atom_array_stack": "ground_truth_atom_array_stack",
            "interfaces_to_score": ("extra_info", "interfaces_to_score"),
        }

    def compute(
        self,
        predicted_atom_array_stack: AtomArrayStack | AtomArray,
        ground_truth_atom_array_stack: AtomArrayStack | AtomArray,
        interfaces_to_score: list = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Calculates interface LDDT between specific pairs of chains/units, grouped by interface type.

        Args:
            predicted_atom_array_stack: Predicted coordinates as AtomArray(Stack)
            ground_truth_atom_array_stack: Ground truth coordinates as AtomArray(Stack)
            interfaces_to_score: List of interface specifications, each as
                (pn_unit_i, pn_unit_j, interface_type)

        Returns:
            List of dictionaries containing interface LDDT results for each interface.
        """
        lddt_features = extract_lddt_features_from_atom_arrays(
            predicted_atom_array_stack, ground_truth_atom_array_stack
        )

        # Short-circuit if no interfaces to score
        if not interfaces_to_score:
            return []

        interface_results = []

        # Parse string inputs (for backwards compatibility)
        if isinstance(interfaces_to_score, str):
            interfaces_to_score = (
                eval(interfaces_to_score) if interfaces_to_score else []
            )

        # Loop over the interfaces to score
        for pn_unit_i, pn_unit_j, interface_type in interfaces_to_score:
            # Get tokens in pn_unit_i and pn_unit_j
            pn_unit_i_tokens = lddt_features["chain_iid_token_lvl"] == pn_unit_i
            pn_unit_j_tokens = lddt_features["chain_iid_token_lvl"] == pn_unit_j

            if pn_unit_i_tokens.sum() == 0 or pn_unit_j_tokens.sum() == 0:
                ranked_logger.warning(
                    f"No atoms found for {pn_unit_i} or {pn_unit_j}! Available chains: {np.unique(lddt_features['chain_iid_token_lvl']).tolist()}"
                )
                continue

            # Convert the token level to the atom level
            pn_unit_i_atoms = pn_unit_i_tokens[lddt_features["tok_idx"]]
            pn_unit_j_atoms = pn_unit_j_tokens[lddt_features["tok_idx"]]

            # Compute the outer product of chain_i and chain_j, which represents the interface
            chain_ij_atoms = torch.einsum(
                "L, K -> LK",
                torch.tensor(pn_unit_i_atoms),
                torch.tensor(pn_unit_j_atoms),
            ).to(lddt_features["X_L"].device)

            # Symmetrize the interface so we can later multiply with an upper triangular without losing information
            chain_ij_atoms = chain_ij_atoms | chain_ij_atoms.T

            # compute lddt using the pairs_to_score from the intersection
            lddt = calc_lddt(
                lddt_features["X_L"],
                lddt_features["X_gt_L"],
                lddt_features["crd_mask_L"],
                torch.tensor(lddt_features["tok_idx"]).to(lddt_features["X_L"].device),
                pairs_to_score=chain_ij_atoms,
                distance_cutoff=30.0,
            )

            # add the results to the interface_results list
            n = len(lddt)
            result = {
                "pn_units": [pn_unit_i, pn_unit_j],
                "type": interface_type,
                "best_of_1_lddt": lddt[0].item(),
                f"best_of_{n}_lddt": lddt.max().item(),
            }

            if self.log_lddt_for_every_batch:
                lddt_by_batch = {f"lddt_{i}": lddt[i].item() for i in range(len(lddt))}
                result.update(lddt_by_batch)

            interface_results.append(result)

        return interface_results


class ChainLDDTByType(Metric):
    """Computes chain-wise LDDT scores from AtomArrays, grouped by chain type."""

    def __init__(self, log_lddt_for_every_batch: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.log_lddt_for_every_batch = log_lddt_for_every_batch

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {
            "predicted_atom_array_stack": "predicted_atom_array_stack",
            "ground_truth_atom_array_stack": "ground_truth_atom_array_stack",
            "pn_units_to_score": ("extra_info", "pn_units_to_score"),
        }

    def compute(
        self,
        predicted_atom_array_stack: AtomArrayStack | AtomArray,
        ground_truth_atom_array_stack: AtomArrayStack | AtomArray,
        pn_units_to_score: list = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Calculates intra-chain LDDT for specific chains/units.

        Args:
            predicted_atom_array_stack: Predicted coordinates as AtomArray(Stack)
            ground_truth_atom_array_stack: Ground truth coordinates as AtomArray(Stack)
            pn_units_to_score: List of chain specifications, each as (pn_unit_iid, chain_type)

        Returns:
            List of dictionaries containing chain LDDT results for each chain.
        """
        lddt_features = extract_lddt_features_from_atom_arrays(
            predicted_atom_array_stack, ground_truth_atom_array_stack
        )

        # Short-circuit if no chains to score
        if not pn_units_to_score:
            return []

        chain_results = []

        # Parse string inputs (for backwards compatibility)
        if isinstance(pn_units_to_score, str):
            pn_units_to_score = eval(pn_units_to_score) if pn_units_to_score else []

        # For all chains (pn_units) to score...
        for chain, chain_type in pn_units_to_score:
            # ... get tokens in chain instance
            chain_tokens = lddt_features["chain_iid_token_lvl"] == chain
            if chain_tokens.sum() == 0:
                ranked_logger.warning(
                    f"No atoms found for {chain}! Available chains: {np.unique(lddt_features['chain_iid_token_lvl']).tolist()}"
                )
                continue

            # ... convert the token level to the atom level
            chain_atoms = chain_tokens[lddt_features["tok_idx"]]

            # ... compute the outer product of the chain with itself (the definition of intra-lddt)
            chain_ij_atoms = torch.einsum(
                "L, K -> LK", torch.tensor(chain_atoms), torch.tensor(chain_atoms)
            ).to(lddt_features["X_L"].device)

            # ... compute lddt using the pairs_to_score from the interface
            lddt = calc_lddt(
                lddt_features["X_L"],
                lddt_features["X_gt_L"],
                lddt_features["crd_mask_L"],
                torch.tensor(lddt_features["tok_idx"]).to(lddt_features["X_L"].device),
                pairs_to_score=chain_ij_atoms,
            )

            # ... and finally add the results to the chain_results list
            n = len(lddt)
            result = {
                "pn_units": [chain],
                "type": chain_type,
                "best_of_1_lddt": lddt[0].item(),
                f"best_of_{n}_lddt": lddt.max().item(),
            }

            if self.log_lddt_for_every_batch:
                lddt_by_batch = {f"lddt_{i}": lddt[i].item() for i in range(len(lddt))}
                result.update(lddt_by_batch)

            chain_results.append(result)

        return chain_results


class ByTypeLDDT(Metric):
    """Calculates LDDT scores by type for both chains and interfaces."""

    def __init__(self, log_lddt_for_every_batch: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.interface_lddt = InterfaceLDDTByType(
            log_lddt_for_every_batch=log_lddt_for_every_batch, **kwargs
        )
        self.chain_lddt = ChainLDDTByType(
            log_lddt_for_every_batch=log_lddt_for_every_batch, **kwargs
        )

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {
            "predicted_atom_array_stack": "predicted_atom_array_stack",
            "ground_truth_atom_array_stack": "ground_truth_atom_array_stack",
            "interfaces_to_score": ("extra_info", "interfaces_to_score"),
            "pn_units_to_score": ("extra_info", "pn_units_to_score"),
        }

    @property
    def optional_kwargs(self) -> set[str]:
        """Mark interfaces_to_score and pn_units_to_score as optional."""
        return {"interfaces_to_score", "pn_units_to_score"}

    def compute(
        self,
        predicted_atom_array_stack: AtomArrayStack | AtomArray,
        ground_truth_atom_array_stack: AtomArrayStack | AtomArray,
        interfaces_to_score: list[tuple[str, str, str]] | None = None,
        pn_units_to_score: list[tuple[str, str]] | None = None,
    ) -> list[dict[str, Any]]:
        """Calculates LDDT scores by type for both chains and interfaces.

        Args:
            predicted_atom_array_stack: Predicted coordinates as AtomArray(Stack)
            ground_truth_atom_array_stack: Ground truth coordinates as AtomArray(Stack)
            interfaces_to_score: Tuples of (pn_unit_i, pn_unit_j, interface_type)
                representing the interfaces to score
            pn_units_to_score: Tuples of (pn_unit_iid, chain_type)
                representing the chains to score
            log_lddt_for_every_batch: Whether to compute LDDT for each model separately (vs. only BO1 and BO{N})

        Returns:
            Combined list of interface and chain LDDT results.
        """

        # Compute interface LDDT scores
        interface_results = self.interface_lddt.compute(
            predicted_atom_array_stack=predicted_atom_array_stack,
            ground_truth_atom_array_stack=ground_truth_atom_array_stack,
            interfaces_to_score=interfaces_to_score,
        )

        # Compute chain LDDT scores
        chain_results = self.chain_lddt.compute(
            predicted_atom_array_stack=predicted_atom_array_stack,
            ground_truth_atom_array_stack=ground_truth_atom_array_stack,
            pn_units_to_score=pn_units_to_score,
        )

        # Merge the results
        combined_results = interface_results + chain_results

        return combined_results
