from rf2aa.metrics.metrics_base import Metric


def calc_lddt(X_L, X_gt_L, crd_mask_L, tok_idx, pairs_to_score=None):
    """
    X_L: predicted coordinates (D, L, 3)
    X_gt_L: ground truth coordinates (D, L, 3)
    crd_mask_L: mask of coordinates (D, L,)
    tok_idx: token index of each atom (L,) 
    pairs_to_score: pairs to score (L, L) | None
    """
    D, L = X_L.shape[:2]
    
    if pairs_to_score is None:
        pairs_to_score = torch.ones((L, L), dtype=torch.bool)
    else:
        assert pairs_to_score.shape == (L, L)
    
    # Compute distance matrix
    predicted_distances = torch.cdist(X_L, X_L)
    ground_truth_distances = torch.cdist(X_gt_L, X_gt_L)
    ground_truth_distances[ground_truth_distances.isnan()] = 9999.0
    difference_distances = torch.abs(ground_truth_distances - predicted_distances)

    lddt_matrix = torch.zeros_like(difference_distances)
    lddt_matrix = 0.25 * (difference_distances < 4.0) + \
                  0.25 * (difference_distances < 2.0) + \
                  0.25 * (difference_distances < 1.0) + \
                  0.25 * (difference_distances < 0.5)

    is_close_distance_LL = (ground_truth_distances < 15.0)
    in_same_residue_LL = tok_idx[None, :] == tok_idx[:, None]

    to_score_LL = pairs_to_score & is_close_distance & in_same_residue

    lddt = lddt_matrix * to_score[None] / (to_score.sum() + 1e-6) 
    return lddt



class InterfaceLDDT(Metric):

    def __call__(self, 
                network_input, 
                network_output, 
                loss_input
        ):
        # for each interface to score
        # produce a mask for that interface
        # calculate the lddt
        pass