import torch
import torch.nn as nn

from rf2aa.metrics.metrics_base import Metric
from rf2aa.kinematics import get_dih

def calc_chiral_loss_masked(pred, chirals, mask):
    """
    calculate error in dihedral angles for chiral atoms
    Input:
     - pred: predicted coords (B, L, :, 3)
     - chirals: True coords (B, nchiral, 5), skip if 0 chiral sites, 5 dimension are indices for 4 atoms that make dihedral and the ideal angle they should form
    Output:
     - mean squared error of chiral angles
    """
    if chirals.shape[1] == 0:
        return (
            torch.tensor(0.0, device=pred.device),
            torch.tensor(0, device=pred.device),
        )
    chiral_dih = pred[:, chirals[..., :-1].long()]
    pred_dih = get_dih(
        chiral_dih[..., 0, :],
        chiral_dih[..., 1, :],
        chiral_dih[..., 2, :],
        chiral_dih[..., 3, :],
    )
    mask = mask[chirals[..., :-1].long()].all(dim=-1)
    l = torch.square(mask*(pred_dih - chirals[..., -1])).sum(dim=-1)
    return l, mask.sum()


class ChiralLoss(Metric):
    def compute(self, network_input, network_output, extra_info):
        chiral_loss = {"chiral_loss_sum": [], "chiral_loss_mean": [], "nchiral_centers": []}

        chain_iid_token_lvl = extra_info["chain_iid_token_lvl"]
        tok_idx = network_input["f"]["atom_to_token_map"].cpu().numpy()

        pred = network_output["X_L"]
        chirals = network_input["f"]['chiral_feats']

        for chain_i, chain_type in extra_info["pn_units_to_score"]:
            # get tokens in chain_i and chain_j
            chain_i_tokens = chain_iid_token_lvl == chain_i

            # convert the token level to the atom level
            chain_i_atoms = chain_i_tokens[tok_idx]

            # compute lddt using the pairs_to_score from the intersection
            ch_loss, nch = calc_chiral_loss_masked(
                pred,
                chirals,
                mask=torch.tensor(chain_i_atoms,device=pred.device),
            )

            chiral_loss["chiral_loss_sum"].append(ch_loss[0].item())
            chiral_loss["chiral_loss_mean"].append((ch_loss[0]/(nch+1e-4)).item())
            chiral_loss["nchiral_centers"].append(nch.item())
        return chiral_loss

