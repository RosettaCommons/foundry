"""Score single substitutions inside the user's upstream LigandMPNN environment.

Usage: python score_worker.py spec.json scores.json
Imports are deliberately local: the lightweight RFO driver does not need torch,
ProDy, or upstream MPNN on its own Python path.
"""

import json
import sys
from pathlib import Path


def score(spec):
    import numpy as np
    import torch

    sys.path.insert(0, spec["mpnn_repo"])
    from data_utils import alphabet, featurize, parse_PDB
    from model_utils import ProteinMPNN

    torch.manual_seed(spec["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        spec["checkpoint_path"], map_location=device, weights_only=True
    )
    model_type = spec["model_type"]
    atom_context_num = (
        checkpoint["atom_context_num"] if model_type == "ligand_mpnn" else 1
    )
    model = ProteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=checkpoint["num_edges"],
        device=device,
        atom_context_num=atom_context_num,
        model_type=model_type,
        ligand_mpnn_use_side_chain_context=0,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    protein, _, _, icodes, _ = parse_PDB(spec["structure_path"], device=device)
    residue_ids = [
        f"{chain}{int(number)}{icode}"
        for chain, number, icode in zip(
            protein["chain_letters"], protein["R_idx"].cpu().tolist(), icodes
        )
    ]
    if len(set(residue_ids)) != len(residue_ids):
        raise ValueError("Upstream MPNN parsed duplicate residue IDs.")
    protein["chain_mask"] = torch.zeros_like(protein["mask"])
    features = featurize(
        protein,
        cutoff_for_score=8.0,
        use_atom_context=True,
        number_of_ligand_atoms=atom_context_num,
        model_type=model_type,
    )
    features["batch_size"] = spec["num_seqs"]
    features["symmetry_residues"] = [[]]
    length = features["S"].shape[1]
    # Put the scored residue last, so its likelihood is conditioned on every
    # other native residue. No sequence is sampled or fed back between scores.
    random_order = torch.rand((spec["num_seqs"], length), device=device) + 1.0
    scores = []
    with torch.no_grad():
        for residue_id in spec["designed_residues"]:
            index = residue_ids.index(residue_id)
            if not features["mask"][0, index]:
                raise ValueError(f"MPNN cannot score incomplete residue {residue_id}.")
            features["chain_mask"] = torch.zeros_like(features["mask"])
            features["chain_mask"][0, index] = 1
            features["randn"] = random_order
            result = model.score(features, use_sequence=True)
            if not torch.all(result["decoding_order"][:, -1] == index):
                raise ValueError("MPNN did not decode the scored residue last.")
            probabilities = result["log_probs"][:, index].exp().mean(0).cpu().numpy()
            if not np.isfinite(probabilities).all():
                raise ValueError(f"Non-finite MPNN probabilities for {residue_id}.")
            scores.append(
                {
                    "residue_id": residue_id,
                    "native_aa": alphabet[int(features["S"][0, index])],
                    "probabilities": dict(zip(alphabet, map(float, probabilities))),
                }
            )
    return {"scores": scores}


if __name__ == "__main__":
    spec = json.loads(Path(sys.argv[1]).read_text())
    Path(sys.argv[2]).write_text(json.dumps(score(spec), indent=2) + "\n")
