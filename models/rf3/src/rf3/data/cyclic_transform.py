import logging

import numpy as np
from atomworks.ml.transforms.base import Transform

logger = logging.getLogger("atomworks.ml")


class AddCyclicBonds(Transform):
    """
    Transform that detects and adds cyclic (head-to-tail) peptide bonds in protein chains.
    This transform analyzes the atom-level structure of each chain in the input data to identify
    cyclic bonds between the N-terminal nitrogen of the first residue and the C-terminal carbon
    of the last residue, based on spatial proximity. If such a bond is detected (0.5 Å < distance < 1.5 Å),
    it updates the token-level bond features to reflect the presence of the cyclic bond and flags that
    chain as being cyclic.

    Requirements:
        - Must be applied after "AddAF3TokenBondFeatures" and "EncodeAF3TokenLevelFeatures" transforms.
    """

    # need to do it after this transform, because we want to include poly-poly bonds, which AF3TokenBondFeatures does not.
    requires_previous_transforms = [
        "AddAF3TokenBondFeatures",
        "EncodeAF3TokenLevelFeatures",
        "AddGlobalTokenIdAnnotation",
    ]

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        token_bonds = data["feats"]["token_bonds"]
        asym_ids = data["feats"]["asym_id"]

        cyclic_token_bonds = np.zeros_like(token_bonds, dtype=bool)
        cyclic_asym_ids = set()

        # check for any cyclic bonds
        for chain in np.unique(atom_array.chain_id):
            chain_mask = atom_array.chain_id == chain
            if not np.any(chain_mask):
                continue
            chain_array = atom_array[chain_mask]
            residue_ids = np.unique(chain_array.res_id)
            if len(residue_ids) < 2:
                continue
            first_residue = residue_ids[0]
            last_residue = residue_ids[-1]
            first_nitrogen_mask = (chain_array.res_id == first_residue) & (
                chain_array.atom_name == "N"
            )
            last_carbon_mask = (chain_array.res_id == last_residue) & (
                chain_array.atom_name == "C"
            )
            if first_nitrogen_mask.sum() == 1 and last_carbon_mask.sum() == 1:
                first_nitrogen = chain_array[first_nitrogen_mask]
                last_carbon = chain_array[last_carbon_mask]
                distance = np.linalg.norm(
                    first_nitrogen.coord[0] - last_carbon.coord[0]
                )
                if distance < 1.5 and distance > 0.5:  # peptide-bond length-ish
                    cyclic_token_bonds[
                        first_nitrogen.token_id[0], last_carbon.token_id[0]
                    ] = True
                    cyclic_token_bonds[
                        last_carbon.token_id[0], first_nitrogen.token_id[0]
                    ] = True
                    logger.warning(
                        f"Detected cyclic bond in chain {chain} of {data['example_id']} between residues {first_residue} and {last_residue} with distance {distance:.2f} Å"
                    )

        cyclic_asym_ids.update(
            asym_ids[np.where(np.any(cyclic_token_bonds, axis=0))[0]].tolist()
        )
        token_bonds |= cyclic_token_bonds
        data["feats"]["token_bonds"] = token_bonds
        data["feats"]["cyclic_asym_ids"] = list(cyclic_asym_ids)

        return data
