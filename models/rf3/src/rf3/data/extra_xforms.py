import logging

import numpy as np
import torch
from atomworks.ml.transforms._checks import (
    check_contains_keys,
)
from atomworks.ml.transforms.base import Transform

logger = logging.getLogger(__name__)


def patch_conformer_fallback_to_input_coords() -> None:
    """Monkey-patch sample_rdkit_conformer_for_atom_array to use input
    coordinates instead of zeros when conformer generation fails.

    This is applied once at pipeline-build time when
    ``fallback_conformer_to_input_coords=True``.
    """
    import atomworks.ml.transforms.af3_reference_molecule as _af3_ref
    import atomworks.ml.transforms.rdkit_utils as _rdkit_utils

    if getattr(
        _rdkit_utils.sample_rdkit_conformer_for_atom_array,
        "_input_coord_fallback_patched",
        False,
    ):
        return  # already patched

    _orig = _rdkit_utils.sample_rdkit_conformer_for_atom_array

    def _patched(atom_array, *args, **kwargs):
        original_coord = atom_array.coord.copy()
        result = _orig(atom_array, *args, **kwargs)
        # _orig may return (AtomArray, mol) when return_mol=True
        aa_result = result[0] if isinstance(result, tuple) else result
        mol = result[1] if isinstance(result, tuple) else None
        if np.all(aa_result.coord == 0) or np.any(np.isnan(aa_result.coord)):
            logger.warning(
                f"Conformer generation failed for {atom_array.res_name[0]}; "
                "using input coordinates as fallback."
            )
            aa_result.coord = original_coord
            # Also add the fallback coords as a conformer on the mol so that downstream
            # steps (e.g. GetRDKitChiralCenters) don't attempt to re-generate conformers.
            if mol is not None and mol.GetNumConformers() == 0:
                from rdkit.Chem import Conformer as _Conformer

                conf = _Conformer(mol.GetNumAtoms())
                for i in range(min(mol.GetNumAtoms(), len(original_coord))):
                    conf.SetAtomPosition(i, original_coord[i].tolist())
                mol.AddConformer(conf, assignId=True)
        return result

    _patched._input_coord_fallback_patched = True
    _rdkit_utils.sample_rdkit_conformer_for_atom_array = _patched
    # af3_reference_molecule imports the function directly, so patch that reference too
    _af3_ref.sample_rdkit_conformer_for_atom_array = _patched


class CheckForNaNsInInputs(Transform):
    """
    This component marks atoms as occ=0 based on bfactor values

    It takes as input 'brange', a list specifying the Mminimum and maximum B factors to
    keep.

    Example:
        brange = [-1.0,70.0] will mark with occ=0 any atom with b>70 or b<-1
    """

    def check_input(self, data: dict):
        check_contains_keys(data, ["coord_atom_lvl_to_be_noised"])
        check_contains_keys(data, ["noise"])

    def forward(self, data: dict) -> dict:
        # During inference, replace coordinates with true noise
        # TODO: Move elsewhere in pipeline; placing it here is a short-term hack
        if data.get("is_inference", False):
            data["coord_atom_lvl_to_be_noised"] = torch.randn_like(
                data["coord_atom_lvl_to_be_noised"]
            )

        assert not torch.isnan(
            data["coord_atom_lvl_to_be_noised"]
        ).any(), "NaN found in network input"
        assert not torch.isnan(data["noise"]).any(), "NaN found in network noise"

        return data
