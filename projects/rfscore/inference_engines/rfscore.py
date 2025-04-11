import logging

import numpy as np
from biotite.structure import AtomArray
from cifutils.utils.selection import get_mask_from_selection_string
from datahub.enums import GroundTruthConformerPolicy

from modelhub.inference_engines.af3 import AF3InferenceEngine
from modelhub.utils.ddp import RankedLogger

logging.basicConfig(level=logging.INFO)
ranked_logger = RankedLogger(__name__, rank_zero_only=True)


class RFScoreInferenceEngine(AF3InferenceEngine):
    """Inference engine for RFScore"""

    def __init__(self, **kwargs):
        template_selection_strings = kwargs.pop("template_selection_strings", None)

        super().__init__(**kwargs)

        # ... RFScore-specific dataset overrides
        rfscore_dataset_overrides = {
            "inference_template_noise_scales": {
                "atomized": 0.0,
                "not_atomized": 0.0,
            },
            "allowed_chain_types_for_conditioning": None,
        }

        self.dataset_overrides.update(rfscore_dataset_overrides)

        # ... identify components of the structure to template, using our selection string API (CHAIN_ID/RES_NAME/RES_ID/ATOM_NAME)
        self.template_selection_strings = template_selection_strings

    def prepare_atom_array(self, atom_array: AtomArray) -> AtomArray:
        atom_array = super().prepare_atom_array(atom_array)

        # ... add the annotation if it does not already exist, defaulting to all False
        if (
            "ground_truth_conformer_policy"
            not in atom_array.get_annotation_categories()
        ):
            atom_array.set_annotation(
                "ground_truth_conformer_policy",
                np.full(
                    len(atom_array), GroundTruthConformerPolicy.IGNORE, dtype=np.int8
                ),
            )

        # If we specified a selection string, we use it to extract the desired components
        if self.template_selection_strings:
            if isinstance(self.template_selection_strings, str):
                self.template_selection_strings = [self.template_selection_strings]

            # ... and set the ground truth conformer policy to REPLACE for the selected components
            for selection_string in self.template_selection_strings:
                mask = get_mask_from_selection_string(atom_array, selection_string)
                atom_array.ground_truth_conformer_policy[mask] = (
                    GroundTruthConformerPolicy.REPLACE
                )

        return atom_array
