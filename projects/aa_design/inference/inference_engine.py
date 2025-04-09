import logging
import os
import tempfile

from modelhub.inference_engines.af3 import AF3InferenceEngine
from modelhub.utils.ddp import RankedLogger
from projects.aa_design.inference.input_parsing import (
    create_spoofed_cif_file_for_backbone,
)

logging.basicConfig(level=logging.INFO)
ranked_logger = RankedLogger(__name__, rank_zero_only=True)

class Atom14InferenceEngine(AF3InferenceEngine):
    """Inference engine for Atom14"""
    def __init__(self, inputs=None, n_residues=200, n_backbones=1, **kwargs):
        inputs = []

        tmp_dir = tempfile.gettempdir()
        for i in range(n_backbones):
            tmp_cif_file = os.path.join(tmp_dir, f"unconditional_backbone_{i}.cif")
            
            create_spoofed_cif_file_for_backbone(n_residues, tmp_cif_file)
            
            inputs.append(tmp_cif_file)

        ranked_logger.info(f"Created temporary CIF files at {tmp_dir}")
        super().__init__(inputs=inputs, **kwargs)

