import logging

import pytest

from projects.rfscore.pipelines.rfscore import build_rfscore_transform_pipeline
from datahub.utils.testing import cached_parse
from projects.rfscore.tests.conftest import PROTEIN_MSA_DIRS, RNA_MSA_DIRS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


PRIOR_PIPELINE_BUGS_AF3 = ["7qbs", "5epq", "2g37", "4v4s"]

@pytest.mark.parametrize("pdb_id", PRIOR_PIPELINE_BUGS_AF3)
@pytest.mark.slow
def test_prior_pipeline_bugs_af3(pdb_id: str):
    """Run a single example through the pipeline. Useful for debugging specific examples."""
    input = cached_parse(pdb_id)
    input["example_id"] = pdb_id
    pipe = build_rfscore_transform_pipeline(
        protein_msa_dirs=PROTEIN_MSA_DIRS, 
        rna_msa_dirs=RNA_MSA_DIRS, 
        is_inference=False,
        use_element_for_atom_names_of_atomized_tokens=True,
        p_use_ground_truth_conformer=0.5,
    )
    output = pipe(input)

    assert output is not None


if __name__ == "__main__":
    pytest.main(["-v", __file__, "-m not very_slow"])
