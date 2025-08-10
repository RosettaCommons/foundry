from modelhub.model.RF3_structure import Recycler
from projects.rfscore.model.template_embedder import RF3TemplateEmbedder


class RFScoreRecycler(Recycler):
    """Recycler that adds ligand conditioning to the template track"""

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        # ... override the template embedder to use the RFScore template embedder, which provides additional conditioning
        self.template_embedder = RF3TemplateEmbedder(
            c_z=kwargs["c_z"],
            **kwargs["template_embedder"],
        )
