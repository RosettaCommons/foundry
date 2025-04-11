from modelhub.model.AF3_structure import Recycler
from projects.rfscore.model.template_embedder import RFScoreTemplateEmbedder


class RFScoreRecycler(Recycler):
    """Recycler that adds ligand conditioning to the template track"""

    def __init__(self, **kwargs):
        # ... pop the use_fourier_encoding from the template_embedder (since otherwise it will cause a KeyError within AF-3 TemplateEmbedder)
        use_fourier_encoding = kwargs["template_embedder"].pop("use_fourier_encoding")

        super().__init__(**kwargs)

        # ... override the template embedder to use the RFScore template embedder, which provides additional conditioning
        self.template_embedder = RFScoreTemplateEmbedder(
            c_z=kwargs["c_z"],
            **kwargs["template_embedder"],
            use_fourier_encoding=use_fourier_encoding,
        )
