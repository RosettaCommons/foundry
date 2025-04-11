from modelhub.model.AF3 import AF3
from projects.rfscore.model.recycler import RFScoreRecycler


class RFScore(AF3):
    """RFScore network"""

    def __init__(self, **kwargs):
        # ... pop the use_fourier_encoding from the template_embedder (since otherwise it will cause a KeyError within AF-3 TemplateEmbedder)
        use_fourier_encoding = kwargs["recycler"]["template_embedder"].pop(
            "use_fourier_encoding"
        )

        super().__init__(**kwargs)

        # ... add back the use_fourier_encoding to the kwargs (since it's needed for RFScoreTemplateEmbedder)
        kwargs["recycler"]["template_embedder"]["use_fourier_encoding"] = (
            use_fourier_encoding
        )

        # ... override the recycler with the RFScore recycler
        self.recycler = RFScoreRecycler(
            c_s=kwargs["c_s"], c_z=kwargs["c_z"], **kwargs["recycler"]
        )
