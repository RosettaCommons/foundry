from modelhub.model.AF3 import AF3, AF3WithConfidence
from projects.rfscore.model.recycler import RFScoreRecycler


class RFScore(AF3):
    """RFScore network"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # ... override the recycler with the RFScore recycler
        self.recycler = RFScoreRecycler(
            c_s=kwargs["c_s"], c_z=kwargs["c_z"], **kwargs["recycler"]
        )

class RFScoreWithConfidence(AF3WithConfidence):
    """RFScore network"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # ... override the recycler with the RFScore recycler
        self.recycler = RFScoreRecycler(
            c_s=kwargs["c_s"], c_z=kwargs["c_z"], **kwargs["recycler"]
        )
