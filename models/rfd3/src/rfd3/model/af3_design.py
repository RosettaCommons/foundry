from rfd3.model.encoders import SimpleRecycler

from modelhub.model.AF3 import AF3


class AF3DesignTrunk(AF3):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.recycler = SimpleRecycler(
            c_s=kwargs["c_s"], c_z=kwargs["c_z"], **kwargs["recycler"]
        )
        self.distogram_head = None
