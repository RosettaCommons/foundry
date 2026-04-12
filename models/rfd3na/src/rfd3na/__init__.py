"""RFD3 - RosettaFold-diffusion model implementation."""

import pydantic
from packaging.version import Version

if Version(pydantic.__version__) < Version("2.0"):
    raise RuntimeError(
        f"Pydantic >=2.0 is required; found {pydantic.__version__}. "
        "Pin pydantic>=2,<3 and upgrade dependent packages."
    )

__version__ = "0.1.0"
