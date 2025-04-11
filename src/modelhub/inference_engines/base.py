from abc import ABC, abstractmethod
from pathlib import Path


class InferenceEngine(ABC):
    """Abstract base class for inference pipelines."""

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def eval(self, inputs: list[Path]) -> None:
        """Run inference on input files."""
        pass
