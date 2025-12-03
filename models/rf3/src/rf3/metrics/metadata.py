import json

from beartype.typing import Any, Literal

from foundry.metrics.metric import Metric


class ExtraInfo(Metric):
    """Stores the extra_info from the dataloader output in the metrics dictionary.
    Only basic Python types that are hashable and can be JSON serialized are stored."""

    def __init__(self, keys_to_store: list[str] | Literal["all"] = "all", **kwargs):
        super().__init__(**kwargs)
        self.keys_to_store = keys_to_store

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {"extra_info": "extra_info"}

    def _is_basic_hashable_type(self, value: Any) -> bool:
        """Check if value is a basic Python type that is both JSON serializable and hashable."""
        try:
            # First check if it's hashable
            hash(value)

            # Then check if it's JSON serializable
            json.dumps(value)
            return True
        except (TypeError, OverflowError):
            return False

    def compute(
        self,
        extra_info: dict,
    ) -> dict[str, Any]:
        result = {}
        for key, value in extra_info.items():
            # Check if we should include this key
            if self.keys_to_store == "all" or key in self.keys_to_store:
                # Check if the value is a basic hashable type
                if self._is_basic_hashable_type(value):
                    result[key] = value
        return result
