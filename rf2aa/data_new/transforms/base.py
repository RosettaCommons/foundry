"""Base classes for transformations."""

from __future__ import annotations

import contextlib
import logging
import os
import pickle
import pprint
from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Callable, Iterable

from rf2aa.data_new.transforms._checks import check_contains_keys, check_does_not_contain_keys
from rf2aa.data_new.transforms._rng import _collect_rng_states, rng_state, serialize_rng_state_dict

__all__ = ["Transform", "Compose", "RemoveKeys", "SubsetToKeys", "AddData", "LogData"]

logger = logging.getLogger(__name__)
DEBUG = os.getenv("DEBUG", True)
if DEBUG:
    logger.setLevel(logging.DEBUG)
    logger.debug("Debug mode is on")
    import traceback
else:
    logger.setLevel(logging.INFO)


class TransformPipelineError(Exception):
    """A custom error class for Transform pipelines (via `Compose`)."""

    def __init__(self, message: str, rng_state_dict: dict[str, Any] | None = None):
        super().__init__(message)
        # expose RNG state dict for debugging
        self.rng_state_dict = rng_state_dict


class TransformedDict(dict):
    """A thin wrapper around a dictionary that can be used to track the transform history."""

    def __new__(cls, __existing_dict_to_wrap: dict[str, Any] | None = None, **kwargs):
        """Create a new instance or return the existing TransformedDict instance.

        NOTE: To get a pure dictionary, simply use `dict(transformed_dict)` on a TransformedDict instance.
        TransformedDict's behave just like dicts for all intents and purposes, so you can use them just like
        a regular dictionary.

        Args:
            __existing_dict_to_wrap (dict, optional): This is useful for wrapping an existing dictionary.
                The odd name `__existing_dict_to_wrap` is used as an unlikely name to avoid conflicts
                with the `dict` class.
            **kwargs: Additional keyword arguments to pass to the dictionary constructor. This ensures
                that a TransformedDict can be initialized just like a regular dictionary if no existing
                dictionary to wrap is provided.
        """
        if isinstance(__existing_dict_to_wrap, TransformedDict):
            return __existing_dict_to_wrap
        instance = super().__new__(cls)
        if __existing_dict_to_wrap is not None:
            instance.update(__existing_dict_to_wrap)
        instance.__transform_history__ = []
        return instance


class Transform(ABC):
    """
    Abstract base class for transformations on dictionary objects.
    """

    validate_input: bool = True
    raise_if_invalid_input: bool = True
    requires_previous_transforms: list[str] = []
    previous_transforms_order_matters: bool = False
    _track_transform_history: bool = True

    # To be implemented by subclasses
    @abstractmethod
    def check_input(self, data: dict[str, Any]) -> None:
        """
        Check if the input dictionary is valid for the transform. Raises an error if the input is invalid.
        """
        pass

    @abstractmethod
    def forward(self, data: dict[str, Any], *args, **kwargs) -> dict[str, Any]:
        """
        Apply a transformation to the input dictionary and return the transformed dictionary.

        Parameters:
            data (dict): The input dictionary to transform.

        Returns:
            dict: The transformed dictionary.
        """
        pass

    # Internal logic for formatting error messages, debugging, logging and transform history tracking
    def _format_error_msg(self, e: Exception) -> str:
        """
        Formats the error message with optional traceback when in DEBUG mode.
        """
        msg = f"Invalid input for {self.__class__.__name__}: {e}"
        if DEBUG:
            msg += f"\n\n{traceback.format_exc()}\n" + "=" * 80
        return msg

    def _get_transform_history(self, data: dict[str, Any]) -> list[str]:
        """
        Get the transform history from the data.
        """
        return TransformedDict(data).__transform_history__

    def _update_transform_history(self, data: dict[str, Any], transform_history: list[str]) -> dict[str, Any]:
        """
        Update the transform history by appending the current transform to the transform history.
        """
        data = TransformedDict(data)

        if self._track_transform_history:
            # record the current transform in the transform history
            data.__transform_history__ = transform_history + [self.__class__.__name__]
        else:
            # do not record the current transform in the transform history
            data.__transform_history__ = transform_history

        return data

    def _check_transform_history(self, data: dict[str, Any]) -> None:
        """
        Check if the previous transforms are valid for the transform.
        Raises an error if the input is invalid.
        """
        data = TransformedDict(data)
        # Get indices of `requires_previous_transforms` in the transform history
        indices = []
        for t in self.requires_previous_transforms:
            # Ensure string comparisons
            if isinstance(t, ABCMeta):
                # case: transform was provided as class, e.g. as `RemoveKeys`
                t = t.__name__
            elif isinstance(t, Transform):
                # case: transform was provided as instance, e.g. as `RemoveKeys()`
                t = t.__class__.__name__
            else:
                # case: transform was provided as string, e.g. as `"RemoveKeys"`
                pass

            idx = data.__transform_history__.index(t) if t in data.__transform_history__ else None
            if idx is None:
                raise ValueError(
                    f"Transform `{t}` is missing from the transform history, which is {data.__transform_history__}."
                )
            indices.append(idx)

        # Check if the indices are in the correct order
        if self.previous_transforms_order_matters and (indices != sorted(indices)):
            current_order = ">".join([data.__transform_history__[i] for i in indices])
            required_order = ">".join(self.requires_previous_transforms)
            raise ValueError(
                f"Transform `{self.__class__.__name__}` requires the transforms {required_order} "
                f"to have been applied before it in this order, but the current order is {current_order}."
            )

    def __call__(self, data: dict[str, Any], *args, **kwargs) -> dict[str, Any]:
        """
        Validate and apply the transformation to the given dictionary.

        Raises:
            ValueError: If the input is invalid and raise_if_invalid_input is True.
        """
        # get previous transform history
        # (NOTE: It is neccessary to carry the transform history outside the `forward` method
        #   and the `data` object to allow users to seamlessly copy the dict and work with the
        #   dict without losing the transform history.)
        transform_history = self._get_transform_history(data)

        if self.validate_input:
            try:
                # check if the input is valid
                self._check_transform_history(data)
                self.check_input(data)
            except Exception as e:
                # if it is not valid, log or raise an error
                formatted_msg = self._format_error_msg(e)
                if self.raise_if_invalid_input:
                    logger.error(formatted_msg)
                    raise RuntimeError(formatted_msg) from e
                else:
                    logger.warning(formatted_msg)
                    return data

        # apply the transformation
        data = self.forward(data, *args, **kwargs)
        assert isinstance(
            data, dict
        ), f"`forward` method of {self.__class__.__name__} must return a dictionary, not {type(data)}."

        # record the transformation history to allow capturing dependencies in future checks
        data = self._update_transform_history(data, transform_history)
        return data

    def __repr__(self) -> str:
        """String representation of the transform for debugging, notebooks and logging."""
        # Get all the attributes of the class
        repr_str = f"{self.__class__.__name__} at {hex(id(self))}"

        if len(self.__dict__) > 0:
            attributes = [
                f"{k}={pprint.pformat(v, indent=2, depth=1, compact=True, sort_dicts=False)}"
                for k, v in self.__dict__.items()
            ]
            repr_str += "(\n " + ",\n  ".join(attributes) + "\n)"
        return repr_str


class Compose(Transform):
    """
    Compose multiple transformations together.
    """

    _track_transform_history: bool = False  # Compose does not show up in the transform history

    def __init__(self, transforms: list[Transform], track_rng_state: bool = True):
        if not isinstance(transforms, (list, tuple)):
            raise ValueError(f"Expected a list or tuple of Transforms, but got a {type(transforms)}")

        if not len(transforms) > 0:
            raise ValueError("Got an empty list of transforms.")

        if not all(isinstance(t, Transform) for t in transforms):
            invalid_type = next(t for t in transforms if not isinstance(t, Transform))
            raise ValueError(f"Expected a list or tuple of Transforms, but got a {type(invalid_type)}")

        self.transforms = transforms
        self.track_rng_state = track_rng_state

    def check_input(self, data: dict):
        # Compose is always valid
        pass

    def _update_transform_history(self, data: dict[str, Any], transform_history: list[str]) -> dict[str, Any]:
        # Compose does not track transform history
        return data

    def forward(self, data: dict, rng_state_dict: dict[str, Any] | None = None) -> dict:
        # set the RNG state context if given
        with (
            rng_state(rng_state_dict, include_cuda=False) if rng_state_dict else contextlib.nullcontext()
        ) as rng_state_dict:
            if self.track_rng_state and rng_state_dict is None:
                # collect RNG states at the start of the pipeline and execute the transforms
                rng_state_dict = _collect_rng_states()

            try:
                # execute the transforms
                for transform in self.transforms:
                    data = transform(data)
            except Exception as e:
                # construct error message including the RNG states
                msg = f"Transform pipeline failed at stage `{transform.__class__.__name__}`."
                if self.track_rng_state:
                    msg += "\nRandom number generator states at the start of the pipeline (you can instantiate the string below with `eval` for debugging):\n"
                    msg += repr(serialize_rng_state_dict(rng_state_dict))
                raise TransformPipelineError(msg, rng_state_dict) from e

        return data

    def __repr__(self) -> str:
        return "Compose(\n  " + ",\n  ".join([str(t.__class__.__name__) for t in self.transforms]) + "\n)"

    def __len__(self) -> int:
        return len(self.transforms)

    def __getitem__(self, idx: int | slice | Iterable[int]) -> Transform:
        if isinstance(idx, slice):
            return Compose(self.transforms[idx], track_rng_state=self.track_rng_state)
        elif hasattr(idx, "__iter__"):
            return Compose([self.transforms[i] for i in idx], track_rng_state=self.track_rng_state)
        else:
            return self.transforms[idx]


class RemoveKeys(Transform):
    """
    Remove keys from the data dictionary.
    """

    def __init__(self, keys: list[str], require_keys_exist: bool = True):
        self.keys = keys
        self.validate_input = require_keys_exist

    def check_input(self, data: dict):
        check_contains_keys(data, self.keys)

    def forward(self, data: dict) -> dict:
        for key in self.keys:
            if key in data:
                del data[key]
        return data


class SubsetToKeys(Transform):
    """
    Keep only the keys in the data dictionary.
    """

    def __init__(self, keys: list[str], require_keys_exist: bool = True):
        self.keys = keys
        self.validate_input = require_keys_exist

    def check_input(self, data: dict):
        check_contains_keys(data, self.keys)

    def forward(self, data: dict) -> dict:
        return {key: data[key] for key in self.keys if key in data}


class AddData(Transform):
    """
    Add data to the data dictionary.
    """

    def __init__(self, data: dict, allow_overwrite: bool = False):
        self.data = data
        self.validate_input = not allow_overwrite

    def check_input(self, data: dict):
        check_does_not_contain_keys(data, self.data.keys())

    def forward(self, data: dict) -> dict:
        data.update(self.data)
        return data


class LogData(Transform):
    """
    Log the data dictionary. Meant for debugging.
    """

    _track_transform_history: bool = False  # LogData does not show up in the transform history

    def __init__(self, log_level: int = logging.INFO, depth: int | None = 1, **pprint_kwargs):
        assert depth is None or depth > 0, "Depth must be a positive integer or None"
        self.log_level = log_level
        self.depth = depth
        self.pprint_kwargs = pprint_kwargs

    def check_input(self, data: dict):
        pass

    def forward(self, data: dict) -> dict:
        # Construct log message
        msg = "=" * 80 + "\n"
        msg += f"Data: \n{pprint.pformat(data, indent=2, depth=self.depth, sort_dicts=False, **self.pprint_kwargs)}\n"
        msg += "=" * 80

        # Log the message
        logger.log(
            level=self.log_level,
            msg=msg,
        )

        return data


class PickleToDisk(Transform):
    """
    Save the data dictionary to a pickle file.
    """

    def __init__(
        self,
        dir_path: str,
        file_name_func: Callable[[dict], str] | None = None,
        save_transform_history: bool = False,
        overwrite: bool = False,
    ):
        self.dir_path = dir_path
        self.file_name_func = file_name_func
        self.overwrite = overwrite
        self.save_transform_history = save_transform_history

        if not file_name_func:
            file_name_func = lambda data: f"{data['id']}.pkl"  # noqa

        # Ensure the directory exists
        os.makedirs(self.dir_path, exist_ok=True)

    def check_input(self, data: dict):
        check_contains_keys(data, ["id"])

    def forward(self, data: dict) -> dict:
        file_name = self.file_name_func(data)
        file_path = os.path.join(self.dir_path, file_name)
        if os.path.exists(file_path) and not self.overwrite:
            raise ValueError(f"File {file_path} already exists. Set overwrite=True to overwrite it.")

        with open(file_path, "wb") as f:
            pickle.dump(data, f)

        return data


class RaiseError(Transform):
    """
    Raises an error for testing and debugging purposes.
    """

    def check_input(self, data: dict[str, Any]) -> None:
        pass

    def forward(self, data: dict) -> dict:
        raise ValueError("User requested raising an error.")
