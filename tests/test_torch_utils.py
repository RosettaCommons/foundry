import os

import pytest
import torch

os.environ["NAN_CHECKING"] = "True"
from foundry.utils.torch import assert_no_nans, map_to


def test_map_to():
    # Test with a simple tensor
    tensor = torch.tensor([1, 2, 3])
    result = map_to(tensor, device="cpu", dtype=torch.float32)
    assert isinstance(result, torch.Tensor)
    assert result.device.type == "cpu"
    assert result.dtype == torch.float32
    assert torch.all(result.eq(torch.tensor([1.0, 2.0, 3.0])))

    # Test with a nested structure
    data = {
        "tensor": torch.tensor([1, 2, 3]),
        "list": [torch.tensor([4, 5]), "string"],
        "nested": {"tensor": torch.tensor([6, 7, 8])},
    }
    result = map_to(data, device="cpu", dtype=torch.float64)

    assert isinstance(result, dict)
    assert isinstance(result["tensor"], torch.Tensor)
    assert result["tensor"].device.type == "cpu"
    assert result["tensor"].dtype == torch.float64
    assert torch.all(
        result["tensor"].eq(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
    )

    assert isinstance(result["list"], list)
    assert isinstance(result["list"][0], torch.Tensor)
    assert result["list"][0].device.type == "cpu"
    assert result["list"][0].dtype == torch.float64
    assert torch.all(
        result["list"][0].eq(torch.tensor([4.0, 5.0], dtype=torch.float64))
    )
    assert result["list"][1] == "string"

    assert isinstance(result["nested"], dict)
    assert isinstance(result["nested"]["tensor"], torch.Tensor)
    assert result["nested"]["tensor"].device.type == "cpu"
    assert result["nested"]["tensor"].dtype == torch.float64
    assert torch.all(
        result["nested"]["tensor"].eq(
            torch.tensor([6.0, 7.0, 8.0], dtype=torch.float64)
        )
    )

    # Test with non-tensor types
    non_tensor_data = {"string": "hello", "int": 42, "float": 3.14}
    result = map_to(non_tensor_data, device="cpu", dtype=torch.float32)
    assert result == non_tensor_data

    # Test with empty input
    assert map_to({}, device="cpu", dtype=torch.float32) == {}
    assert map_to([], device="cpu", dtype=torch.float32) == []

    # Test error case: no device or dtype provided
    with pytest.raises(AssertionError):
        map_to(tensor)


def test_assert_no_nans():
    # Test with clean tensor
    clean_tensor = torch.tensor([1.0, 2.0, 3.0])
    assert_no_nans(clean_tensor)  # Should not raise

    # Test with tensor containing NaNs
    nan_tensor = torch.tensor([1.0, float("nan"), 3.0])
    with pytest.raises(AssertionError, match="Tensor contains NaNs!"):
        assert_no_nans(nan_tensor)

    # Test with numpy array
    import numpy as np

    clean_array = np.array([1.0, 2.0, 3.0])
    assert_no_nans(clean_array)  # Should not raise

    nan_array = np.array([1.0, np.nan, 3.0])
    with pytest.raises(AssertionError, match="Numpy array contains NaNs!"):
        assert_no_nans(nan_array)

    # Test with float
    clean_float = 1.0
    assert_no_nans(clean_float)  # Should not raise

    nan_float = float("nan")
    with pytest.raises(AssertionError, match="float is NaN!"):
        assert_no_nans(nan_float)

    # Test with nested dictionary
    clean_dict = {
        "a": torch.tensor([1.0, 2.0]),
        "b": {"c": np.array([3.0, 4.0])},
        "d": 5.0,
    }
    assert_no_nans(clean_dict)  # Should not raise

    nan_dict = {
        "a": torch.tensor([1.0, float("nan")]),
        "b": {"c": torch.tensor([3.0, 4.0])},
    }
    with pytest.raises(AssertionError, match=r"a: Tensor contains NaNs!"):
        assert_no_nans(nan_dict)

    # Test with nested list/tuple
    clean_list = [torch.tensor([1.0, 2.0]), (np.array([3.0, 4.0]),)]
    assert_no_nans(clean_list)  # Should not raise

    nan_list = [torch.tensor([1.0, 2.0]), (torch.tensor([float("nan"), 4.0]),)]
    with pytest.raises(AssertionError, match=r"1.0: Tensor contains NaNs!"):
        assert_no_nans(nan_list)

    # Test with fail_if_not_tensor=True
    with pytest.raises(ValueError, match="Unsupported type"):
        assert_no_nans(42, fail_if_not_tensor=True)

    # Test that integers don't raise error with fail_if_not_tensor=False
    assert_no_nans(42)  # Should not raise

    # Test custom error message
    with pytest.raises(AssertionError, match="custom.a: Tensor contains NaNs!"):
        assert_no_nans({"a": torch.tensor([1.0, float("nan")])}, msg="custom")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
