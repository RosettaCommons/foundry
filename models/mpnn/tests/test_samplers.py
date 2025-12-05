"""Tests for MPNN samplers."""

import pandas as pd
import pytest
from mpnn.samplers.samplers import PaddedTokenBudgetBatchSampler
from torch.utils.data import Sampler


class MockIndexSampler(Sampler):
    """Mock sampler that returns indices."""

    def __init__(self, data_df):
        self.data = data_df

    def __iter__(self):
        # Return indices (0, 1, 2, ...)
        for i in range(len(self.data)):
            yield i

    def __len__(self):
        return len(self.data)


@pytest.fixture
def test_data():
    """Create mock data with varying token counts."""
    return pd.DataFrame(
        {
            "n_non_atomized_tokens": [50, 100, 150, 75, 200, 25, 175, 80],
            "other_data": ["a", "b", "c", "d", "e", "f", "g", "h"],
        }
    )


@pytest.fixture
def index_sampler(test_data):
    """Create mock index sampler."""
    return MockIndexSampler(test_data)


@pytest.fixture
def get_num_tokens(test_data):
    """Create token extraction function that works with indices."""

    def get_tokens(idx):
        return test_data.iloc[idx]["n_non_atomized_tokens"]

    return get_tokens


class TestPaddedTokenBudgetBatchSampler:
    """Test cases for PaddedTokenBudgetBatchSampler."""

    def test_basic_functionality(self, index_sampler, get_num_tokens, test_data):
        """Test basic sampler functionality."""
        sampler = PaddedTokenBudgetBatchSampler(
            sampler=index_sampler,
            get_num_tokens=get_num_tokens,
            max_tokens_with_padding=200,
            shuffle_batches=False,
        )

        batches = list(sampler)

        # Should create some batches
        assert len(batches) > 0

        # All indices should be present exactly once
        all_indices = []
        for batch in batches:
            all_indices.extend(batch)

        # Check that we have the right number of indices
        assert len(all_indices) == len(test_data)

        # Check that all token counts are present
        token_counts_in_batches = [get_num_tokens(idx) for idx in all_indices]
        original_token_counts = test_data["n_non_atomized_tokens"].tolist()
        assert sorted(token_counts_in_batches) == sorted(original_token_counts)

    def test_max_tokens_constraint(self, index_sampler, get_num_tokens):
        """Test that max_tokens_with_padding constraint is respected."""
        max_tokens = 200
        sampler = PaddedTokenBudgetBatchSampler(
            sampler=index_sampler,
            get_num_tokens=get_num_tokens,
            max_tokens_with_padding=max_tokens,
            shuffle_batches=False,
        )

        batches = list(sampler)

        for batch in batches:
            # Get token counts for this batch
            token_counts = [get_num_tokens(idx) for idx in batch]
            max_tokens_in_batch = max(token_counts)
            batch_size = len(batch)

            # Check constraint: max_length * batch_size <= max_tokens_with_padding
            # Exception: single samples that exceed max_tokens are not allowed (should raise ValueError)
            assert (
                max_tokens_in_batch * batch_size <= max_tokens
            ), f"Batch violates max_tokens_with_padding constraint: {max_tokens_in_batch} * {batch_size} > {max_tokens}"

    def test_length_sorting(self, index_sampler, get_num_tokens):
        """Test that samples are generally grouped by similar lengths."""
        sampler = PaddedTokenBudgetBatchSampler(
            sampler=index_sampler,
            get_num_tokens=get_num_tokens,
            max_tokens_with_padding=1000,  # Large enough to allow multiple items per batch
            shuffle_batches=False,
        )

        batches = list(sampler)

        # Check that within each batch, token counts are reasonably similar
        for batch in batches:
            if len(batch) > 1:
                token_counts = [get_num_tokens(idx) for idx in batch]
                min_tokens = min(token_counts)
                max_tokens = max(token_counts)

                # The ratio shouldn't be too extreme for batches with multiple items
                if min_tokens > 0:
                    ratio = max_tokens / min_tokens
                    assert ratio < 10.0, "Token counts in batch vary too much"

    def test_empty_sampler(self):
        """Test behavior with empty sampler."""
        empty_data = pd.DataFrame({"n_non_atomized_tokens": []})
        empty_sampler = MockIndexSampler(empty_data)

        def get_tokens_empty(idx):
            return empty_data.iloc[idx]["n_non_atomized_tokens"]

        sampler = PaddedTokenBudgetBatchSampler(
            sampler=empty_sampler,
            get_num_tokens=get_tokens_empty,
            max_tokens_with_padding=200,
        )

        batches = list(sampler)
        assert len(batches) == 0

    def test_single_large_sample(self):
        """Test handling of samples that exceed max_tokens_with_padding."""
        large_data = pd.DataFrame(
            {
                "n_non_atomized_tokens": [
                    1000,
                    50,
                ],  # First sample exceeds max_tokens_with_padding
            }
        )
        large_sampler = MockIndexSampler(large_data)

        def get_tokens_large(idx):
            return large_data.iloc[idx]["n_non_atomized_tokens"]

        # This should raise a ValueError because we don't allow samples that exceed max_tokens_with_padding
        with pytest.raises(ValueError, match="exceeding max_tokens_with_padding"):
            sampler = PaddedTokenBudgetBatchSampler(
                sampler=large_sampler,
                get_num_tokens=get_tokens_large,
                max_tokens_with_padding=200,
            )
            list(sampler)  # Trigger the batching

    @pytest.mark.parametrize("max_tokens_with_padding", [200, 500])
    @pytest.mark.parametrize("shuffle_batches", [True, False])
    def test_parameterized_batching(
        self,
        index_sampler,
        get_num_tokens,
        max_tokens_with_padding,
        shuffle_batches,
        test_data,
    ):
        """Test batching with different parameters."""
        sampler = PaddedTokenBudgetBatchSampler(
            sampler=index_sampler,
            get_num_tokens=get_num_tokens,
            max_tokens_with_padding=max_tokens_with_padding,
            shuffle_batches=shuffle_batches,
        )

        batches = list(sampler)

        # Should create some batches
        assert len(batches) > 0

        # All indices should be present exactly once
        all_indices = []
        for batch in batches:
            all_indices.extend(batch)

        # Check that we have the right number of indices
        assert len(all_indices) == len(test_data)

        # Check that all token counts are present
        token_counts_in_batches = [get_num_tokens(idx) for idx in all_indices]
        original_token_counts = test_data["n_non_atomized_tokens"].tolist()
        assert sorted(token_counts_in_batches) == sorted(original_token_counts)

    def test_invalid_inputs(self, index_sampler, get_num_tokens):
        """Test validation of invalid inputs."""
        # Test max_tokens_with_padding <= 0
        with pytest.raises(
            ValueError, match="max_tokens_with_padding must be greater than 0"
        ):
            PaddedTokenBudgetBatchSampler(
                sampler=index_sampler,
                get_num_tokens=get_num_tokens,
                max_tokens_with_padding=0,
            )

        # Test negative max_tokens_with_padding
        with pytest.raises(
            ValueError, match="max_tokens_with_padding must be greater than 0"
        ):
            PaddedTokenBudgetBatchSampler(
                sampler=index_sampler,
                get_num_tokens=get_num_tokens,
                max_tokens_with_padding=-100,
            )

        # Test None get_num_tokens
        with pytest.raises(
            ValueError, match="get_num_tokens function must be provided"
        ):
            PaddedTokenBudgetBatchSampler(
                sampler=index_sampler, get_num_tokens=None, max_tokens_with_padding=200
            )

    def test_invalid_token_types(self):
        """Test validation of invalid token types returned by get_num_tokens."""
        bad_data = pd.DataFrame({"n_non_atomized_tokens": ["invalid", 100]})
        bad_sampler = MockIndexSampler(bad_data)

        def get_tokens_bad(idx):
            return bad_data.iloc[idx]["n_non_atomized_tokens"]

        sampler = PaddedTokenBudgetBatchSampler(
            sampler=bad_sampler,
            get_num_tokens=get_tokens_bad,
            max_tokens_with_padding=200,
        )

        with pytest.raises(TypeError, match="invalid type"):
            list(sampler)  # Trigger the batching

    def test_invalid_token_values(self, index_sampler):
        """Test validation of invalid token values."""

        def bad_get_tokens(idx):
            return -10  # Invalid negative value

        sampler = PaddedTokenBudgetBatchSampler(
            sampler=index_sampler,
            get_num_tokens=bad_get_tokens,
            max_tokens_with_padding=200,
        )

        with pytest.raises(ValueError, match="invalid value"):
            list(sampler)  # Trigger the batching
