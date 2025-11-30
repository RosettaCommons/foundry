from typing import Any, Callable, Iterator, List

import numpy as np
import torch
from atomworks.ml.samplers import set_sampler_epoch
from torch.utils.data import BatchSampler, Sampler


class PaddedTokenBudgetBatchSampler(BatchSampler):
    """
    Token-based batch sampler that wraps existing samplers and creates batches
    of similar-token length samples, respecting a maximum token count
    constraint (considering that the batches will be padded to the maximum
    length in the batch).

    Args:
        sampler: The underlying sampler to wrap around.
        get_num_tokens: Function that takes an index from the previous sampler
            and returns the number of tokens for that sample.
        max_tokens_with_padding: Maximum number of tokens allowed per batch,
            including padding. The constraint is
            max(batch_lengths) * len(batch) <= max_tokens.
        shuffle_batches: Whether to randomize the order of batches after
            grouping by length. Defaults to True.
    """

    def __init__(
        self,
        sampler: Sampler,
        get_num_tokens: Callable[[Any], int],
        max_tokens_with_padding: int = 6000,
        shuffle_batches: bool = True,
    ):
        # Initialize BatchSampler with a dummy batch_size (we don't use it).
        super().__init__(sampler, batch_size=1, drop_last=False)

        self.sampler = sampler
        self.get_num_tokens = get_num_tokens
        self.max_tokens_with_padding = max_tokens_with_padding
        self.shuffle_batches = shuffle_batches

        # Add drop_last attribute for DataLoader compatibility
        self.drop_last = False

        self.epoch = 0

        # Will hold our "one epoch" cache.
        self._batches: List[List[Any]] | None = None

        # Validate inputs
        if max_tokens_with_padding <= 0:
            raise ValueError("max_tokens_with_padding must be greater than 0")
        if get_num_tokens is None:
            raise ValueError("get_num_tokens function must be provided")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        set_sampler_epoch(self.sampler, epoch)

    def _build_batches(self) -> List[List[Any]]:
        """
        Compute all batches and cache them for the current epoch.

        Returns:
            List of batches, where each batch is a list of indices.
        Raises:
            TypeError: If get_num_tokens returns invalid types.
            ValueError: If get_num_tokens returns invalid values.
        """
        # Extract all indices and their token counts
        sample_indices_and_lengths = []
        for idx in self.sampler:
            num_tokens = self.get_num_tokens(idx)

            # Validate num_tokens type
            if not isinstance(num_tokens, (int, float, np.integer, np.floating)):
                raise TypeError(
                    f"get_num_tokens returned invalid type {type(num_tokens)} "
                    f"for index {idx}. Expected numeric type."
                )

            # Validate num_tokens value.
            num_tokens = int(num_tokens)
            if num_tokens <= 0:
                raise ValueError(
                    f"get_num_tokens returned invalid value {num_tokens} "
                    f"for index {idx}. Expected positive integer."
                )
            if num_tokens > self.max_tokens_with_padding:
                raise ValueError(
                    f"Index {idx} has {num_tokens} tokens, exceeding "
                    f"max_tokens_with_padding={self.max_tokens_with_padding}."
                )

            sample_indices_and_lengths.append((idx, num_tokens))

        # Sort by token length (ascending order).
        sample_indices_and_lengths.sort(key=lambda x: x[1])

        # Batch by length
        batches = []
        current_batch = []
        current_max_length = 0
        for idx, length in sample_indices_and_lengths:
            # Check if adding this sample would violate max_tokens_with_padding
            # constraint.
            if current_batch:
                potential_max_length = max(length, current_max_length)
                new_batch_size = len(current_batch) + 1

                if potential_max_length * new_batch_size > self.max_tokens_with_padding:
                    # Current batch is full, start a new batch
                    batches.append(current_batch)

                    current_batch = [idx]
                    current_max_length = length
                else:
                    # Add to current batch
                    current_batch.append(idx)
                    current_max_length = potential_max_length
            else:
                # First sample in batch
                current_batch = [idx]
                current_max_length = length

        # Add the last batch if it's not empty
        if current_batch:
            batches.append(current_batch)

        # Randomize batch order if requested
        if self.shuffle_batches:
            # Set the seed based on the epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)

            perm = torch.randperm(len(batches), generator=g).tolist()

            batches = [batches[i] for i in perm]

        return batches

    def __iter__(self) -> Iterator[List[Any]]:
        """
        Generate batches of indices grouped by token length while respecting
        the max_tokens_with_padding constraint.

        Returns:
            Iterator[List[Any]]: Iterator over batches of dataset indices.
        """
        # Build/reference cached batches if not already done.
        if self._batches is None:
            self._batches = self._build_batches()

        for batch in self._batches:
            yield batch

        # End of __iter__ — clear cache so next call recomputes.
        self._batches = None

    def __len__(self) -> int:
        """
        Return the exact number of batches that will be produced.
        """
        # Build/reference cached batches if not already done.
        if self._batches is None:
            self._batches = self._build_batches()
        return len(self._batches)
