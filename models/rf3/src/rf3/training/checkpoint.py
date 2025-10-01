"""Utilities for gradient checkpointing to reduce memory usage during training.

Gradient checkpointing (also called activation checkpointing) trades compute for memory
by recomputing intermediate activations during the backward pass instead of storing them.
This enables training larger models or using larger batch sizes within GPU memory constraints.

References:
  * `PyTorch Checkpoint Documentation`_

  .. _PyTorch Checkpoint Documentation: https://pytorch.org/docs/stable/checkpoint.html
"""

import torch
from torch.utils.checkpoint import checkpoint


def create_custom_forward(module, **kwargs):
    """Create a custom forward function for gradient checkpointing with fixed kwargs.

    This helper enables passing keyword arguments to a module when using PyTorch's
    checkpoint function, which only accepts positional arguments for the function to
    be checkpointed.

    Args:
      module: The callable (typically a nn.Module) to wrap.
      **kwargs: Keyword arguments to pass to the module during forward.

    Returns:
      A callable that accepts only positional arguments and forwards them along
      with the fixed kwargs to the original module.

    Examples:
      Use with PyTorch checkpoint::

        custom_fn = create_custom_forward(my_module, frame_atom_idxs=frame_idxs)
        output = checkpoint(custom_fn, input_tensor, use_reentrant=False)

    See Also:
      :py:func:`activation_checkpointing`
    """

    def custom_forward(*inputs):
        return module(*inputs, **kwargs)

    return custom_forward


def activation_checkpointing(function):
    """Decorator to enable gradient checkpointing for a function during training.

    When gradients are enabled (training mode), this decorator wraps the function
    with PyTorch's checkpoint to save memory by recomputing activations during
    the backward pass. During inference (gradients disabled), the function runs
    normally without checkpointing overhead.

    Args:
      function: The function to apply gradient checkpointing to.

    Returns:
      Wrapped function that conditionally applies checkpointing based on gradient state.

    Examples:
      Apply to a forward pass method::

        @activation_checkpointing
        def forward(self, x, mask=None):
            return self.layer(x, mask)

    Notes:
      Uses ``use_reentrant=False`` for better compatibility with modern PyTorch
      features like autograd hooks and higher-order gradients.

    See Also:
      :py:func:`create_custom_forward`
    """

    def wrapper(*args, **kwargs):
        if torch.is_grad_enabled():
            return checkpoint(
                create_custom_forward(function, **kwargs), *args, use_reentrant=False
            )
        return function(*args, **kwargs)

    return wrapper
