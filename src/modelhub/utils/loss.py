import torch


def convert_batched_losses_to_list_of_dicts(loss_dict: dict[str, torch.Tensor]):
    """Converts a dictionary of batched and non-batched loss tensors into a list of dictionaries.

    Args:
        loss_dict (dict): A dictionary where keys are loss names and values are PyTorch tensors.
                          Some values may be batched (1D tensors), while others are not (0D tensors).

    Returns:
        list: A list of dictionaries, each representing a batch or non-batched losses.

    Example:
        >>> outputs = {
        ...     "loss_dict": {
        ...         "diffusion_loss": torch.tensor([0.0509, 0.0062]),
        ...         "smoothed_lddt_loss": torch.tensor([0.2507, 0.2797]),
        ...         "t": torch.tensor([1.7329, 9.3498]),
        ...         "distogram_loss": torch.tensor(1.7663),
        ...         "total_loss": torch.tensor(1.2281),
        ...     }
        ... }
        >>> convert_batched_losses_to_list_of_dicts(outputs["loss_dict"])
        [{'batch_idx': 0, 'diffusion_loss': 0.0509, 'smoothed_lddt_loss': 0.2507, 't': 1.7329},
         {'batch_idx': 1, 'diffusion_loss': 0.0062, 'smoothed_lddt_loss': 0.2797, 't': 9.3498},
         {'distogram_loss': 1.7663, 'total_loss': 1.2281}]
    """
    result = []
    batch_size = next((v.size(0) for v in loss_dict.values() if v.dim() == 1), 1)

    # Create a dictionary for each batch index
    for batch_idx in range(batch_size):
        batch_dict = {"batch_idx": batch_idx}

        for key, value in loss_dict.items():
            if value.dim() == 1:  # Check if the tensor is batched
                batch_dict[key] = value[batch_idx].item()

        result.append(batch_dict)

    # Create a dictionary for non-batched losses
    non_batched_dict = {}
    for key, value in loss_dict.items():
        if value.dim() == 0:  # Check if the tensor is not batched
            non_batched_dict[key] = value.item()

    result.append(non_batched_dict)

    return result


def mean_losses(loss_dict_batched: dict[str, torch.Tensor]) -> dict:
    """Compute the mean of each tensor in a dictionary of batched losses.

    Args:
        loss_dict_batched (Dict[str, torch.Tensor]): A dictionary where each key maps to a tensor of losses.

    Returns:
        dict: A dictionary with the mean loss for each key (as a tensor).

    Example:
        >>> loss_dict_batched = {"loss1": torch.tensor([0.5, 0.7]), "loss2": torch.tensor([1.0])}
        >>> mean_losses(loss_dict_batched)
        {'loss1': 0.6, 'loss2': 1.0}
    """
    loss_dict = {}
    for key, batched_loss in loss_dict_batched.items():
        # Compute the mean of the tensor and store it in the dictionary
        loss_dict[key] = batched_loss.mean().item()

    return loss_dict
