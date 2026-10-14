"""Conservative dispatch for RF3's measured, bitwise-exact inference kernels."""

from functools import lru_cache

import torch
from torch import nn


@lru_cache(maxsize=None)
def validated_stack(device_index: int) -> bool:
    """Changing this allowlist requires repeating the checkpoint benchmark."""
    if torch.__version__.split("+")[0] != "2.11.0" or torch.version.cuda != "12.8":
        return False
    if "H200" not in torch.cuda.get_device_name(device_index):
        return False
    try:
        from cuda.bindings import driver, nvrtc  # noqa: F401
    except ImportError:
        return False
    return True


class ExactLayerNorm(nn.LayerNorm):
    """ATen's arithmetic on a different schedule; training always uses ATen."""

    exact_enabled: bool = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            not self.exact_enabled
            or self.training
            or torch.is_grad_enabled()
            or type(x) is not torch.Tensor
            or not x.is_cuda
            or self.normalized_shape not in ((64,), (128,), (256,))
            or x.numel() // x.shape[-1] < 131072
            or x.dtype not in (torch.float32, torch.bfloat16)
            or any(
                p is not None
                and (
                    p.dtype != torch.float32
                    or p.device != x.device
                    or not p.is_contiguous()
                )
                for p in (self.weight, self.bias)
            )
            or not validated_stack(x.get_device())
        ):
            return super().forward(x)
        # CUDA autocast promotes LayerNorm to FP32, including its output.
        # Plain BF16 LayerNorm and misaligned vector loads use different ATen paths.
        if x.dtype == torch.bfloat16 and not torch.is_autocast_enabled("cuda"):
            return super().forward(x)
        if x.dtype == torch.float32 and x.is_contiguous() and x.data_ptr() % 16:
            return super().forward(x)
        from rf3.model.kernels.exact_layer_norm import Unsupported, layer_norm

        try:
            return layer_norm(
                x,
                self.normalized_shape,
                self.weight,
                self.bias,
                self.eps,
                widen=x.dtype == torch.bfloat16,
            )
        except Unsupported:
            return super().forward(x)


def set_exact_kernels(model: nn.Module, enabled: bool) -> None:
    """Install on this model only, retaining parameter objects and state-dict keys.

    Calling again switches existing adapters. No process-wide monkeypatch or
    checkpoint conversion is involved. Custom LayerNorm subclasses and modules
    with hooks retain their original behavior.
    """
    replacements: dict[int, ExactLayerNorm] = {}
    for parent in list(model.modules()):
        for name, child in list(parent._modules.items()):
            if isinstance(child, ExactLayerNorm):
                child.exact_enabled = enabled
            elif (
                enabled
                and type(child) is nn.LayerNorm
                and child.normalized_shape in ((64,), (128,), (256,))
                and not any(
                    (
                        child._forward_hooks,
                        child._forward_pre_hooks,
                        child._backward_hooks,
                        child._backward_pre_hooks,
                        child._state_dict_hooks,
                        child._state_dict_pre_hooks,
                        child._load_state_dict_pre_hooks,
                        child._load_state_dict_post_hooks,
                    )
                )
            ):
                replacement = replacements.get(id(child))
                if replacement is None:
                    replacement = ExactLayerNorm(
                        list(child.normalized_shape),
                        child.eps,
                        child.elementwise_affine,
                        bias=child.bias is not None,
                        device="meta",
                    )
                    replacement.weight, replacement.bias = child.weight, child.bias
                    replacement.train(child.training)
                    replacements[id(child)] = replacement
                setattr(parent, name, replacement)
