"""Conservative dispatch for RF3's measured, bitwise-exact inference kernels."""

import logging
from functools import lru_cache

import torch
from torch import nn

# Inference quiet mode raises the root level to ERROR and silences Python warnings.
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


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


@lru_cache(maxsize=None)
def _warn_fallback(reason: str) -> None:
    """Report each inference fallback reason once per process, not per layer."""
    logger.warning("RF3 optimized LayerNorm is using PyTorch fallback: %s.", reason)


class ExactLayerNorm(nn.LayerNorm):
    """ATen's arithmetic on a different schedule; training always uses ATen."""

    def _fallback(self, x: torch.Tensor, reason: str) -> torch.Tensor:
        _warn_fallback(reason)
        return super().forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Training and autograd intentionally use the native operation.
        if self.training or torch.is_grad_enabled():
            return super().forward(x)
        if type(x) is not torch.Tensor or not x.is_cuda:
            return self._fallback(x, "a plain CUDA tensor is required")
        if not validated_stack(x.get_device()):
            return self._fallback(
                x, "requires H200, Torch 2.11.0/CUDA 12.8 and cuda.bindings"
            )
        if self.normalized_shape not in ((64,), (128,), (256,)):
            return self._fallback(x, "unsupported feature width")
        if x.numel() // x.shape[-1] < 131072:
            return self._fallback(
                x, "below the measured speedup threshold of 131072 rows"
            )
        if x.dtype not in (torch.float32, torch.bfloat16) or any(
            p is not None
            and (
                p.dtype != torch.float32
                or p.device != x.device
                or not p.is_contiguous()
            )
            for p in (self.weight, self.bias)
        ):
            return self._fallback(x, "unsupported input or parameter dtype/layout")
        # CUDA autocast promotes LayerNorm to FP32, including its output.
        # Plain BF16 LayerNorm and misaligned vector loads use different ATen paths.
        if x.dtype == torch.bfloat16 and not torch.is_autocast_enabled("cuda"):
            return self._fallback(x, "BF16 input requires CUDA autocast")
        if x.dtype == torch.float32 and x.is_contiguous() and x.data_ptr() % 16:
            return self._fallback(x, "input is not aligned for vector loads")
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
        except Unsupported as exc:
            return self._fallback(x, str(exc))


def enable_exact_kernels(model: nn.Module) -> None:
    """Install on this model only, retaining parameter objects and state-dict keys.

    Repeated calls preserve existing adapters. No process-wide monkeypatch or
    checkpoint conversion is involved. Custom LayerNorm subclasses and modules
    with hooks retain their original behavior.
    """
    replacements: dict[int, ExactLayerNorm] = {}
    for parent in list(model.modules()):
        for name, child in list(parent._modules.items()):
            if (
                type(child) is nn.LayerNorm
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
