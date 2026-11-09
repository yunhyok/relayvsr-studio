"""FP8 (E4M3) weight-only storage for the Wan transformer blocks.

The one-step distilled Wan model is 5.0B parameters: 10 GB in BF16. Storing the block
linear weights as float8_e4m3fn with one scale per output row halves that to about
5 GB while the matmuls still run in BF16 (weights are dequantized per layer on the
fly). This is the default on GPUs with less than ~20 GB of VRAM, which includes the
RTX 5070 Ti the app is tuned for. Dequantization costs a few milliseconds per keyframe.

RTX 40/50 series GPUs convert FP8 natively; older GPUs still work because the cast is
done by PyTorch element-wise kernels.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

FP8_E4M3_MAX = 448.0


class Float8Linear(nn.Module):
    """nn.Linear replacement holding an FP8 weight plus per-row BF16 scales."""

    def __init__(self, weight_fp8: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None):
        super().__init__()
        self.weight = nn.Parameter(weight_fp8, requires_grad=False)
        self.scale = nn.Parameter(scale, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None
        self.in_features = weight_fp8.shape[1]
        self.out_features = weight_fp8.shape[0]

    def dequantized_weight(self, dtype: torch.dtype) -> torch.Tensor:
        return self.weight.to(dtype) * self.scale.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.dequantized_weight(x.dtype), bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, fp8=e4m3, bias={self.bias is not None}"


@torch.no_grad()
def quantize_linear(linear: nn.Linear, *, device: torch.device, scale_dtype=torch.bfloat16) -> Float8Linear:
    weight = linear.weight.detach().to(device=device, dtype=torch.float32)
    amax = weight.abs().amax(dim=1, keepdim=True).clamp_(min=1e-8)
    scale = amax / FP8_E4M3_MAX
    q = (weight / scale).clamp_(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    bias = linear.bias.detach().to(device=device, dtype=scale_dtype) if linear.bias is not None else None
    return Float8Linear(q, scale.to(scale_dtype), bias)


@torch.no_grad()
def quantize_module_linears_(module: nn.Module, *, device: torch.device, scale_dtype=torch.bfloat16) -> int:
    """Replace every nn.Linear below ``module`` (in place) by Float8Linear. Returns count."""
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, quantize_linear(child, device=device, scale_dtype=scale_dtype))
            count += 1
        else:
            count += quantize_module_linears_(child, device=device, scale_dtype=scale_dtype)
    return count


def fp8_supported() -> bool:
    try:
        torch.zeros(1, dtype=torch.float8_e4m3fn)
        return True
    except (TypeError, RuntimeError, AttributeError):  # pragma: no cover
        return False


__all__ = ["Float8Linear", "fp8_supported", "quantize_linear", "quantize_module_linears_"]
