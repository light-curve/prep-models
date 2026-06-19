"""Shared building blocks for Chronos-family ONNX encoder wrappers.

Chronos 2 and Chronos-Bolt share the same instance-normalisation preprocessing
and the same ``context -> (mean, sequence)`` ONNX interface; only the encoder
internals differ.  The per-model wrappers live in their own packages and call
into the helpers here.
"""

from __future__ import annotations

import torch


def masked_instance_norm(
    x: torch.Tensor, *, eps: float, use_arcsinh: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Masked mean/std normalisation + optional arcsinh, ONNX-compatible.

    ``x`` is ``(batch, seq)`` with NaN marking padding.  Returns ``(normed,
    valid)`` where ``normed`` is 0 at masked positions and ``valid`` is the
    boolean observed-mask.  Uses ``torch.where`` rather than ``nanmean`` so the
    graph exports cleanly (and ``NaN * 0.0 != 0`` in IEEE 754).
    """
    valid = ~torch.isnan(x)
    zeros = torch.zeros_like(x)
    x_safe = torch.where(valid, x, zeros)  # NaN -> 0

    count = valid.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
    loc = x_safe.sum(dim=-1, keepdim=True) / count

    residual = torch.where(valid, x - loc, zeros)
    var = (residual * residual).sum(dim=-1, keepdim=True) / count
    scale = var.sqrt()
    # Replace zero scale with eps
    scale = scale + (scale == 0).float() * eps

    normed = torch.where(valid, (x_safe - loc) / scale, zeros)
    if use_arcsinh:
        # arcsinh(x) = log(x + sqrt(x^2 + 1)) — avoids aten::asinh which has no
        # TorchScript ONNX symbolic registered in any opset.
        normed = torch.log(normed + torch.sqrt(normed * normed + 1.0))

    return normed, valid
