"""PyTorch-backed device runtime for ForgeML's CUDA milestone.

The standalone CUDA kernels remain the research reference. This module provides
the production Python boundary: tensors stay on their selected device, normal
PyTorch autograd supplies backward propagation, and CPU fallback is explicit.
"""
from __future__ import annotations

import math
from typing import Optional

try:
    import torch
except ImportError:  # pragma: no cover - exercised in minimal CPU installs
    torch = None


def require_torch():
    if torch is None:
        raise RuntimeError("ForgeML CUDA backend requires the optional 'torch' dependency")
    return torch


def select_device(preferred: Optional[str] = None):
    """Return a torch.device, falling back to CPU when CUDA is unavailable."""
    t = require_torch()
    if preferred:
        requested = t.device(preferred)
        if requested.type == "cuda" and not t.cuda.is_available():
            return t.device("cpu")
        return requested
    return t.device("cuda" if t.cuda.is_available() else "cpu")


def device_info() -> dict:
    t = require_torch()
    info = {"cuda_available": bool(t.cuda.is_available()), "torch_version": t.__version__}
    if t.cuda.is_available():
        info.update({"device": t.cuda.get_device_name(0), "cuda_version": t.version.cuda})
    else:
        info.update({"device": "cpu", "cuda_version": None})
    return info


def causal_attention(q, k, v, *, scale: Optional[float] = None):
    """Memory-safe causal attention using device-native PyTorch operations.

    Inputs are ``[batch, heads, sequence, head_dim]`` and may be CPU or CUDA
    tensors. The operation is differentiable with respect to all three inputs.
    ``scaled_dot_product_attention`` is used when available; the fallback keeps
    the same contract for older PyTorch versions.
    """
    t = require_torch()
    if not all(isinstance(x, t.Tensor) for x in (q, k, v)):
        raise TypeError("q, k and v must be torch.Tensor objects")
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k and v must have equal [B,H,T,D] shapes")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k and v must be on the same device")
    if q.dtype not in (t.float16, t.float32, t.float64, t.bfloat16):
        raise TypeError("attention expects a floating-point dtype")
    scale = (q.shape[-1] ** -0.5) if scale is None else scale
    if hasattr(t.nn.functional, "scaled_dot_product_attention"):
        return t.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
    scores = t.matmul(q, k.transpose(-2, -1)) * scale
    mask = t.triu(t.ones((q.shape[-2], q.shape[-1]), device=q.device, dtype=t.bool), diagonal=1)
    scores = scores.masked_fill(mask, -t.inf)
    return t.softmax(scores, dim=-1) @ v


def softmax(x, dim=-1):
    t = require_torch()
    return t.softmax(x, dim=dim)


def layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5):
    t = require_torch()
    return t.nn.functional.layer_norm(x, normalized_shape, weight, bias, eps)


def gelu(x):
    return require_torch().nn.functional.gelu(x)


def matmul(a, b):
    return require_torch().matmul(a, b)


def reduce_sum(x, dim=None, keepdim=False):
    return require_torch().sum(x, dim=dim, keepdim=keepdim)
