# Python CUDA runtime boundary

`forge.cuda_backend` is the first device-aware Python boundary for ForgeML. It
accepts `[B,H,T,D]` PyTorch tensors, preserves their device and dtype, exposes
causal attention with autograd, and provides reusable softmax, LayerNorm, GELU,
matmul, and reduction primitives. `forge.device.DeviceTensor` makes ownership
and placement explicit at API boundaries.

The backend uses PyTorch's device-native scaled dot-product attention when
available and a mathematically equivalent masked fallback otherwise. This gives
ForgeML a stable training API while the standalone CUDA kernels remain the
low-level research implementations being optimized and profiled.

```python
from forge import causal_attention, select_device
import torch

device = select_device()
q = torch.randn(1, 2, 128, 32, device=device, requires_grad=True)
out = causal_attention(q, q, q)
out.square().mean().backward()
```

The unit test validates shape, finite gradients, and device selection on CPU so
the regular suite remains portable. Run the CUDA kernel sweeps separately on a
machine with a working driver/toolkit pair. A future extension will replace the
PyTorch dispatch with the checked-in custom kernels without changing this API.
