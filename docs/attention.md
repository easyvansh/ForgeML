# Attention implementation notes

There are two separate paths today. `nn.Attention` is differentiable dense attention used in training. `attention.streaming_attention` is a forward-only NumPy algorithm reference used in benchmarks. Switching the training path to the streaming function is not supported.

For each query tile, maintain row maximum m, denominator l, and unnormalized output A. For the next key/value tile with scores S:

```text
m_new = max(m, rowmax(S))
alpha = exp(m - m_new)
P = exp(S - m_new)
l_new = alpha * l + rowsum(P)
A_new = alpha * A + P @ V_tile
output = A / l
```

Causal positions are masked before the row maximum. The first processed key tile always includes a valid key for every active query row. Query and key loops handle partial tiles. The implementation assumes equal Q/K/V shapes [B,H,T,D] and does not handle cross-attention, arbitrary masks, attention dropout or grouped-query attention.

Dense score storage grows as BHT² elements. With fixed tile sizes, the streaming score temporary is bounded by the tile dimensions; input/output storage still grows with T. This is an algorithmic storage observation, **not a measured GPU memory result**. Python/NumPy allocation and BLAS workspaces complicate CPU peak-memory measurements.

The algorithm is inspired by [FlashAttention](https://arxiv.org/abs/2205.14135), but this code does not implement its CUDA scheduling or optimized backward. [FlashAttention-2](https://arxiv.org/abs/2307.08691) motivates future work partitioning experiments; no FA2 reproduction is claimed.

Before a GPU speed claim, add backward, finite-difference and PyTorch parity, non-multiple tiles, FP32/FP16 stress tests, memory safety checks, warmup, CUDA-event timings, and a full training-step comparison. Persist the selected baseline backend and launch parameters.
