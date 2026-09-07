# Online causal attention algorithm

For each query row and key tile, ForgeML maintains:

$$m = \max_j s_j, \qquad \ell = \sum_j e^{s_j-m}, \qquad a = \sum_j e^{s_j-m}v_j.$$ 

```text
m'     = max(m, row_max(scores))
scale  = exp(m - m')
probs  = exp(scores - m')
ell'   = scale * ell + row_sum(probs)
a'     = scale * a + probs @ values_tile
output = a' / ell'
```

The causal mask removes keys after the query position before the row maximum. This avoids materializing the full score matrix. The Python path is the CPU oracle; the CUDA source implements the same forward recurrence and remains a correctness-first kernel awaiting MSVC compilation and backward support.
