# Findings — baseline v0.1

Date: 2026-09-07. Implementation revision: `7b5fa9764c6333a21a8ba56207e6fd28681b745b`.

## Supported observations

The test suite discovered 23 methods: 21 passed and two optional PyTorch reference tests were skipped. Parameterized subcases cover additional shapes and functions. This is not 23 successful independent reference comparisons.

The configured 3,832-parameter Transformer learned a periodic eight-token sequence. Training loss started at 2.0854742269 and reached 0.0031068468 after 120 updates, with 7,680 token presentations from repeated use of 64 target positions. The reported training command took approximately 0.58 seconds on the active CPU environment; this is a single local timing including logging and saving, not a throughput benchmark.

Reloading saved weights produced `0,1,2,3,4,5,6,7,0,1,2,3,4,5` from the prompt `0,1`. This demonstrates the toy inference path.

## CPU attention pilot

Arrays: B=1, H=2, D=16, float64; streaming tile=32; seven timed calls per method after a correctness/warmup call. Seed 42. Dense always ran first. Values below are observed medians.

| Context | Dense, ms | Streaming, ms | Maximum absolute error |
|---:|---:|---:|---:|
| 32 | 0.1153 | 0.1250 | 5.55e-16 |
| 64 | 0.2433 | 0.3467 | 5.55e-16 |
| 128 | 0.5824 | 0.9792 | 5.55e-16 |
| 256 | 4.0029 | 3.7813 | 5.55e-16 |

The streaming implementation agrees numerically with dense attention in this pilot. It is not consistently faster. The small difference at context 256 must not be promoted as a general speedup: there is one process, fixed method order, seven samples and no control of CPU/BLAS threading or background load.

`dense_score_array_bytes` in the raw file is an analytical estimate for one dense score array, including rows from both methods for comparison. It is **not** measured peak memory, total attention workspace or memory traffic.

## Unsupported claims

There are no measured CUDA results, GPU memory savings, training attention speedups, held-out language-model losses, optimizer ranking, scaling exponents or compute-optimal allocations. No confidence interval for model training is reported because the smoke run uses one seed. The synthetic result cannot be used as evidence for the proposed 1M–50M parameter regime.

## Evidence

- `smoke/config.json`: exact configuration.
- `smoke/data.json`: synthetic data description and fingerprint.
- `smoke/system.json`: software environment and source revision.
- `smoke/metrics.jsonl`: each training update.
- `smoke/summary.json`: final measured summary.
- `smoke/weights.npz`: inference weights.
- `attention.json`: raw timing samples and numerical errors.
- `tests.txt`: captured baseline verification output.
- `generation.txt`: checkpoint-load generation output.
