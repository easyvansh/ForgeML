# Engineering log

## EXP-001 — CPU correctness baseline, 2026-09-07

Problem: the proposed GPU study needs an independent executable numerical reference.

Decision: implement float64 NumPy storage with custom VJPs. Keep rank-one matmul out of the first API and make its rejection explicit. Use iterative graph traversal and leaf-only accumulation.

Evidence: 21 test methods passed, including finite differences for representative operations and selected coordinates across all Transformer parameters. Two optional PyTorch methods skipped because the dependency is absent. See `results/tests.txt`.

Limit: finite differences and a toy overfit do not prove correctness for every shape or precision.

## EXP-002 — Train and reload

Hypothesis: if gradients, causal attention and updates cooperate, a tiny model should learn a repeated token cycle.

Result: 3,832 parameters; 120 updates; loss 2.085474 to 0.003107. Reloaded weights generate the expected cycle. Evidence: `results/smoke/` and `results/generation.txt`.

Interpretation: the training and inference checkpoint paths work on this deterministic smoke workload. No independent generalization claim follows.

## EXP-003 — Streaming attention CPU reference

Hypothesis: online normalization can reproduce dense causal attention without storing the whole score matrix.

Result: maximum absolute difference 5.55e-16 for contexts 32–256. Streaming was slower at 32–128 and slightly faster in the 256 pilot. The fixed-order short CPU benchmark is not sufficient for a robust speedup claim.

Next: implement GPU forward/backward and collect repeated-session timings with profiler evidence.

## ENV-001 — Tooling constraints

The machine reports a 6 GB NVIDIA GPU. CUDA compiler was not found on PATH and PyTorch was not importable. Existing Matplotlib failed to import because its compiled extension was incompatible with NumPy 2.3.5. No global packages were modified. Figures are deferred until an isolated compatible plotting environment is established; raw evidence remains available.
