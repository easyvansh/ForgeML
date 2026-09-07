# ForgeML implementation audit

Updated 2026-09-07. This document is the authoritative status report. A checked item means code and a local verification artifact exist; it does not mean the final research claim is established.

## Done

- CPU tensor storage, broadcasting, indexing, reductions, reshape and transpose.
- Reverse-mode autodiff with gradient accumulation, finite differences, stable cross entropy and causal masking.
- Linear, embedding, LayerNorm, GELU, dropout primitive, decoder Transformer and three optimizers.
- Synthetic smoke training and local character-text training with deterministic held-out validation and corpus hashes.
- Dense and online-softmax streaming CPU attention references with numerical comparisons.
- CLI, configs, raw JSONL metrics, checkpoints, findings, engineering log, reproduction guide and paper draft.
- CUDA toolkit discovery: RTX 3060, `nvcc` 13.3, driver CUDA compatibility 12.6.
- CUDA-enabled PyTorch 2.14.0+cu126 in `.venv`; `torch.cuda.is_available()` is true on the RTX 3060 Laptop GPU.
- PyTorch gradient and optimizer parity: 25 tests pass, including both formerly optional reference tests.
- Reproducible PyTorch CUDA attention baseline at contexts 128, 256 and 512 with CUDA-event timings and memory artifacts.
- Custom CUDA causal forward kernel executes at contexts 128, 256, and 512 with maximum CPU-reference error `2.98e-7`; raw JSONL output is checked in.

## In progress

- Custom CUDA forward milestone is implemented as a standalone correctness-first kernel; it still needs binding into the training runtime.

## Remaining implementation

- Add device-aware Tensor storage and C++/CUDA extension boundaries.
- Bind the verified attention forward kernel into the training runtime and add a backward kernel.
- Implement and test CUDA vector/reduction/softmax/normalization primitives.
- Implement attention forward and backward kernels; validate against CPU and PyTorch at non-multiple tile sizes.
- Add CUDA-event timing, peak allocation tracking, profiler metadata and warmup protocols.
- Add exact resume state: optimizer moments, counters, RNG, scheduler and sampler position.
- Add a licensed, pinned corpus and tokenizer; the checked-in character corpus is only a plumbing smoke test.
- Run three-seed optimizer, context and parameter/token allocation studies.
- Fit scaling relationships only after enough independent observations and uncertainty analysis.
- Replace preliminary paper findings with measured GPU and held-out language-model results.

## Current evidence boundary

The project does not yet support claims of CUDA speedup, GPU memory reduction, natural-language quality, compute-optimal allocation or scaling exponents. Those claims require the remaining experiments. The local CPU result is a correctness and plumbing baseline.
