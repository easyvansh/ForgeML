# ForgeML

**A minimal Transformer runtime for studying compute-efficient learning.**

Status: **v0.1 CPU correctness baseline**. The long-term objective is a GPU-native training runtime; this release uses NumPy storage and independently implemented reverse-mode autodiff. It does not yet run training on the GPU.

Research question: under a limited budget, how should small Transformer models trade parameters, training tokens, and context, and when does IO-aware attention change the achievable allocation?

## What works today

- Float64 tensors, broadcasting, reductions, rank-2-or-higher batched matrix multiplication, indexing, and reverse-mode gradients.
- Linear layers, embeddings, LayerNorm, approximate GELU, dropout primitive, and stable cross entropy.
- A pre-normalized decoder-only Transformer with learned positional embeddings and causal multi-head attention.
- SGD with momentum, Adam, and AdamW.
- Configured synthetic training, JSONL metrics, weight checkpoints, and greedy integer-token generation.
- Local character language-model training with deterministic held-out validation, corpus hashing, and metadata.
- Dense and tiled online-softmax attention **CPU forward references**.
- Finite-difference checks, optional PyTorch comparisons, causal tests, and an end-to-end overfit test.

## Quick start

Run these commands from this repository. Python 3.10+ and NumPy are required.

```powershell
python -m unittest discover -s tests -v
python -m forge train configs/smoke.json --output runs/my-first-run
python -m forge train configs/char_smoke.json --output runs/char-smoke
python -m forge generate --checkpoint runs/my-first-run --prompt 0,1 --tokens 16
python -m forge benchmark attention --seq 32 64 128 256 --output runs/attention.json
```

The current computer already has NumPy. For a separate environment:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e .
.venv\Scripts\python -m forge train configs/smoke.json --output runs/isolated-smoke
```

Installing the package also provides `forge`. `python -m forge` works directly from the checkout without installation. Use a fresh output directory for each training run; existing runs are protected from overwrite.

## Measured results: September 7, 2026

| Check | Observed result |
|---|---|
| Unit/integration tests | 21 passed, 2 optional PyTorch tests skipped |
| Synthetic model | 3,832 parameters, 1 layer, width 16, 2 heads |
| Training loss | 2.085474 → 0.003107 after 120 updates |
| Token presentations | 7,680, reusing the same 64 target positions |
| Checkpoint generation | Correctly continues the periodic 0–7 sequence |
| CPU streaming attention | Max absolute difference ≤ 5.56e-16 at tested contexts |

The character run is a tiny smoke corpus, not evidence of broad language-model quality. Full evidence is in [findings](results/FINDINGS.md), [raw metrics](results/char_smoke/metrics.jsonl), and [attention samples](results/attention.json). Timing is descriptive and local; no GPU speedup is claimed.

## Project map

```text
forge/          Tensor engine, layers, optimizers, attention references, CLI
configs/        Reproducible executable configurations
tests/          Numerical, behavioral, and integration checks
docs/           Architecture, research protocol, roadmap, engineering log
results/        Small checked-in baseline evidence
paper/          Living research report and LaTeX manuscript
```

Start with [the research proposal](docs/research-proposal.md), [architecture](docs/architecture.md), and [reproduction guide](docs/reproduction.md). The [paper](paper/main.md) reports only existing evidence and labels future studies explicitly.

## Next milestones

1. Complete PyTorch full-model and optimizer parity, input validation, FP32 support, and resumable optimizer/RNG state.
2. Add a licensed corpus, pinned tokenizer, document-disjoint splits, and held-out evaluation.
3. Add CUDA storage/dispatch and verified forward/backward primitives; use a library GEMM for the training path while developing educational custom GEMM separately.
4. Develop and profile causal tiled attention forward **and backward**.
5. Run controlled optimizer and matched-budget experiments with independent seeds.

The local GPU reports 6 GB VRAM. CUDA compiler availability must be resolved before GPU development; the CUDA version displayed by `nvidia-smi` does not establish an installed toolkit. See [GPU design](docs/cuda.md).

## Scope and limitations

No CUDA backend, mixed precision, attention backward optimization, real-corpus pipeline, scaling-law fit, or publishable GPU findings are claimed. Weight files support inference, not exact training resume. All tensors currently use float64. Matrix multiplication deliberately requires rank ≥ 2. Gradients are retained on leaves only. Do not modify tensor data between forward and backward. The Transformer currently has no dropout in its blocks, although a tested standalone dropout layer is available.

No remote repository has been created. This is a local Git repository. No distribution license has been selected yet; choose one before publishing.
