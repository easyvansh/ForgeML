# Roadmap and completion gates

Time estimates in the original proposal are planning assumptions, not deadlines. CUDA backward and experiment repetition are likely to determine the critical path.

| Milestone | State | Exit condition |
|---|---|---|
| CPU tensor/autograd | Implemented baseline | Current numerical suite passes; extend edge cases as API grows |
| CPU Transformer | Implemented baseline | Tiny overfit and causality checks pass |
| Optimizers | Implemented baseline | Hand checks pass; optional PyTorch parity awaits dependency |
| Reproducible smoke run | Implemented | Config, data fingerprint, metrics and inference weights retained |
| Real language modeling | Planned | Licensed data, tokenizer, split and held-out evaluation verified |
| GPU runtime | Planned | Device storage and full-model backward pass match CPU |
| IO-aware CUDA attention | Planned | Forward/backward correctness plus profiler evidence |
| Optimizer study | Planned | Equal tuning budgets and independent repeats |
| Scaling/allocation study | Planned | Predeclared cells, budgets and uncertainty |
| Final paper | Living draft | All claims tied to reproducible evidence |

## Next concrete engineering tasks

1. Add input/hyperparameter validation and FP32 support with separate numerical tolerances.
2. Extend checkpointing to exact resume and test uninterrupted versus resumed trajectories.
3. Run optional references and extend full-model comparisons.
4. Introduce a document dataset and a fixed tokenizer only after recording provenance.
5. Establish the CUDA toolchain and an explicit backend interface.

## Exclusions

No distributed training, custom compiler, RAG, agents, multimodality, web frontend or billion-parameter models. A CLI remains the primary interface. Additional model families need a research reason before being added.
