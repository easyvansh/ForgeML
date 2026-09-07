# ForgeML: Toward Studying Compute-Efficient Transformer Training through a Minimal Runtime

**Living technical report, version 0.1 — September 7, 2026**

**Status:** implemented CPU baseline and research protocol. GPU experiments and language-model scaling studies have not been conducted. This is an independent project report, not a peer-reviewed publication.

## Abstract

We present the first executable baseline of ForgeML, a minimal runtime intended to study interactions among Transformer model size, token allocation, context length and attention implementation under limited compute. The current system implements float64 tensor operations, reverse-mode automatic differentiation, neural-network primitives, three optimizers and a decoder-only Transformer using NumPy storage. It also provides a forward-only streaming causal attention reference. Twenty-one automated test methods pass, while two optional PyTorch comparison methods remain unexecuted because the dependency is absent. A 3,832-parameter model reduces loss on a synthetic periodic sequence from 2.0855 to 0.0031 after 120 updates. Streaming attention agrees with dense attention to a maximum absolute error below 5.56×10⁻¹⁶ in the recorded CPU pilot. These results establish a narrow correctness baseline; they do not establish GPU efficiency, natural-language generalization or scaling behavior. We specify the experiments and implementation gates required to investigate those questions.

## 1. Introduction

Compute-efficient learning depends both on how a training budget is allocated and on how efficiently a runtime executes the selected workload. ForgeML aims to make attention implementation a controlled variable in small Transformer experiments. The proposed research question is: under a limited budget, how should models trade parameter count, training-token presentations and context, and when does IO-aware attention change the feasible allocation?

This report separates the implemented prerequisite from the proposed research contribution. Rebuilding an entire general-purpose framework is outside scope. The present contribution is a compact, inspectable CPU execution path with explicit gradients, tests and raw run artifacts.

## 2. Background and related work

Reverse-mode automatic differentiation provides the computational-graph foundation [1]. Transformer attention provides the model workload [2]. Adam and decoupled weight decay motivate controlled optimizer implementations [3,4]. FlashAttention motivates the online-normalization reference and future GPU tiling [5], while FlashAttention-2 motivates later work-partitioning analysis [6]. The scaling studies of Kaplan et al. and Hoffmann et al. motivate questions about allocation across parameters and token budgets [7,8]. Their large-scale empirical conclusions are hypotheses to examine in our smaller regime, not constants assumed by this project.

The implementation is not a reproduction of the original encoder-decoder Transformer, FlashAttention CUDA implementation, or Chinchilla. It uses a pre-normalized decoder with learned positions and an approximate GELU MLP.

## 3. Runtime design

Each tensor owns a float64 array and, when required, references to parent tensors and a vector-Jacobian-product function. Backward execution uses iterative topological traversal and accumulates contributions from shared graph paths before propagating them. Only leaf gradients persist. A context-local no-gradient mode prevents graph construction during inference.

The supported operations include arithmetic, rank-2-or-higher matrix products, reductions, reshape, transpose, indexing and elementary nonlinearities. Broadcast gradients sum over expanded axes. Repeated indices use scatter-add, which is essential for embedding updates. Parameters must remain unchanged between forward and backward; mutation version tracking is not implemented.

The model consists of token/position embeddings, causal attention and MLP residual blocks, final LayerNorm and an untied output projection. The dense attention path is differentiable. Cross entropy uses shifted log-sum-exp to remain stable for extreme finite logits. SGD uses classical momentum; Adam and AdamW differ in whether weight decay enters the adaptive gradient or is applied independently.

Weights are saved as named non-pickled arrays. Architecture is reconstructed from the adjacent configuration. Optimizer and RNG state are not saved, so exact resume is not supported.

## 4. Streaming attention reference

For each query tile, the algorithm retains a row maximum m, exponential sum l and unnormalized weighted-value sum A. On each key tile it forms masked scores S, updates the maximum to m′, rescales the previous accumulators by exp(m−m′), and adds the new exponential weights and value products. Division by l yields the final output.

This avoids a full T×T score allocation when tile sizes are fixed, while retaining quadratic arithmetic in sequence length. It does not make total storage constant: inputs and outputs still grow with sequence length. The reference runs in Python/NumPy and is forward-only. It supplies an algorithmic oracle for future CUDA work, not a GPU performance result.

## 5. Methods

The baseline used Python 3.11.4 and NumPy 2.3.5 on Windows. Tests check finite differences, broadcasting, repeated indices, stable losses, causal isolation, selected gradients across all model parameters, optimizer updates and synthetic overfitting. Numerical checks typically use central differences with step 10⁻⁶ and mixed absolute/relative tolerances. Tests at nondifferentiable boundaries are avoided.

The recorded training workload uses vocabulary 8, context 16, width 16, two heads, one block and seed 42. Four periodic sequences supply 64 target positions reused for 120 AdamW updates at learning rate 0.01. Weight decay is zero in this smoke configuration. There is no held-out split. Token presentations count repeated examples and do not measure unique data volume.

The attention pilot uses one batch, two heads, head dimension 16, float64 and contexts 32, 64, 128 and 256. Streaming tile width is 32. Each method receives a correctness/warmup call and seven timing calls. Fixed method ordering and uncontrolled CPU scheduling limit performance inference.

## 6. Results

The test runner discovered 23 methods: 21 passed and two optional reference methods were skipped. These results establish agreement with the tested numerical properties, not exhaustive correctness.

Synthetic loss decreased from 2.085474 to 0.003107. The model contains 3,832 parameters and processed 7,680 target-token presentations. Loading the saved weights and using greedy generation reproduced the periodic sequence. This verifies a minimal learning and checkpoint path.

| Context | Dense median (ms) | Streaming median (ms) |
|---:|---:|---:|
| 32 | 0.1153 | 0.1250 |
| 64 | 0.2433 | 0.3467 |
| 128 | 0.5824 | 0.9792 |
| 256 | 4.0029 | 3.7813 |

Maximum absolute output disagreement was approximately 5.55×10⁻¹⁶. Streaming was slower at the first three contexts and slightly faster in the final pilot cell. The evidence does not support a robust speedup claim. Peak memory and GPU traffic were not measured.

## 7. Proposed research experiments

RQ1 extends correctness validation to PyTorch full-model trajectories and GPU forward/backward. RQ2 compares optimizers using equal hyperparameter-search budgets, shared initialization and batches, predeclared decay policies and independent seeds. RQ3 measures verified CUDA attention variants across context lengths with explicit baseline backend selection, CUDA-event timings, peak memory and profiler counters. RQ4 evaluates parameter/token/context allocation using separate fixed-FLOP and fixed-time studies.

A faster exact kernel does not automatically change the allocation curve under an unchanged algorithmic-FLOP budget. It can change the useful work achievable under a time budget. These estimands must remain separate. The screening approximation C≈6ND is insufficient for matching context-dependent workloads; the final model must account for attention and vocabulary-projection costs.

The corpus and tokenizer will be pinned, with license provenance, document-level splits, deduplication and hashes. Unique tokens and repeated presentations will be recorded separately. Three independent training seeds are the initial target where affordable. Scaling exponents will be fitted only after sufficient independently varied observations exist, with residual analysis and sensitivity checks.

## 8. Limitations and threats to validity

The runtime currently lacks GPU storage, mixed precision, optimized attention backward, a real-corpus pipeline and exact training resume. Float64 CPU behavior cannot establish FP16 numerical stability. Selected-coordinate finite differences do not prove all full-model gradients. The dataset is deterministic and trivial; overfitting it provides no language generalization evidence. The timing pilot is short and fixed-order, and training used one seed. No uncertainty interval or scaling exponent is justified by these observations.

The local machine reports an NVIDIA GPU with 6 GB VRAM, but the CUDA compiler was not available on PATH. GPU compatibility, compilation and profiler access remain development tasks. Large proposed model configurations must be budgeted from measured counts and memory use rather than nominal model names.

## 9. Reproducibility

The source revision for the recorded runs is `7b5fa9764c6333a21a8ba56207e6fd28681b745b`. Exact configuration, data fingerprint, metrics, environment information and inference weights are retained in `results/smoke/`. Raw attention samples are in `results/attention.json`. See `docs/reproduction.md` for commands and `results/FINDINGS.md` for the claim/evidence boundary.

## 10. Conclusion

ForgeML v0.1 provides an executable and tested CPU foundation for the proposed ML systems study. The immediate next contribution is a numerically verified GPU training path and held-out language modeling. Claims about memory efficiency, optimization rankings and compute-optimal learning remain contingent on those future experiments.

## References

1. Baydin et al. Automatic Differentiation in Machine Learning: a Survey. JMLR, 2018. https://arxiv.org/abs/1502.05767
2. Vaswani et al. Attention Is All You Need. 2017. https://arxiv.org/abs/1706.03762
3. Kingma and Ba. Adam: A Method for Stochastic Optimization. ICLR, 2015. https://arxiv.org/abs/1412.6980
4. Loshchilov and Hutter. Decoupled Weight Decay Regularization. ICLR, 2019. https://arxiv.org/abs/1711.05101
5. Dao et al. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. 2022. https://arxiv.org/abs/2205.14135
6. Dao. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. 2023 preprint. https://arxiv.org/abs/2307.08691
7. Kaplan et al. Scaling Laws for Neural Language Models. 2020. https://arxiv.org/abs/2001.08361
8. Hoffmann et al. Training Compute-Optimal Large Language Models. 2022. https://arxiv.org/abs/2203.15556
