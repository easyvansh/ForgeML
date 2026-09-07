# Architecture and contracts

The runtime has a single storage backend: float64 NumPy arrays. NumPy provides numerical kernels and storage, but ForgeML constructs its own graph and defines every implemented VJP. There is no PyTorch dependency in training.

## Graph lifecycle

`Tensor.op` attaches parents and a VJP when recording is enabled and at least one parent requires gradients. `backward` traverses the graph iteratively, then propagates an ephemeral gradient map in reverse topological order. Contributions from shared subgraphs are added before a node is visited. Leaves accumulate gradients across backward calls; callers clear them before an optimizer update. Intermediate `.grad` retention and higher-order differentiation are not supported.

`no_grad` uses a context-local flag and restores the previous value on exit. It is used for generation and final evaluation. It prevents graph construction, not parameter mutation.

Operations read array values in their backward closures. Parameters must not be updated until the corresponding backward call has completed. There are no mutation version counters or automatic GPU synchronization semantics yet.

## Shapes and VJPs

Broadcast reversal sums leading extra dimensions and dimensions whose original extent was one. For batched matrix products, form dA = dC Bᵀ and dB = Aᵀ dC, then reverse broadcasting. Rank-one matmul is rejected explicitly. Advanced indexing uses scatter-add so repeated token indices accumulate correctly.

Reshape reverses through the original shape. Transpose uses the inverse permutation. Reduction gradients reinsert removed dimensions and broadcast. Maximum splits gradient equally at ties; finite-difference tests avoid nondifferentiable boundaries. GELU uses the tanh approximation, not the exact Gaussian CDF.

## Model

Token and learned positional embeddings feed a stack of pre-LayerNorm blocks. Each block contains causal multi-head attention, residual addition, a second LayerNorm, a 4× width GELU MLP and a second residual. Final LayerNorm and an untied linear vocabulary head produce logits. Attention projections are separate Q/K/V linear layers with biases.

Softmax subtracts a detached row maximum. Cross entropy computes log-sum-exp on shifted logits and gathers target scores directly, avoiding a log of a potentially underflowed probability. Every causal row includes its diagonal, so this model never constructs a fully masked row.

## Optimizers and persistence

SGD implements classical momentum without dampening or Nesterov. Adam has coupled weight decay; AdamW applies multiplicative decoupled decay. Adam uses per-parameter update counters so missing gradients do not advance that parameter's bias correction.

`weights.npz` stores named arrays without pickle. The adjacent model configuration reconstructs architecture. These files are inference checkpoints only. Exact resume will require optimizer moments, counters, sampler position, RNG state, schedule state and an atomic checkpoint manifest.
