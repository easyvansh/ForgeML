# Research proposal

## Thesis

The central contribution is an experimentally controlled runtime that makes attention implementation an explicit variable in small-language-model budget allocation. The framework exists to support this question; API breadth is secondary.

## Questions and falsifiable hypotheses

| Question | Hypothesis | Evidence required |
|---|---|---|
| RQ1: correctness | VJPs and updates agree with numerical and independent references within declared tolerances | Operator, optimizer, whole-model comparisons and short matched trajectories |
| RQ2: optimization | Optimizer choice affects the best held-out loss achievable under an equal tuning budget | Equal-search SGD/Adam/AdamW experiment, independent seeds |
| RQ3: systems | Avoiding a materialized score matrix lowers memory demand and may improve long-context training throughput | CUDA forward/backward comparisons, peak memory, profiler counters |
| RQ4: allocation | The best parameter/token/context combination depends on whether the budget is FLOPs or elapsed time | Separate matched-FLOP and matched-time studies |

No hypothesis is a promised positive result. CPU timing does not test GPU IO behavior.

## Critical distinction: two budgets

For **fixed algorithmic FLOPs**, a faster exact kernel principally changes elapsed time, not the underlying allocation curve when model, tokens, context and numerics stay fixed. For **fixed wall-clock time**, improved throughput can permit more training tokens or a different feasible model/context combination. Report these experiments separately.

The proxy C ≈ 6ND omits important context-dependent attention and vocabulary costs, particularly for small models. Use it only for initial screening. Derive/count dense projections, attention score/value products, MLPs and output projection for the actual shapes, and label all FLOP estimates. CUDA profiling is not an exact accounting of algorithmic FLOPs.

## Controlled variables

Pin corpus version, document split, tokenizer hash, token order, architecture conventions, dtype, batch token count, optimizer update count, schedule, evaluation token count, seed, hardware and code revision. Report embedding and total parameter counts separately. A vocabulary of 16K and width 128 already consumes roughly 2M parameters in one embedding table, so the original 1M model target requires a revised vocabulary or width. Never assign model names from approximate counts without measuring them.

Define D as token presentations and U as unique corpus tokens. Reusing a 10M-token corpus for ten epochs is D=100M, not U=100M. Record both to distinguish repetition from additional data.

## Initial study design

First build a reliable 1–3M-parameter real-corpus baseline after GPU parity. Use a compact pilot to estimate cost before selecting the final grid. Main studies should use 3 seeds where affordable; make single-seed pilots explicit. Report failed and out-of-memory cells rather than dropping them silently.

Optimizer study: identical initial weights and batches per seed, equal numbers of learning-rate candidates, predeclared weight-decay settings and exclusions, common selection split, and one untouched final evaluation split. Include Adam with coupled L2 versus AdamW with decoupled decay to isolate the distinction.

Systems study: contexts 128–4096 as feasible, fixed B/H/head dimension per sweep, then an end-to-end sweep at fixed tokens per update. Benchmark both a forced mathematical attention baseline and explicitly identified optimized SDPA backends. Verify the backend actually selected.

Allocation study: choose three model sizes and three affordable compute levels after pilots. Match token budgets using the actual cost model. Separately repeat selected cells with a fixed training-time budget. Freeze validation windows and document how shorter-context evaluation is made comparable.

## Definition of completion

Completion requires a verified GPU training path, real held-out data, repeatable experiments, uncertainty estimates and an evidence-backed report. Version 0.1 is the implemented CPU prerequisite, not the completed research project.
