# Experiment protocol

## Current executable experiment

`configs/smoke.json` trains a periodic integer-token sequence. The same 64 targets are reused for 120 steps. This is designed to catch a broken learning path, not to evaluate generalization. Raw pre-update losses, gradient norms, parameter norms, token presentations and elapsed times are retained. The final summary loss is evaluated after the last update.

## Real-corpus admission gate

Before downloading training data, record source URL, exact revision, license and any attribution or use restrictions. Public access is not equivalent to an open license. Split at document level before tokenization and deduplicate across splits. Pin tokenizer vocabulary and version, hash all prepared artifacts, record filtering, and maintain an untouched evaluation split. No external corpus is bundled in v0.1.

## Run record for future experiments

Store configuration, Git commit and dirty status, dependency versions, CPU/GPU identity, driver/toolkit, dtype, seed, tokenizer/data hashes, parameter counts, unique tokens, token presentations, optimizer configuration, warmup/schedule, loss, validation loss, gradient/update norms, wall-clock time, and estimated algorithmic FLOPs. Preserve failure logs and reasons for excluded runs.

Current metadata covers the CPU smoke subset of this schema. It does not yet provide the entire research schema or an experiment scheduler.

## Statistical plan

Use independent training seeds as the experimental replication unit. Multiple timing samples within one process are not independent model-training replicates. Report all seed values and per-seed results. With three seeds, include the range and explain that interval estimates are unstable; bootstrap intervals are not a cure for a tiny sample.

For kernel timing, randomize method order in the final study, collect repeated sessions, separate warmup and timed iterations, report distributions and synchronization policy. The checked-in pilot uses seven samples in a fixed method order and cannot establish robust performance superiority.

Do not fit L-infinity and exponents from a few confounded points. Scaling fits require enough independently varied N/D cells, an identifiable model, residual inspection and uncertainty analysis. Compare held-out predictions against simpler baselines; report sensitivity to omitted cells and alternative irreducible-loss values. An optimum outside the sampled range is an extrapolation, not a finding.

## Primary final figure

Plot held-out loss against model size and token presentations for a declared budget and context. Use separate panels for fixed-FLOP and fixed-time budgets. Overlay only measured feasible configurations and annotate OOMs. Repeat selected configurations across attention implementations after numerical parity; avoid a decorative expected-results curve.
