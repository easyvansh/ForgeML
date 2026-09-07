# Literature packet

These are the eight core papers that motivate ForgeML. The repository stores citations and summaries; the authoritative full text remains at the linked arXiv/DOI pages. Do not copy papers into the repository unless their license and distribution terms permit it.

1. Baydin, Pearlmutter, Radul, Siskind. *Automatic Differentiation in Machine Learning: a Survey* (2018). [arXiv:1502.05767](https://arxiv.org/abs/1502.05767). Runtime graph and reverse-mode VJP foundation.
2. Vaswani et al. *Attention Is All You Need* (2017). [arXiv:1706.03762](https://arxiv.org/abs/1706.03762). Decoder attention workload and Transformer architecture.
3. Kingma, Ba. *Adam: A Method for Stochastic Optimization* (2014/2015). [arXiv:1412.6980](https://arxiv.org/abs/1412.6980). Adaptive optimizer baseline.
4. Loshchilov, Hutter. *Decoupled Weight Decay Regularization* (2017/2019). [arXiv:1711.05101](https://arxiv.org/abs/1711.05101). Adam versus AdamW experiment.
5. Dao et al. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (2022). [arXiv:2205.14135](https://arxiv.org/abs/2205.14135). Online softmax and tiled memory movement.
6. Dao. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning* (2023). [arXiv:2307.08691](https://arxiv.org/abs/2307.08691). Future warp/block scheduling analysis.
7. Kaplan et al. *Scaling Laws for Neural Language Models* (2020). [arXiv:2001.08361](https://arxiv.org/abs/2001.08361). Loss and compute scaling hypotheses.
8. Hoffmann et al. *Training Compute-Optimal Large Language Models* (2022). [arXiv:2203.15556](https://arxiv.org/abs/2203.15556). Parameter/token allocation hypothesis.

The bibliography is also available in `paper/references.bib`, with the current project interpretation in `paper/main.md`. The browser-verified abstracts describe FlashAttention as reducing HBM reads/writes through tiling, AdamW as decoupling weight decay from adaptive updates, and the scaling papers as empirical hypotheses. ForgeML claims only to test analogous behavior at small scale.
