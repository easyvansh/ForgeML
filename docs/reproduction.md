# Reproduction guide

All paths below are relative to the repository root. Baseline environment: Python 3.11.4, NumPy 2.3.5, Windows. Runtime requires Python 3.10+ and NumPy 1.24+; other combinations have not been validated here.

```powershell
python -m unittest discover -s tests -v
python -m forge train configs/smoke.json --output runs/reproduction-01
python -m forge generate --checkpoint runs/reproduction-01 --prompt 0,1 --tokens 12
python -m forge benchmark attention --seq 32 64 128 256 --repeats 7 --output runs/attention-01.json
```

Expected generation after the supplied smoke training: `0,1,2,3,4,5,6,7,0,1,2,3,4,5`. Final loss should be low; exact timings are not portable. The fixed seed and full-batch synthetic input make this a deterministic numerical smoke test within a fixed numerical environment. BLAS implementations may introduce small differences.

The original checked-in results are in `results/smoke` and `results/attention.json`. Their `git_revision` identifies the code commit used. Keep them unchanged and create fresh run directories for repeats. Training refuses existing output directories. Benchmark output currently overwrites its named file, so choose a new filename when preserving evidence.

Optional PyTorch validation in a dedicated virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[reference]"
.venv\Scripts\python -m unittest discover -s tests -v
```

The optional tests cover batched matmul/softmax gradients and repeated optimizer updates. Full-model PyTorch trajectory parity remains future work. Do not describe skipped tests as passed.

The research report is readable as `paper/main.md`. `paper/main.tex` and `paper/references.bib` provide a LaTeX manuscript. If a TeX distribution is installed, run `pdflatex main`, `bibtex main`, then `pdflatex main` twice from `paper/`. Compilation has not been validated in the current environment.
