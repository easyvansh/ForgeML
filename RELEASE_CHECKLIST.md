# Release checklist

Before publishing a tagged ForgeML release:

- Run the full test suite and record optional skips.
- Confirm `git status` is clean and remove generated local runs from version control.
- Check that no secrets, personal data, or unlicensed corpus content are tracked.
- Confirm README results match raw artifacts and label CPU, GPU, pilot, and final evidence separately.
- Record Python, NumPy, PyTorch, CUDA, driver, GPU, and Git revision metadata.
- Build and inspect the package in a clean environment.
- Update `docs/AUDIT.md`, `docs/engineering-log.md`, and `paper/main.md`.
- Tag with a semantic version such as `v0.2.0` only after the preceding checks pass.
