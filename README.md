# Accelerating hybrid quantum-classical ML on HPC

Topic-aware QNLP sentence classifier (programming vs cooking), accelerated from a
>1hr/6-word-sentence CPU prototype toward GPU-based quantum simulation on HPC.

- **Role:** Software Development Lead — dependency mgmt, backend integration, CI/CD, data pipeline.
- **Docs:** project guide + progress log in [`CLAUDE.md`](CLAUDE.md); layout in [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md).

## Quick start

```bash
poetry install --with dev          # CPU baseline + tooling
poetry run pytest                  # smoke test
poetry run pre-commit install      # enable git hooks
```

On the HPC node (CUDA 12.4–12.6, NVIDIA driver 550+):

```bash
poetry install --with dev,gpu      # add qiskit-aer-gpu + cuQuantum
```

## Stack

Python 3.11 · lambeq 0.5.0 · Qiskit 2.x · Qiskit Aer (CPU) / Aer-GPU + cuQuantum (HPC) · pytket.
See `CLAUDE.md` §7 for full version matrix.
