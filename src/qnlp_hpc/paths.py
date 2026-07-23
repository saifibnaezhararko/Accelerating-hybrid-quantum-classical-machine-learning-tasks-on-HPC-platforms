"""Filesystem anchors shared by the data pipeline and the experiment configs.

Everything that writes to or reads from the repository resolves through here, so a
run behaves the same whether it is launched from the repo root, from ``scripts/``,
or from a scheduler working directory on the HPC node.
"""

from __future__ import annotations

import os
from pathlib import Path

_MARKERS = ("pyproject.toml", ".git")


def find_repo_root(start: Path | None = None) -> Path:
    """Return the repository root.

    Resolution order:

    1. ``QNLP_HPC_ROOT`` if set — the escape hatch for HPC batch jobs where the
       package is installed outside the checkout.
    2. The nearest ancestor of ``start`` (default: this file) containing
       ``pyproject.toml`` or ``.git``.
    3. The current working directory, as a last resort for installed wheels.
    """
    override = os.environ.get("QNLP_HPC_ROOT")
    if override:
        return Path(override).expanduser().resolve()

    origin = (start or Path(__file__)).resolve()
    for candidate in (origin, *origin.parents):
        if candidate.is_dir() and any((candidate / marker).exists() for marker in _MARKERS):
            return candidate

    return Path.cwd().resolve()


def resolve(path: str | os.PathLike[str]) -> Path:
    """Make ``path`` absolute, anchoring relative paths at the repo root."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (find_repo_root() / candidate).resolve()


REPO_ROOT = find_repo_root()
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
CONFIG_DIR = REPO_ROOT / "configs"
OUTPUTS_DIR = REPO_ROOT / "outputs"
