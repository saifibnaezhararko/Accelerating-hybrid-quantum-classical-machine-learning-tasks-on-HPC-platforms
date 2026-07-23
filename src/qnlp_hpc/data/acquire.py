"""Dataset acquisition — Kaggle API download into ``data/raw/``.

The ``kaggle`` package is an optional dependency (``poetry install --with data``):
the training pipeline never needs it, and the HPC compute nodes are usually offline.
Import failures are turned into an actionable message rather than a stack trace.

Credentials come from ``~/.kaggle/kaggle.json`` or the ``KAGGLE_USERNAME`` /
``KAGGLE_KEY`` environment variables — the latter is what to use on HPC, since the
home directory is often shared and world-readable.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from qnlp_hpc.paths import RAW_DIR

_INSTALL_HINT = (
    "The 'kaggle' package is not installed. It lives in the optional data group:\n"
    "    poetry install --with data"
)

_AUTH_HINT = (
    "Kaggle authentication failed. Either place kaggle.json at ~/.kaggle/kaggle.json "
    "(chmod 600), or export KAGGLE_USERNAME and KAGGLE_KEY."
)


def _kaggle_api():
    # No return annotation: the type only exists when the optional extra is installed.
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(_INSTALL_HINT) from exc

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:  # pragma: no cover - depends on local credentials
        raise RuntimeError(_AUTH_HINT) from exc
    return api


def download_kaggle_dataset(
    dataset: str,
    destination: Path | None = None,
    force: bool = False,
) -> Path:
    """Download and unzip a Kaggle dataset (``owner/dataset-name``).

    Returns the directory the files were extracted into. Existing downloads are
    reused unless ``force`` is set, so reruns on a metered HPC link are cheap.
    """
    if "/" not in dataset:
        raise ValueError(
            f"Expected a Kaggle reference of the form 'owner/dataset-name', " f"got {dataset!r}."
        )

    target = Path(destination) if destination else RAW_DIR / dataset.split("/")[-1]

    if target.exists() and any(target.iterdir()) and not force:
        print(f"Reusing existing download: {target}")
        return target

    if force and target.exists():
        shutil.rmtree(target)

    target.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Kaggle dataset {dataset} -> {target}")
    _kaggle_api().dataset_download_files(dataset, path=str(target), unzip=True)

    files = sorted(p.name for p in target.rglob("*") if p.is_file())
    print(f"Downloaded {len(files)} file(s): {files}")
    return target
