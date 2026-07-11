"""Smoke test — package imports and exposes a version."""

import qnlp_hpc


def test_version() -> None:
    assert qnlp_hpc.__version__ == "0.1.0"
