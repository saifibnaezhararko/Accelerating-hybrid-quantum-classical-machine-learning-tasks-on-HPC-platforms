import sys
from pathlib import Path

# Make src/ importable when run as `python scripts/train_mc1_spider.py` without an
# editable install (`poetry install` / `pip install -e .`). No-op once installed.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from qnlp_hpc.mc1_spider.experiment import run_experiment


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    main()
