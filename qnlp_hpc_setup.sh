#!/usr/bin/env bash
set -e

echo "=== System Report ==="
cat /etc/os-release || true
python3 --version || true
gcc --version || true
cmake --version || true
nvcc --version || true
nvidia-smi || true
lscpu || true
free -h || true

echo "=== Create Python Environment ==="
python3.11 -m venv qnlp_env
source qnlp_env/bin/activate

python -m pip install --upgrade pip

echo "=== Install Qiskit ==="
pip install qiskit qiskit-aer-gpu

echo "=== Verify Qiskit Aer GPU ==="
python - <<'PY'
from qiskit_aer import AerSimulator
backend = AerSimulator(device="GPU")
print("Backend:", backend)
PY

echo "=== Install lambeq (optional) ==="
pip install lambeq

echo "=== Verify Bobcat Parser ==="
python - <<'PY'
from lambeq import BobcatParser
print("Bobcat parser available.")
PY

echo "=== Freeze Environment ==="
pip freeze > requirements.txt

echo "Setup complete."
