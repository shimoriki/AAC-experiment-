#!/usr/bin/env bash
# One-time environment setup on the VM (run on the login/head node, NOT via sbatch):
#   bash slurm/setup_env.sh
# Creates/updates .venv, installs the v2 requirements (incl. SMAC3), and installs
# the package in editable mode. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d .venv ] || [ ! -x .venv/bin/python ]; then
    # A Windows-created .venv (Scripts/ instead of bin/) must be rebuilt on Linux.
    if [ -d .venv ] && [ ! -e .venv/bin ]; then
        echo "Existing .venv is not a Linux venv - recreating as .venv"
        rm -rf .venv
    fi
    python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install -U pip

# pyrfr (SMAC's random forest) compiles from source and needs swig + a C++
# compiler. No sudo required for swig: PyPI ships a prebuilt binary wheel that
# lands on the venv's PATH, where pyrfr's build picks it up.
if ! command -v swig >/dev/null 2>&1; then
    echo "swig not found on the system - installing the prebuilt PyPI wheel into the venv"
    pip install swig
fi
if ! command -v g++ >/dev/null 2>&1 && ! command -v c++ >/dev/null 2>&1; then
    echo "ERROR: no C++ compiler (g++) found - pyrfr cannot build."
    echo "Ask the VM admin for build-essential, or use a conda env (conda install gxx swig)."
    exit 1
fi

pip install -r requirements_v2.txt
pip install -e .

python - <<'EOF'
import numba, optuna
print(f"numba {numba.__version__}, optuna {optuna.__version__}")
try:
    import smac, ConfigSpace
    print(f"smac {smac.__version__}, ConfigSpace {ConfigSpace.__version__} - all 4 arms available")
except ImportError as e:
    raise SystemExit(f"SMAC import failed ({e}); install swig/build-essential and re-run") from e
EOF
echo "Environment ready."
