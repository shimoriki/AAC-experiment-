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
    if ! python3 -m venv .venv; then
        echo "python3 -m venv is unavailable; trying rootless fallbacks."
        rm -rf .venv
        if command -v uv >/dev/null 2>&1; then
            uv venv --python python3 .venv
        elif python3 -m pip --version >/dev/null 2>&1; then
            python3 -m pip install --user --upgrade virtualenv
            python3 -m virtualenv .venv
        else
            echo "ERROR: could not create .venv without administrator access."
            echo "Install uv in your home directory, then retry:"
            echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
            echo "  export PATH=\"$HOME/.local/bin:$HOME/.cargo/bin:$PATH\""
            exit 1
        fi
    fi
fi
source .venv/bin/activate
python -m pip install -U pip

# pyrfr (SMAC's random forest) compiles from source and needs a *working* SWIG
# executable plus a C++ compiler. Install SWIG in the venv so setup does not need
# root. If the wheel's Python console launcher is broken, bypass it with a tiny
# wrapper around the native executable shipped in the same wheel.
if command -v swig >/dev/null 2>&1 && ! swig -version >/dev/null 2>&1; then
    BROKEN_SWIG="$(command -v swig)"
    echo "Broken SWIG launcher detected at $BROKEN_SWIG; removing venv package."
    python -m pip uninstall -y swig >/dev/null 2>&1 || true
    case "$BROKEN_SWIG" in
        "$VIRTUAL_ENV"/bin/*) rm -f -- "$BROKEN_SWIG" ;;
    esac
    hash -r
fi
if ! command -v swig >/dev/null 2>&1 || ! swig -version >/dev/null 2>&1; then
    echo "No working system SWIG found; installing SWIG 4.4.1 in .venv (no sudo)."
    python -m pip install --no-cache-dir --force-reinstall "swig==4.4.1"
    hash -r
fi

# The wheel normally creates a Python console launcher. It can pass `swig
# -version` here but still fail inside pip's isolated PEP 517 build environment,
# where the wheel's Python module is hidden. Whenever SWIG comes from this venv,
# replace that launcher unconditionally with a native-binary wrapper.
case "$(command -v swig 2>/dev/null || true)" in
    "$VIRTUAL_ENV"/bin/*)
        SITE_PACKAGES="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
        SWIG_NATIVE="$SITE_PACKAGES/swig/data/bin/swig"
        SWIG_SHARE_ROOT="$SITE_PACKAGES/swig/data/share/swig"
        SWIG_SHARE="$(find "$SWIG_SHARE_ROOT" -mindepth 1 -maxdepth 1 -type d -print -quit 2>/dev/null || true)"
        if [ ! -x "$SWIG_NATIVE" ] || [ -z "$SWIG_SHARE" ]; then
            echo "ERROR: the SWIG wheel did not contain its expected native files."
            exit 1
        fi
        rm -f -- "$VIRTUAL_ENV/bin/swig" "$VIRTUAL_ENV/bin/swig4.0"
        printf '#!/usr/bin/env bash\nexport SWIG_LIB=%q\nexec %q "$@"\n' \
            "$SWIG_SHARE" "$SWIG_NATIVE" > "$VIRTUAL_ENV/bin/swig"
        chmod +x "$VIRTUAL_ENV/bin/swig"
        ln -s swig "$VIRTUAL_ENV/bin/swig4.0"
        hash -r
        ;;
esac
if ! swig -version >/dev/null 2>&1; then
    echo "ERROR: the rootless SWIG installation failed."
    echo "If conda is available, try: conda install -c conda-forge swig gxx_linux-64"
    exit 1
fi
swig -version | sed -n '1,3p'
if ! command -v g++ >/dev/null 2>&1 && ! command -v c++ >/dev/null 2>&1; then
    echo "ERROR: no C++ compiler (g++) found - pyrfr cannot build."
    echo "Ask the VM admin for build-essential, or use a conda env (conda install gxx swig)."
    exit 1
fi

# Install the base stack first so a later native SMAC build error does not also
# leave simple packages such as PyYAML missing. Then install the v2 additions.
python -m pip install -r requirements.txt
python -m pip install "ConfigSpace>=1.1" "smac>=2.2,<2.4"
python -m pip install -e .

python - <<'EOF'
from importlib.metadata import version

import numba, optuna
print(f"numba {version('numba')}, optuna {version('optuna')}")
try:
    import smac, ConfigSpace
    print(f"smac {version('smac')}, ConfigSpace {version('ConfigSpace')} - all 4 arms available")
except ImportError as e:
    raise SystemExit(f"SMAC import failed ({e}); install swig/build-essential and re-run") from e
EOF
echo "Environment ready."
