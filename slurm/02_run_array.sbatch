#!/bin/bash
#SBATCH --job-name=aac_v2
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:00:00

# One array task = one configurator run, read from slurm/experiments_v2.txt.
# The --array range is supplied at submit time (see submit_all.sh), e.g.:
#   sbatch --array=0-199%64 slurm/02_run_array.sbatch
# The %64 throttle matches the 64-vCPU VM; each task is single-threaded.
#
# Idempotent: main_v2.py skips runs whose sidecar says status=complete, so a
# partially failed array can simply be resubmitted with the same command.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1

MANIFEST="${MANIFEST:-slurm/experiments_v2.txt}"
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
if [ -z "$LINE" ]; then
    echo "No manifest line for array index $SLURM_ARRAY_TASK_ID - nothing to do."
    exit 0
fi

echo "[task $SLURM_ARRAY_TASK_ID] python scripts/main_v2.py $LINE"
# shellcheck disable=SC2086
python scripts/main_v2.py $LINE
