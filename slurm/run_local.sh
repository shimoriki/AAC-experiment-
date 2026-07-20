#!/usr/bin/env bash
# Run the whole v2 experiment WITHOUT SLURM: plain xargs fan-out on one machine.
# Equivalent to submit_all.sh (per-profile output roots, idempotent, postprocess
# at the end) for VMs where SLURM is not installed.
#
# Usage (from anywhere; run inside screen/tmux, it blocks until done):
#   PROFILE=bovsrs_resampling_smoke bash slurm/run_local.sh  # 16-run smoke first
#   PROFILE=bovsrs_resampling_signal JOBS=4 bash slurm/run_local.sh # 192-run main
#   PROFILE=bovsrs_resampling_train_sizes_smoke JOBS=2 bash slurm/run_local.sh # 48
#   PROFILE=bovsrs_resampling_train_sizes JOBS=2 bash slurm/run_local.sh # 576
#   bash slurm/run_local.sh                         # same 192-run factorial benchmark
#   # detached:  screen -dmS aacv2 bash slurm/run_local.sh ; screen -r aacv2
#
# Env overrides:
#   PROFILE   experiment profile (default bovsrs_resampling_signal)
#   JOBS      parallel workers (default: 2; conservative for SMAC-AAC)
#   DATA_DIR  instance dataset (default: /local/rohit/datasets/tsp)
set -euo pipefail
cd "$(dirname "$0")/.."

PROFILE="${PROFILE:-bovsrs_resampling_signal}"
JOBS="${JOBS:-2}"
DATA_DIR="${DATA_DIR:-/local/rohit/datasets/tsp}"
export DATA_DIR

if [ "$PROFILE" = "bovsrs" ]; then
    RESULTS_ROOT="results/v2"
    FIG_DIR="figures/v2"
else
    RESULTS_ROOT="results/v2_${PROFILE}"
    FIG_DIR="figures/v2_${PROFILE}"
fi

# First-run bootstrap. A fresh clone intentionally has no committed .venv.
if [ ! -x .venv/bin/python ]; then
    echo "Linux virtualenv not found; running one-time v2 environment setup..."
    bash slurm/setup_env.sh
fi
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1
python scripts/preflight_v2.py
mkdir -p "$RESULTS_ROOT/logs" "$RESULTS_ROOT/failures"
rm -f "$RESULTS_ROOT/V2_COMPLETE"

DATASET_COMPLETE=1
for FAMILY in uniform clustered mixed; do
    for ROLE in train test; do
        [ -r "$DATA_DIR/instances/${FAMILY}_50_${ROLE}.npz" ] || DATASET_COMPLETE=0
    done
done
BEST_KNOWN="$DATA_DIR/best_known/best_known.csv"
[ -r "$BEST_KNOWN" ] || DATASET_COMPLETE=0
if [ -r "$BEST_KNOWN" ] && [ "$(wc -l < "$BEST_KNOWN")" -lt 751 ]; then
    DATASET_COMPLETE=0
fi
if [ "$DATASET_COMPLETE" -ne 1 ]; then
    echo "Dataset missing or incomplete - generating into $DATA_DIR (takes a few minutes)..."
    python scripts/generate_instances.py --n_cities 50 --n_train_pool 150 \
        --n_test 100 --data_dir "$DATA_DIR"
    python scripts/compute_best_known.py --n_cities 50 --n_starts 20 \
        --data_dir "$DATA_DIR"
fi

python scripts/create_experiments_v2.py --profile "$PROFILE" --data_dir "$DATA_DIR" \
    --out_dir "$RESULTS_ROOT/raw" --smac_scratch "$RESULTS_ROOT/smac_output"

N=$(wc -l < slurm/experiments_v2.txt)
echo "Running $N runs with $JOBS parallel workers (profile=$PROFILE, data=$DATA_DIR)"

# Completed runs are skipped by main_v2.py, so re-running this script after a
# crash/interrupt only executes what is missing.
if ! xargs -a slurm/experiments_v2.txt -d '\n' -P "$JOBS" -I{} \
        bash -c "python scripts/main_v2.py {}"; then
    echo "WARNING: at least one run process failed. Inspect per-run logs and FAILED.json."
fi

# Never publish a completion sentinel from a partial run.  This also catches a
# process that exited cleanly without producing its expected sidecar.
python scripts/check_v2_completion.py --manifest slurm/experiments_v2.txt \
    --raw_dir "$RESULTS_ROOT/raw"

python scripts/postprocess_v2.py --raw_dir "$RESULTS_ROOT/raw" \
    --processed_dir "$RESULTS_ROOT/processed" --summaries_dir "$RESULTS_ROOT/summaries"
python scripts/make_plots_v2.py --raw_dir "$RESULTS_ROOT/raw" \
    --processed_dir "$RESULTS_ROOT/processed" --summaries_dir "$RESULTS_ROOT/summaries" \
    --fig_dir "$FIG_DIR"
date > "$RESULTS_ROOT/V2_COMPLETE"
echo "DONE - figures in $FIG_DIR/, tables in $RESULTS_ROOT/summaries/"
echo "Per-run logs: $RESULTS_ROOT/logs/"
