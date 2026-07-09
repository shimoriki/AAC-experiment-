#!/usr/bin/env bash
# Run the whole v2 experiment WITHOUT SLURM: plain xargs fan-out on one machine.
# Equivalent to submit_all.sh (per-profile output roots, idempotent, postprocess
# at the end) for VMs where SLURM is not installed.
#
# Usage (from anywhere; run inside screen/tmux, it blocks until done):
#   PROFILE=bovsrs_smoke bash slurm/run_local.sh    # 8-run smoke test first
#   bash slurm/run_local.sh                         # full 200-run experiment
#   # detached:  screen -dmS aacv2 bash slurm/run_local.sh ; screen -r aacv2
#
# Env overrides:
#   PROFILE   experiment profile (default bovsrs)
#   JOBS      parallel workers (default: all cores)
#   DATA_DIR  instance dataset; auto-detected, generated if missing
set -euo pipefail
cd "$(dirname "$0")/.."

PROFILE="${PROFILE:-bovsrs}"
JOBS="${JOBS:-$(nproc)}"
if [ -z "${DATA_DIR:-}" ]; then
    if [ -f /local/anmol/datasets/tsp/best_known/best_known.csv ]; then
        DATA_DIR=/local/anmol/datasets/tsp          # reuse the v1 dataset if readable
    else
        DATA_DIR="/local/$USER/datasets/tsp"
    fi
fi

if [ "$PROFILE" = "bovsrs" ]; then
    RESULTS_ROOT="results/v2"
    FIG_DIR="figures/v2"
else
    RESULTS_ROOT="results/v2_${PROFILE}"
    FIG_DIR="figures/v2_${PROFILE}"
fi

source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1

if [ ! -f "$DATA_DIR/best_known/best_known.csv" ]; then
    echo "Dataset not found - generating into $DATA_DIR (takes a few minutes)..."
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
    echo "WARNING: some runs failed - check output above. Re-running this script"
    echo "         retries only the failed/missing runs."
fi

python scripts/postprocess_v2.py --raw_dir "$RESULTS_ROOT/raw" \
    --processed_dir "$RESULTS_ROOT/processed" --summaries_dir "$RESULTS_ROOT/summaries"
python scripts/make_plots_v2.py --raw_dir "$RESULTS_ROOT/raw" \
    --processed_dir "$RESULTS_ROOT/processed" --summaries_dir "$RESULTS_ROOT/summaries" \
    --fig_dir "$FIG_DIR"
date > "$RESULTS_ROOT/V2_COMPLETE"
echo "DONE - figures in $FIG_DIR/, tables in $RESULTS_ROOT/summaries/"
