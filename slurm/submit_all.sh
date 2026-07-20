#!/usr/bin/env bash
# Submit the whole v2 experiment as a dependency chain:
#   dataset check -> run array -> postprocess + figures
#
# Usage (from the repo root on the VM):
#   bash slurm/submit_all.sh                 # 192-run configurator x resampling experiment
#   PROFILE=bovsrs_resampling_smoke bash slurm/submit_all.sh # 16-run smoke first
#   PROFILE=bovsrs bash slurm/submit_all.sh  # explicit legacy 200-run experiment
#   PROFILE=bovsrs_smoke bash slurm/submit_all.sh   # legacy 8-run smoke
#
# Env overrides: PROFILE, DATA_DIR, THROTTLE (concurrent array tasks, default 64).
set -euo pipefail
cd "$(dirname "$0")/.."

PROFILE="${PROFILE:-bovsrs_resampling_signal}"
THROTTLE="${THROTTLE:-64}"
DATA_DIR="${DATA_DIR:-/local/rohit/datasets/tsp}"
export DATA_DIR
mkdir -p slurm/logs

# Each profile gets its own output root so smoke-test runs never pollute the
# real experiment's analysis (postprocess reads everything under raw/).
if [ "$PROFILE" = "bovsrs" ]; then
    RESULTS_ROOT="results/v2"
    FIG_DIR="figures/v2"
else
    RESULTS_ROOT="results/v2_${PROFILE}"
    FIG_DIR="figures/v2_${PROFILE}"
fi
mkdir -p "$RESULTS_ROOT/logs" "$RESULTS_ROOT/failures"
rm -f "$RESULTS_ROOT/V2_COMPLETE"

if [ ! -x .venv/bin/python ]; then
    echo "Linux virtualenv not found; running one-time v2 environment setup..."
    bash slurm/setup_env.sh
fi
source .venv/bin/activate
python scripts/preflight_v2.py
python scripts/create_experiments_v2.py --profile "$PROFILE" --data_dir "$DATA_DIR" \
    --out_dir "$RESULTS_ROOT/raw" --smac_scratch "$RESULTS_ROOT/smac_output"

N=$(wc -l < slurm/experiments_v2.txt)
echo "Submitting $N runs (profile=$PROFILE, throttle=$THROTTLE, outputs=$RESULTS_ROOT)"

GEN_ID=$(sbatch --parsable --export=ALL,DATA_DIR="$DATA_DIR" slurm/01_gen_instances.sbatch)
echo "  dataset job:      $GEN_ID"

ARRAY_ID=$(sbatch --parsable --dependency=afterok:"$GEN_ID" \
    --array=0-$((N - 1))%"$THROTTLE" slurm/02_run_array.sbatch)
echo "  run array:        $ARRAY_ID  (0-$((N - 1))%$THROTTLE)"

POST_ID=$(sbatch --parsable --dependency=afterany:"$ARRAY_ID" \
    --export=ALL,RESULTS_ROOT="$RESULTS_ROOT",FIG_DIR="$FIG_DIR" slurm/03_postprocess.sbatch)
echo "  postprocess:      $POST_ID"

echo ""
echo "Monitor with:   squeue -u \$USER    and    tail -f slurm/logs/aac_v2_${ARRAY_ID}_*.out"
echo "Finished when:  $RESULTS_ROOT/V2_COMPLETE exists; figures land in $FIG_DIR/"
echo "Per-run logs:    $RESULTS_ROOT/logs/"
echo "Failures:        find $RESULTS_ROOT -name FAILED.json -print -exec cat {} \;"
echo "If some tasks failed/timed out: re-run this script - completed runs are skipped."
