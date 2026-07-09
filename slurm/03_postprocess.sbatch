#!/bin/bash
#SBATCH --job-name=aac_v2_post
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# Aggregates raw v2 runs into trajectories/summaries/statistics and renders all
# comparison figures. Usually submitted with a dependency on the run array
# (submit_all.sh does this), but can be run standalone at any time to look at
# partial results.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate

RESULTS_ROOT="${RESULTS_ROOT:-results/v2}"
FIG_DIR="${FIG_DIR:-figures/v2}"

python scripts/postprocess_v2.py \
    --raw_dir "$RESULTS_ROOT/raw" \
    --processed_dir "$RESULTS_ROOT/processed" \
    --summaries_dir "$RESULTS_ROOT/summaries"

python scripts/make_plots_v2.py \
    --raw_dir "$RESULTS_ROOT/raw" \
    --processed_dir "$RESULTS_ROOT/processed" \
    --summaries_dir "$RESULTS_ROOT/summaries" \
    --fig_dir "$FIG_DIR"

date > "$RESULTS_ROOT/V2_COMPLETE"
echo "Post-processing complete: figures in $FIG_DIR/, tables in $RESULTS_ROOT/summaries/"
