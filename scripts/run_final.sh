#!/usr/bin/env bash
# Final experiment driver (meant to run inside: screen -S test).
# Runs both config spaces (full + fixed_2opt) = 720 cells, then post-processes and plots
# each variant into its own subdir. Logs to results/final_run.log and drops a sentinel
# file results/FINAL_COMPLETE when finished. No `set -e`: a single failed cell must not
# abort the whole sweep (xargs returns nonzero if any cell fails).
set -u
REPO=/local/anmol/aac_tsp_overtuning
cd "$REPO"
PY="$REPO/.venv/bin/python"
LOG="$REPO/results/final_run.log"
JOBS=48

rm -f "$REPO/results/FINAL_COMPLETE"
echo "=== FINAL START $(date) (jobs=$JOBS) ===" | tee -a "$LOG"

# Fresh per-variant output dirs (pilot artifacts in results/raw/*.csv are left untouched).
rm -rf "$REPO"/results/raw/full "$REPO"/results/raw/fixed_2opt "$REPO"/results/raw_verify
mkdir -p results/processed/full results/processed/fixed_2opt \
         results/summaries/full results/summaries/fixed_2opt \
         figures/full figures/fixed_2opt

# 1. Generate the 720-run list (each cell routed to results/raw/<config_space>).
$PY scripts/create_experiments.py --profile final --out_dir "$REPO/results/raw" 2>&1 | tee -a "$LOG"
NRUN=$(tail -n +3 scripts/run_experiments.sh | wc -l)
echo "launching $NRUN cells $(date)" | tee -a "$LOG"

# 2. Run all cells in parallel (GNU parallel is absent on this VM, so xargs -P).
SECONDS=0
tail -n +3 scripts/run_experiments.sh | xargs -P "$JOBS" -I{} bash -c '{}' >> "$LOG" 2>&1
echo "all cells finished in ${SECONDS}s $(date)" | tee -a "$LOG"
echo "raw csv counts: full=$(ls results/raw/full/*.csv 2>/dev/null | wc -l) fixed_2opt=$(ls results/raw/fixed_2opt/*.csv 2>/dev/null | wc -l)" | tee -a "$LOG"

# 3. Post-process + plot each config space into its own dirs.
for CS in full fixed_2opt; do
  echo "=== postprocess/plot $CS $(date) ===" | tee -a "$LOG"
  $PY scripts/postprocess.py --raw_dir "results/raw/$CS" \
      --processed_dir "results/processed/$CS" --summaries_dir "results/summaries/$CS" 2>&1 | tee -a "$LOG"
  $PY scripts/make_plots.py --raw_dir "results/raw/$CS" \
      --processed_dir "results/processed/$CS" --summaries_dir "results/summaries/$CS" \
      --fig_dir "figures/$CS" 2>&1 | tee -a "$LOG"
done

echo "=== FINAL DONE $(date) ===" | tee -a "$LOG"
touch "$REPO/results/FINAL_COMPLETE"
