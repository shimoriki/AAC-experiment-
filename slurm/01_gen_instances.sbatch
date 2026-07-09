#!/bin/bash
#SBATCH --job-name=aac_v2_gen
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00

# Generates the TSP instance pools + best-known references if (and only if) they
# are missing. The v2 experiment reuses the v1 dataset unchanged (n_cities=50,
# 150 train + 100 test per family), so on the existing VM this is usually a no-op.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate

DATA_DIR="${DATA_DIR:-/local/anmol/datasets/tsp}"
N_CITIES="${N_CITIES:-50}"

if [ -f "$DATA_DIR/best_known/best_known.csv" ] && \
   [ -f "$DATA_DIR/instances/mixed_${N_CITIES}_train.npz" ]; then
    echo "Dataset already present at $DATA_DIR - nothing to do."
    exit 0
fi

echo "Generating instances into $DATA_DIR ..."
python scripts/generate_instances.py --n_cities "$N_CITIES" --n_train_pool 150 \
    --n_test 100 --data_dir "$DATA_DIR"
python scripts/compute_best_known.py --n_cities "$N_CITIES" --n_starts 20 \
    --data_dir "$DATA_DIR"
echo "Done."
