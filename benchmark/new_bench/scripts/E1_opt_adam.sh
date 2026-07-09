#!/bin/bash
#SBATCH --job-name=E1_opt_adam
#SBATCH --partition=cpulong

# E1-opt sweep for Adam (unconstrained reference): full dataset as train, no val/test split.
#   Run from benchmark/new_bench/:   bash scripts/E1_opt_adam.sh
#   Local smoke test:   LAUNCHER=local INIT_SEEDS=0 bash scripts/E1_opt_adam.sh
set -euo pipefail
source scripts/_env.sh

ALGO=adam
# init_seed loops OUTSIDE the multirun (one -m per seed): each seed re-runs the SAME
# grid, so select_best matches configs across seeds by hyperparameter signature. The
# FULL grid (lr included) is swept INSIDE each -m, so hydra.job.num is unique per config
# and runs never overwrite each other. The launcher blocks per -m (chunks run serially).
for s in ${INIT_SEEDS}; do
  python3 -u run.py -m ${LAUNCHER_ARG} \
    +sweep=${ALGO} data=${DATA} task=${TASK} \
    approach=opt init_seed=${s}
done
