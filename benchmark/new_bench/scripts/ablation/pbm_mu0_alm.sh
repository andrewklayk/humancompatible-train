#!/bin/bash
#SBATCH --job-name=ablation_pbm_mu0
#SBATCH --partition=cpulong

# E1-opt sweep for PBM: full dataset as train, no val/test split (approach=opt).
# Only the optimization variance axis (init_seed) is swept; no fold loop needed.
set -euo pipefail
source scripts/_env.sh

ALGO=pbm_mu0_alm
# init_seed loops OUTSIDE the multirun (one -m per seed): each seed re-runs the SAME
# grid, so select_best matches configs across seeds by hyperparameter signature. The
# FULL grid (lr included) is swept INSIDE each -m, so hydra.job.num is unique per config
# and runs never overwrite each other. The launcher blocks per -m (chunks run serially).
for s in ${INIT_SEEDS}; do
  python3 -u run.py -m ${LAUNCHER_ARG} \
    +sweep=${ALGO} data=${DATA} task=${TASK} \
    approach=opt init_seed=${s} n_epochs=${N_EPOCHS}
done
