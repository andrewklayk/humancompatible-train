#!/bin/bash
#SBATCH --job-name=E1_opt_ghost
#SBATCH --partition=cpulong

# E1-opt sweep for stochastic Ghost SQP: full dataset as train, no val/test split.
#   Run from benchmark/new_bench/:   bash scripts/E1_opt_ghost.sh
#   Local smoke test:   LAUNCHER=local INIT_SEEDS=0 bash scripts/E1_opt_ghost.sh
set -euo pipefail
source scripts/_env.sh

ALGO=ghost
# ~1-2 s/step on Dutch pairwise (m=306) -> ~5-9 h per 40-epoch run, far past submitit's
# 60-min default timeout_min (the slurm_gpu_slow launcher sets none).
: "${TIMEOUT_MIN:=720}"
TIMEOUT_ARG=""
if [ -n "${LAUNCHER_ARG}" ]; then
  TIMEOUT_ARG="hydra.launcher.timeout_min=${TIMEOUT_MIN}"
fi
# init_seed loops OUTSIDE the multirun (one -m per seed): each seed re-runs the SAME
# grid, so select_best matches configs across seeds by hyperparameter signature. The
# FULL grid (lr included) is swept INSIDE each -m, so hydra.job.num is unique per config
# and runs never overwrite each other. The launcher blocks per -m (chunks run serially).
for s in ${INIT_SEEDS}; do
  python3 -u run.py -m ${LAUNCHER_ARG} ${TIMEOUT_ARG} \
    +sweep=${ALGO} data=${DATA} task=${TASK} \
    approach=opt init_seed=${s} n_epochs=${N_EPOCHS}
done
