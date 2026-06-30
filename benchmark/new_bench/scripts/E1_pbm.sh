#!/bin/bash
#SBATCH --job-name=E1_pbm
# E1 sweep for PBM only. Independently (re)runnable; touches only PBM results.
#   Run from benchmark/new_bench/:   bash scripts/E1_pbm.sh
#   Local smoke test:   LAUNCHER=local INIT_SEEDS=0 N_FOLDS=2 bash scripts/E1_pbm.sh
set -euo pipefail
source scripts/_env.sh   # paths are relative to benchmark/new_bench/ (run from there)

ALGO=pbm
# The basic sweeper enumerates the manual grid (conf/sweep/pbm.yaml) once per
# (fold, init_seed) -- the grid is identical across the CV grid, so select_best
# matches configs by hyperparameter signature. Folds = data variance;
# init_seeds = optimization variance.
for f in $(seq 0 $((N_FOLDS - 1))); do
  for s in ${INIT_SEEDS}; do
    python3 -u run.py -m ${LAUNCHER_ARG} \
      +sweep=${ALGO} data=${DATA} task=${TASK} \
      n_folds=${N_FOLDS} cv_seed=${CV_SEED} fold=${f} init_seed=${s}
  done
done
