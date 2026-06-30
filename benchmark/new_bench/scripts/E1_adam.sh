#!/bin/bash
#SBATCH --job-name=E1_adam
# E1 unconstrained reference (Adam). Basic sweeper over its small lr grid
# (conf/sweep/adam.yaml), with folds and init_seeds as cartesian multirun axes.
# Independently (re)runnable.
#   Run from benchmark/new_bench/:   bash scripts/E1_adam.sh
set -euo pipefail
source scripts/_env.sh   # paths are relative to benchmark/new_bench/ (run from there)

# Folds (0..K-1) and init_seeds are cartesian multirun axes alongside the lr grid.
FOLD_CSV=$(seq -s, 0 $((N_FOLDS - 1)))
INIT_CSV=$(echo "${INIT_SEEDS}" | tr ' ' ',')
python3 -u run.py -m ${LAUNCHER_ARG} \
  +sweep=adam data=${DATA} task=${TASK} \
  n_folds=${N_FOLDS} cv_seed=${CV_SEED} fold=${FOLD_CSV} init_seed=${INIT_CSV}
