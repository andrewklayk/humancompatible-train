#!/bin/bash
#SBATCH --job-name=E1_ssg
# E1 sweep for switching-subgradient (SSG) only. Independently (re)runnable.
#   Run from benchmark/new_bench/:   bash scripts/E1_ssg.sh
#   Local smoke test:   LAUNCHER=local INIT_SEEDS=0 N_FOLDS=2 bash scripts/E1_ssg.sh
set -euo pipefail
source scripts/_env.sh   # paths are relative to benchmark/new_bench/ (run from there)

ALGO=ssg
# Basic sweeper enumerates the manual grid (conf/sweep/ssg.yaml) per (fold, init_seed).
for f in $(seq 0 $((N_FOLDS - 1))); do
  for s in ${INIT_SEEDS}; do
    python3 -u run.py -m ${LAUNCHER_ARG} \
      +sweep=${ALGO} data=${DATA} task=${TASK} \
      n_folds=${N_FOLDS} cv_seed=${CV_SEED} fold=${f} init_seed=${s}
  done
done
