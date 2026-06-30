#!/bin/bash
#SBATCH --job-name=E1_opt_adam
# E1-opt sweep for Adam (unconstrained reference): full dataset as train, no val/test split.
#   Run from benchmark/new_bench/:   bash scripts/E1_opt_adam.sh
#   Local smoke test:   LAUNCHER=local INIT_SEEDS=0 bash scripts/E1_opt_adam.sh
set -euo pipefail
source scripts/_env.sh

INIT_CSV=$(echo "${INIT_SEEDS}" | tr ' ' ',')
python3 -u run.py -m ${LAUNCHER_ARG} \
  +sweep=adam data=${DATA} task=${TASK} \
  approach=opt init_seed=${INIT_CSV}
