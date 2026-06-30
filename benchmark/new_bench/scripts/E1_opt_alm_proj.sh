#!/bin/bash
#SBATCH --job-name=E1_opt_alm_proj
# E1-opt sweep for ALM (projected): full dataset as train, no val/test split.
#   Run from benchmark/new_bench/:   bash scripts/E1_opt_alm_proj.sh
#   Local smoke test:   LAUNCHER=local INIT_SEEDS=0 bash scripts/E1_opt_alm_proj.sh
set -euo pipefail
source scripts/_env.sh

ALGO=alm_proj
INIT_CSV=$(echo "${INIT_SEEDS}" | tr ' ' ',')
python3 -u run.py -m ${LAUNCHER_ARG} \
  +sweep=${ALGO} data=${DATA} task=${TASK} \
  approach=opt init_seed=${INIT_CSV}
