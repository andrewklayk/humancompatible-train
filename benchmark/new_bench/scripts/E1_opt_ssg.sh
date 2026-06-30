#!/bin/bash
#SBATCH --job-name=E1_opt_ssg
# E1-opt sweep for SSG: full dataset as train, no val/test split.
#   Run from benchmark/new_bench/:   bash scripts/E1_opt_ssg.sh
#   Local smoke test:   LAUNCHER=local INIT_SEEDS=0 bash scripts/E1_opt_ssg.sh
set -euo pipefail
source scripts/_env.sh

ALGO=ssg
INIT_CSV=$(echo "${INIT_SEEDS}" | tr ' ' ',')
python3 -u run.py -m ${LAUNCHER_ARG} \
  +sweep=${ALGO} data=${DATA} task=${TASK} \
  approach=opt init_seed=${INIT_CSV}
