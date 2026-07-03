#!/bin/bash
#SBATCH --job-name=E1_opt_pbm_mirror
# E1-opt sweep for PBM Mirror: full dataset as train, no val/test split (approach=opt).
# Only the optimization variance axis (init_seed) is swept; no fold loop needed.
#   Run from benchmark/new_bench/:   bash scripts/E1_opt_pbm_mirror.sh
#   Local smoke test:   LAUNCHER=local INIT_SEEDS=0 bash scripts/E1_opt_pbm_mirror.sh
set -euo pipefail
source scripts/_env.sh

ALGO=pbm_logscaled
INIT_CSV=$(echo "${INIT_SEEDS}" | tr ' ' ',')
python3 -u run.py -m ${LAUNCHER_ARG} \
  +sweep=${ALGO} data=${DATA} task=${TASK} \
  approach=opt init_seed=${INIT_CSV}
