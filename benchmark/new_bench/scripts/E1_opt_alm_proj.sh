#!/bin/bash
#SBATCH --job-name=E1_opt_alm_proj
# E1-opt sweep for ALM (projected): full dataset as train, no val/test split.
#   Run from benchmark/new_bench/:   bash scripts/E1_opt_alm_proj.sh
#   Local smoke test:   LAUNCHER=local INIT_SEEDS=0 bash scripts/E1_opt_alm_proj.sh
set -euo pipefail
source scripts/_env.sh

ALGO=alm_proj
# init_seed AND lr loop OUTSIDE the multirun so each -m invocation submits ONE array of
# size = grid/#lr (not grid x n_init_seeds), staying under Slurm's MaxArraySize and the
# QOS submit cap. The launcher blocks per -m, so these chunks run one at a time.
for s in ${INIT_SEEDS}; do
  for lr in $(sweep_lrs ${ALGO}); do
    python3 -u run.py -m ${LAUNCHER_ARG} \
      +sweep=${ALGO} data=${DATA} task=${TASK} \
      approach=opt init_seed=${s} algorithm.primal.lr=${lr}
  done
done
