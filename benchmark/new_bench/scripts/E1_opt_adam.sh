#!/bin/bash
#SBATCH --job-name=E1_opt_adam
# E1-opt sweep for Adam (unconstrained reference): full dataset as train, no val/test split.
#   Run from benchmark/new_bench/:   bash scripts/E1_opt_adam.sh
#   Local smoke test:   LAUNCHER=local INIT_SEEDS=0 bash scripts/E1_opt_adam.sh
set -euo pipefail
source scripts/_env.sh

# init_seed loops OUTSIDE the multirun so each -m invocation submits ONE array of
# size = grid only (not grid x n_init_seeds), staying under Slurm's MaxArraySize.
for s in ${INIT_SEEDS}; do
  python3 -u run.py -m ${LAUNCHER_ARG} \
    +sweep=adam data=${DATA} task=${TASK} \
    approach=opt init_seed=${s}
done
