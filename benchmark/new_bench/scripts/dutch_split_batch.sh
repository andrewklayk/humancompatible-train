#!/bin/bash
#SBATCH --job-name=dutch_split_batch
#SBATCH --partition=cpu

# Dutch PBM with INDEPENDENT samples for the loss and the constraints
# (+algorithm.split_batch=true, see train.py): batch size 2*${BASE_BS}, loss on one half,
# constraints on the other, so each estimate still uses ${BASE_BS} samples. Twice the
# epochs of the baseline -> same number of optimizer steps as the B=${BASE_BS} run.
# Baseline = the existing dutch_pbm_sweep.sh run at B=${BASE_BS}, same penalty_range.
# split_batch is part of the algorithm config, so it gets its own hp-hash dir.
#   Run from benchmark/new_bench/:   sbatch scripts/dutch_split_batch.sh
#   Adapt variant:   PENALTY_UPDATE=dimin_adapt sbatch scripts/dutch_split_batch.sh
# Then:
#   python aggregate.py --runs multirun/dutch_positive_rate_pair --out selection/dutch --approach opt
#   python plotting/plot_split.py --agg selection/dutch/aggregated --out plots/dutch_split_batch.pdf
set -euo pipefail
: "${LAUNCHER:=slurm_gpu}"
source scripts/_env.sh
set -f

: "${BASE_BS:=72}"
: "${RANGE:=[0.001,1.0]}"
: "${PENALTY_UPDATE:=dimin}"
SEEDS=$(echo ${INIT_SEEDS} | tr ' ' ',')

python3 -u run.py -m ${LAUNCHER_ARG} +sweep=pbm data=dutch task=dutch_positive_rate_pair approach=opt \
  init_seed=${SEEDS} n_epochs=$((2 * N_EPOCHS)) task.batch_size=$((2 * BASE_BS)) \
  "algorithm.dual.penalty_range=${RANGE}" algorithm.dual.penalty_update=${PENALTY_UPDATE} \
  +algorithm.split_batch=true
