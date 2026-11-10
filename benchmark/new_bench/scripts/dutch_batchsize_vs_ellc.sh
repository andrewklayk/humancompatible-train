#!/bin/bash
#SBATCH --job-name=dutch_pbm_sweep
#SBATCH --partition=cpulong

# Dutch PBM sweep (approach=opt), two grids submitted AT THE SAME TIME:
#   A: every penalty_range in conf/sweep/pbm.yaml   x  batch size ${BASE_BS}
#   B: penalty_range ${FIXED_RANGE}                 x  batch sizes ${BATCH_SIZES}
# (${BASE_BS} @ ${FIXED_RANGE} is shared by both, so it is only in A.)
# All seeds go INSIDE each -m (one job array per grid, not one per seed); run dirs are
# keyed by fold/init/batch size/hp-hash, so nothing overwrites.
#   Run from benchmark/new_bench/:   sbatch scripts/dutch_pbm_sweep.sh
#   Adaptive variant:   PENALTY_UPDATE=dimin_adapt sbatch scripts/dutch_pbm_sweep.sh
#   Local smoke test:   LAUNCHER=local INIT_SEEDS=0 N_EPOCHS=1 bash scripts/dutch_pbm_sweep.sh
# Then:
#   python aggregate.py --runs multirun/dutch_positive_rate_pair --out selection/dutch --approach opt
#   python plotting/plot_sweep.py --agg selection/dutch/aggregated --penalty-update dimin --out plots/dutch_pbm_sweep_dimin.pdf
#   python plotting/plot_sweep.py --agg selection/dutch/aggregated --penalty-update dimin_adapt --out plots/dutch_pbm_sweep_adapt.pdf
set -euo pipefail
: "${LAUNCHER:=slurm_gpu}"   # CPU partition: run.py forces CPU, so a GPU would sit idle
source scripts/_env.sh
set -f                   # keep [..] list overrides away from shell globbing

: "${BASE_BS:=72}"
: "${BATCH_SIZES:=36,144,288}"   # multiples of 18 (dutch has 18 groups)
: "${FIXED_RANGE:=[0.1,1.0]}"
: "${PENALTY_UPDATE:=dimin}"     # dimin | dimin_adapt (separate run dirs via the hp hash)
SEEDS=$(echo ${INIT_SEEDS} | tr ' ' ',')
COMMON="${LAUNCHER_ARG} +sweep=pbm data=dutch task=dutch_positive_rate_pair approach=opt init_seed=${SEEDS} n_epochs=${N_EPOCHS} algorithm.dual.penalty_update=${PENALTY_UPDATE}"

python3 -u run.py -m ${COMMON} task.batch_size=${BASE_BS} &
A=$!
python3 -u run.py -m ${COMMON} task.batch_size=${BATCH_SIZES} "algorithm.dual.penalty_range=${FIXED_RANGE}" &
B=$!
wait $A
wait $B
