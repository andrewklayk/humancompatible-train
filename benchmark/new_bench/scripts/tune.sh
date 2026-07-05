#!/bin/bash
# Optuna tuning launcher for the opt (KKT) experiments -- the sampled-search
# counterpart to the exhaustive grid in scripts/E1_opt_*.sh.
#
# Submits a FIXED small number of workers that share ONE Optuna study
# (JournalStorage file), so there is NO giant job array -- no MaxArraySize and no
# QOSMaxSubmitJobPerUser limit to hit. When the workers finish, a dependent job
# re-runs the best config at full CV and writes it into multirun/ for the usual
# aggregate + select_best + plot_kkt pipeline (scripts/E1_opt_select.sh).
#
# Run from benchmark/new_bench/:
#   ALGO=pbm bash scripts/tune.sh                       # 8 workers x 25 trials on Slurm
#   ALGO=pbm N_WORKERS=4 N_TRIALS=40 bash scripts/tune.sh
#   LAUNCHER=local ALGO=pbm N_TRIALS=50 bash scripts/tune.sh   # single local worker
#
# After it completes:  bash scripts/E1_opt_select.sh
set -euo pipefail
source scripts/_env.sh
python3 -m pip install -q optuna

ALGO=${ALGO:-pbm}
N_WORKERS=${N_WORKERS:-8}          # Slurm array tasks sharing the study (keep small)
N_TRIALS=${N_TRIALS:-25}           # trials PER worker (total ~ N_WORKERS * N_TRIALS)
N_EPOCHS=${N_EPOCHS:-60}
TAIL=${TAIL:-5}
STUDY=${STUDY:-E1opt_${ALGO}_${DATA}}
STORAGE=${STORAGE:-selection/opt/tuning/${STUDY}.log}
TIME=${TIME:-04:00:00}
mkdir -p "$(dirname "${STORAGE}")"

# init seeds are averaged inside each trial; pass comma-separated to tune.py.
INIT_SEEDS_CSV=$(echo ${INIT_SEEDS} | tr ' ' ',')
COMMON="--algo ${ALGO} --data ${DATA} --task ${TASK} --study ${STUDY} \
  --storage ${STORAGE} --n-epochs ${N_EPOCHS} --tail ${TAIL} --init-seeds ${INIT_SEEDS_CSV}"

if [ "${LAUNCHER}" = "local" ]; then
  # single worker: optimize AND finalize in one process.
  python3 -u tune.py ${COMMON} --n-trials ${N_TRIALS}
  echo "Done. Next:  bash scripts/E1_opt_select.sh"
  exit 0
fi

# Slurm. Modules loaded above are inherited by the job (--export=ALL default).
# 1) N_WORKERS workers optimize the shared study. Each worker writes EVERY trial's
#    per-epoch curves into multirun/ (like the grid); --no-finalize just skips the
#    best_params.json pointer, which the finalize job writes once.
JID=$(sbatch --parsable --array=0-$((N_WORKERS-1)) \
  --job-name=tune_${ALGO} --partition=amdgpufast --gres=gpu:1 \
  --cpus-per-task=4 --mem-per-cpu=8G --time=${TIME} \
  --wrap="cd ${PWD} && python3 -u tune.py ${COMMON} --n-trials ${N_TRIALS} --no-finalize")
echo "submitted worker array ${JID} (${N_WORKERS} workers x ${N_TRIALS} trials)"

# 2) one finalize job, after ALL workers succeed: just reads the study and writes
#    best_params.json (no training -> no GPU needed; all trial curves already saved).
FID=$(sbatch --parsable --dependency=afterok:${JID} \
  --job-name=tune_${ALGO}_final --partition=amdgpufast \
  --cpus-per-task=1 --mem-per-cpu=4G --time=00:10:00 \
  --wrap="cd ${PWD} && python3 -u tune.py ${COMMON} --n-trials 0")
echo "submitted finalize job ${FID} (afterok:${JID})"
echo "When done:  bash scripts/E1_opt_select.sh"
