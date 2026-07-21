# Shared environment + defaults for the E1 sweep scripts. SOURCED, not executed.
# All variables are overridable inline, e.g. a quick local smoke test:
#   LAUNCHER=local INIT_SEEDS=0 N_FOLDS=2 bash scripts/E1_pbm.sh
#
# Assumes the working directory is benchmark/new_bench/ (run.py, multirun/, and
# ../env_humancompatible are relative to it -- same convention as scripts/E3.sh).

# Cluster modules (skipped off-cluster, e.g. a local conda env, where `ml` is absent).
if command -v ml >/dev/null 2>&1; then
  ml PyTorch/2.10.0-foss-2025b-CUDA-12.9.1
  ml Hydra/1.3.2-GCCcore-14.3.0
  ml torchvision/0.25.0-foss-2025b-CUDA-12.9.1
  ml Optuna/4.6.0-foss-2025b
fi


# When a driver runs UNDER sbatch, its SLURM_MEM_* env is exported (--export=ALL) into
# the submitit CHILD jobs, where it collides with the launcher's --mem-per-cpu:
#   srun: fatal: SLURM_MEM_PER_CPU, SLURM_MEM_PER_GPU, and SLURM_MEM_PER_NODE are
#   mutually exclusive.
# Clear them here (sourced before any `run.py -m`) so each child sets its OWN memory
# from conf/hydra/launcher/slurm_gpu.yaml (mem_per_cpu). This does NOT change the
# driver's own already-granted allocation -- it only stops the leak into children.
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU
export SLURM_CPU_BIND=none

# Launcher plugin (install only if missing). The sweep uses Hydra's built-in
# BASIC sweeper (manual grids in conf/sweep/), so no Optuna plugin is needed.
python3 -m pip install -q hydra-submitit-launcher

# --- E1 defaults (override via environment) ---
: "${DATA:=income}"
: "${TASK:=folktables_positive_rate_pair}"
# Cross-validation: separated randomness axes (see README "Cross-validation").
#   INIT_SEEDS -- network-init / batch-order repetitions (optimization variance)
#   N_FOLDS    -- K-fold over the dev set (data variance); CV_SEED fixes the
#                 stratified test hold-out + fold partition.
# Each (fold, init_seed) is a separate basic-sweeper run over the SAME manual grid,
# so every run produces identical configs and select_best can match them across
# folds and inits.
: "${INIT_SEEDS:=0 1 2 3 4}"
: "${N_FOLDS:=5}"
: "${CV_SEED:=0}"
: "${LAUNCHER:=slurm_h200}"    # set LAUNCHER=local to drop the launcher (local run)

# Optional launcher override (omitted when LAUNCHER=local / empty).
LAUNCHER_ARG=""
if [ -n "${LAUNCHER}" ] && [ "${LAUNCHER}" != "local" ]; then
  LAUNCHER_ARG="hydra/launcher=${LAUNCHER}"
fi

# Manual grid via Hydra's BASIC sweeper (the default): each +sweep=<algo> file declares
# hydra.sweeper.params as choice() grids, enumerated as a full cartesian product. The
# opt drivers loop only init_seed outside the -m call; the FULL grid (lr included) is
# swept INSIDE each -m, so hydra.job.num is unique per config and runs never overwrite.
#
# SLURM LIMITS. submitit submits ONE job array per `run.py -m` call (one task per grid
# point), so a grid must stay under both:
#   - MaxArraySize    (`scontrol show config | grep MaxArraySize`, e.g. 1001), and
#   - the QOS submit cap (queued+running jobs). NOTE `array_parallelism`/`%N` throttles
#     RUNNING tasks but does NOT relieve the submit cap -- pending array tasks still count.
# Current opt grids are small (adam 5, ssg 50, alm_proj 100, pbm 100), well under both.
# If you enlarge a grid past a limit, peel its biggest axis into an outer shell loop
# (remove that line from conf/sweep/<algo>.yaml and pass it as a scalar override, e.g.
# `for lr in 0.001 0.01; do run.py -m +sweep=pbm ... algorithm.primal.lr=$lr; done`).
# Peeled chunks are collision-safe automatically: the multirun subdir is a per-config
# hash (conf/config.yaml -> run.py:_hp_hash), so every distinct config gets its own dir.
