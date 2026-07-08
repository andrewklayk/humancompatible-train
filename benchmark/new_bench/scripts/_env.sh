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
: "${LAUNCHER:=slurm_gpu}"    # set LAUNCHER=local to drop the launcher (local run)

# Optional launcher override (omitted when LAUNCHER=local / empty).
LAUNCHER_ARG=""
if [ -n "${LAUNCHER}" ] && [ "${LAUNCHER}" != "local" ]; then
  LAUNCHER_ARG="hydra/launcher=${LAUNCHER}"
fi

# Manual grid via Hydra's BASIC sweeper (the default): each +sweep=<algo> file
# declares hydra.sweeper.params as choice()/range() grids, enumerated as a full
# cartesian product. No extra sweeper args needed.
#
# SLURM ARRAY-SIZE LIMIT. submitit submits ONE job array per `run.py -m` call, one
# task per grid point, so a single invocation's grid must stay under the cluster's
# MaxArraySize (`scontrol show config | grep MaxArraySize`; e.g. 1001). NOTE:
# `array_parallelism` in conf/hydra/launcher/slurm*.yaml only caps CONCURRENCY
# (the `%N` in --array=0-M%N); it does NOT change the array size and will NOT fix
# "Invalid job array specification". These scripts already loop fold/init_seed
# OUTSIDE the -m call so each array = grid size only. Current grids: pbm 600,
# alm_max 225, ssg 75, alm_proj 16 -- all under 1001.
#
# If you enlarge a grid past MaxArraySize, peel its biggest axis into an outer shell
# loop: remove that line from conf/sweep/<algo>.yaml and pass it as a scalar override,
# e.g.  for lr in 0.001 0.005 0.01; do python run.py -m +sweep=pbm ... \
#          algorithm.primal.lr=$lr; done   # each array = grid / (#lr values)
#
# QOS SUBMIT LIMIT (a SEPARATE, second cap). Even a valid-size array is rejected with
# "sbatch: error: QOSMaxSubmitJobPerUserLimit" if it would push your queued+running
# job count over the QOS cap -- and, unlike the running-job cap, array_parallelism
# (`%N`) does NOT relieve it (pending array tasks still count). We stay under it by
# ALSO peeling algorithm.primal.lr into an outer loop (below), so each `run.py -m`
# submits only grid/#lr tasks. This is safe WITHOUT any throttle because the submitit
# launcher BLOCKS per `-m` (it waits on job.results()), so chunks run strictly one at
# a time and never stack -- provided each chunk (grid/#lr) is itself under the cap.
# If a single lr-chunk is still too big, peel a second axis the same way.
#
# sweep_lrs echoes the lr grid from an algo's own sweep yaml (kept as the single
# source of truth), space-separated, for the peel loop. Each value is passed as a
# PLAIN override algorithm.primal.lr=<v>, which takes precedence over the yaml's
# sweeper param and collapses that axis to one value. (The hydra.sweeper.params.*
# override form is rejected -- Hydra cannot sweep its own config namespace.)
sweep_lrs() {  # $1 = algo name (matches conf/sweep/<algo>.yaml)
  grep 'algorithm.primal.lr' "conf/sweep/$1.yaml" \
    | sed -E 's/.*choice\(([^)]*)\).*/\1/; s/,/ /g'
}
