# Shared environment + defaults for the E1 sweep scripts. SOURCED, not executed.
# All variables are overridable inline, e.g. a quick local smoke test:
#   LAUNCHER=local INIT_SEEDS=0 N_FOLDS=2 bash scripts/E1_pbm.sh
#
# Assumes the working directory is benchmark/new_bench/ (run.py, multirun/, and
# ../env_humancompatible are relative to it -- same convention as scripts/E3.sh).

# Cluster modules (skipped off-cluster, e.g. a local conda env).
# ml PyTorch/2.10.0-foss-2025b-CUDA-12.9.1
# ml Hydra/1.3.2-GCCcore-14.3.0
# ml torchvision/0.25.0-foss-2025b-CUDA-12.9.1

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
: "${INIT_SEEDS:=0 1 2}"
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
