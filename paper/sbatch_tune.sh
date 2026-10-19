#!/usr/bin/env bash
#SBATCH --job-name=tune
#SBATCH --partition=h200fast
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-tune-%j.out
#
# Run a paper/tune.py hyperparameter sweep inside ONE allocation.
#
#   sbatch paper/sbatch_tune.sh                                 # e3_llm_alm,  8 cells
#   SWEEP=e3_llm_nupi sbatch paper/sbatch_tune.sh                # e3_llm_nupi, 16 cells
#   SWEEP=e3_llm_spbm SEEDS=0,1 sbatch paper/sbatch_tune.sh      # 16 cells x 2 seeds
#   SWEEP=e2a_alm sbatch --partition=amd --gres=none paper/sbatch_tune.sh
#
# Or fan the cells out as a SLURM array instead of walking them -- from a LOGIN node, with
# no sbatch, because this mode SUBMITS rather than computes:
#
#   LAUNCHER=slurm_h200 SWEEP=e3_llm_nupi bash paper/sbatch_tune.sh
#
# It blocks until every array task finishes (then reports), so run it under tmux/nohup.
#
# Trailing arguments are passed to tune.py as extra Hydra overrides, and come last, so
# they win over everything this script sets:
#
#   sbatch paper/sbatch_tune.sh ++experiment.steps=1000 ++experiment.eps=0.3
#   MODEL=stub sbatch paper/sbatch_tune.sh          # or ++experiment.model=stub
#
# E3-LM tuning is SINGLE-RANK, and this script pins `experiment.ranks=1`. tune.py drives
# run_llm.train() directly -- a config object cannot cross the torchrun re-exec that
# run_llm.main() performs, and train() sits underneath it, so the adapter refuses
# anything else. To tune *for* a multi-rank run, match its effective batch:
#
#   sbatch paper/sbatch_tune.sh ++experiment.batch_size=16      # tuning for 2 ranks x 8
#
# and use paper/e3/sweep.py for the multi-rank runs themselves.
#
# Budget: one E3-LM cell is ~8 min at experiment.steps=5000 on one H200, so a 16-cell
# sweep is ~2 h and three seeds of it ~6 h. Raise --time before adding seeds.
#
# Without LAUNCHER the grid runs SEQUENTIALLY: with no `hydra/launcher=` override Hydra
# uses BasicLauncher, which walks the cells in one process. `-m` still gives each cell its
# own output directory and row.json, but not its own job. A 16-cell E3-LM sweep is then
# ~2 h on the one GPU this job holds.
#
# With LAUNCHER=slurm_h200 the submitit plugin submits one array task per cell instead,
# up to that launcher's `array_parallelism`. That is why it must NOT run under sbatch: a
# compute node should not be submitting jobs, and the script refuses to.
# conf/hydra/launcher/slurm_h200.yaml carries the module and venv in its `setup:` block,
# which the older slurm_gpu.yaml leaves commented out.
#
# Prerequisites are paper/e3/sbatch_e3.sh's -- the venv and the token shard are both built
# on a login node, because compute nodes have no network -- PLUS Hydra in that venv, which
# sbatch_e3.sh does not need and so does not install:
#
#   ~/venv-e3/bin/pip install hydra-core hydra-submitit-launcher
#
# An e3_cifar_* sweep needs neither the shard nor transformers, but it does need
# torchvision in that venv and CIFAR already on disk -- again from a login node:
#
#   python paper/e3/prepare_cifar.py          # -> ./data, where experiment.data points
set -euo pipefail

# SLURM exports these into child processes, where they collide with any per-job memory
# flag. Clearing them does not touch this job's own allocation.
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU

ml PyTorch/2.10.0-foss-2025b-CUDA-12.9.1

VENV="${VENV:-$HOME/venv-e3}"
if [ ! -d "$VENV" ]; then
  echo "no venv at $VENV -- create it on a login node; see paper/e3/sbatch_e3.sh" >&2
  exit 1
fi
source "$VENV/bin/activate"

# tune.py imports hydra and omegaconf at module level, and sbatch_e3.sh's venv recipe
# installs neither -- fail with the fix rather than a bare ModuleNotFoundError.
python - <<'PY' || { echo "  ^ $VENV/bin/pip install hydra-core hydra-submitit-launcher" >&2; exit 1; }
import importlib.util as u, sys
missing = [m for m in ("hydra", "omegaconf") if u.find_spec(m) is None]
if missing:
    print(f"venv is missing {', '.join(missing)}; paper/tune.py cannot import", file=sys.stderr)
    sys.exit(1)
PY

export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false

# NOT `dirname $0`: SLURM runs a spooled copy of this script out of
# /var/spool/slurmd/job<N>/, so $0 does not point into the repository.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
if [ ! -f paper/tune.py ]; then
  echo "not at the repository root (cwd=$PWD); submit from the repo root" >&2
  exit 1
fi

SWEEP="${SWEEP:-e3_llm_alm}"
SEEDS="${SEEDS:-0}"
LAUNCHER="${LAUNCHER:-}"
if [ -n "$LAUNCHER" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "LAUNCHER=$LAUNCHER submits a job array, so this must run on a LOGIN node:" >&2
  echo "  LAUNCHER=$LAUNCHER SWEEP=$SWEEP bash paper/sbatch_tune.sh" >&2
  exit 1
fi
if [ -n "$LAUNCHER" ] && [ ! -f "paper/conf/hydra/launcher/$LAUNCHER.yaml" ]; then
  echo "no launcher config paper/conf/hydra/launcher/$LAUNCHER.yaml" >&2
  exit 1
fi
SWEEP_FILE="paper/conf/sweep/$SWEEP.yaml"
if [ ! -f "$SWEEP_FILE" ]; then
  echo "no sweep config $SWEEP_FILE; available:" >&2
  ls -1 paper/conf/sweep/*.yaml | xargs -n1 basename | sed 's/\.yaml$/  /;s/^/  /' >&2
  exit 1
fi
# Absolute, and passed explicitly rather than left to `experiment.tokens: null`. tune.py
# points the artifact root at each job's own Hydra output directory before run_llm
# resolves its default shard path, so an unset `tokens` makes every cell look for the
# shard inside its own job directory and fail.
SHARD="${SHARD:-$PWD/paper/results/e3/tokens.bin}"
# conf/experiment/e3_llm.yaml defaults to `model: stub` so that tune.py is runnable on a
# laptop, but this script requires a real token shard -- and feeding Qwen-range ids to
# the stub's 256-row embedding is a device-side assert, not a clear error. Default to the
# model the shard was tokenized for, as paper/e3/sbatch_e3.sh does.
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B}"

OVERRIDES=(-m "+sweep=$SWEEP" "seed=$SEEDS")
if [ -n "$LAUNCHER" ]; then
  OVERRIDES+=("hydra/launcher=$LAUNCHER")
fi
if [[ "$SWEEP" == e3_llm* ]]; then
  if [ ! -f "$SHARD" ]; then
    echo "no token shard at $SHARD -- run paper/e3/prepare_data.py on a login node" >&2
    exit 1
  fi
  # ranks is pinned, not configurable: see the header.
  OVERRIDES+=("++experiment.model=$MODEL" "++experiment.tokens=$SHARD"
              "++experiment.ranks=1")
fi

# Informational, and absent on a login node, hence the guard.
command -v nvidia-smi >/dev/null && \
  nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

echo "+ python paper/tune.py ${OVERRIDES[*]} $*"
python paper/tune.py "${OVERRIDES[@]}" "$@"

# hydra.sweep.dir is paper/multirun/<target>/<algorithm.name>. Report on the ALGORITHM
# directory, not the target: tune_report lists every key that differs across the rows it
# is given, so a target directory holding two methods prints each one's knobs against
# `None` for the other. The algorithm comes from the sweep file's own defaults list --
# the file's `_<suffix>` is not it (e3_llm_alm selects alm_hpr).
ALGO=$(sed -n 's|^[[:space:]]*-[[:space:]]*override[[:space:]]*/algorithm:[[:space:]]*||p' \
       "$SWEEP_FILE" | head -1 | tr -d '[:space:]')
RUNS="paper/multirun/${SWEEP%_*}${ALGO:+/$ALGO}"
echo
echo "=============================================================="
python paper/tune_report.py --runs "$RUNS" --out best
