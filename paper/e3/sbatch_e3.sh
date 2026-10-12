#!/usr/bin/env bash
#SBATCH --job-name=e3-sparse-lm
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=slurm-e3-%j.out
#
# E3 on one node with 2 H200s.
#
#   sbatch paper/e3/sbatch_e3.sh
#
# ONE task, not two: the drivers launch their own ranks through
# `torch.distributed.run`, so SLURM must hand them the whole node rather than start two
# copies of the driver. This is the opposite of what submitit wants, which is why E3 uses
# plain sbatch and no Hydra.
#
# Prerequisites, all three on a login node (compute nodes have no network):
#   1. python -m venv --system-site-packages ~/venv-e3   # after `ml PyTorch/...`
#      ~/venv-e3/bin/pip install transformers datasets
#   2. python paper/e3/prepare_data.py --model Qwen/Qwen2.5-0.5B --out paper/results/e3/tokens.bin
#   3. huggingface-cli download Qwen/Qwen2.5-0.5B     # ungated, Apache-2.0, ~1 GB
#
# `transformers` is needed at *training* time too, not only to prepare the shard:
# run_llm.build_model imports it for any --model that is not `stub`.
set -euo pipefail

# SLURM exports these into child processes, where they collide with any per-job memory
# flag: "SLURM_MEM_PER_CPU, SLURM_MEM_PER_GPU, and SLURM_MEM_PER_NODE are mutually
# exclusive". Clearing them does not touch this job's own allocation.
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU

ml PyTorch/2.10.0-foss-2025b-CUDA-12.9.1

# No EasyBuild module provides `transformers`, so it lives in a venv layered on the
# module (--system-site-packages keeps the module's torch, which is the CUDA-matched one).
VENV="${VENV:-$HOME/venv-e3}"
if [ ! -d "$VENV" ]; then
  # Creating it here cannot work: pip needs the network and this is a compute node.
  echo "no venv at $VENV -- create it on a login node first, see the header" >&2
  exit 1
fi
source "$VENV/bin/activate"

# The weights and the shard are already local; fail loudly rather than reaching out.
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

# NOT `dirname $0`: SLURM runs a spooled copy of this script out of
# /var/spool/slurmd/job<N>/, so $0 does not point into the repository.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
if [ ! -f paper/e3/run_llm.py ]; then
  echo "not at the repository root (cwd=$PWD); submit from the repo root" >&2
  exit 1
fi

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B}"
SHARD="${SHARD:-paper/results/e3/tokens.bin}"
if [ ! -f "$SHARD" ]; then
  echo "no token shard at $SHARD -- run prepare_data.py on a login node first" >&2
  exit 1
fi

nvidia-smi --query-gpu=index,name,memory.total --format=csv

python paper/e3/sweep.py   --model "$MODEL" --tokens "$SHARD" --ranks 2
python paper/e3/scaling.py --model "$MODEL" --tokens "$SHARD" --worlds 1,2
