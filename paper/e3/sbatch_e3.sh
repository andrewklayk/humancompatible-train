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
# Prerequisites, both on a login node (compute nodes have no network):
#   1. python paper/e3/prepare_data.py --model Qwen/Qwen2.5-0.5B --out "$SHARD"
#   2. huggingface-cli download Qwen/Qwen2.5-0.5B     # ungated, Apache-2.0, ~1 GB
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
  python -m venv --system-site-packages "$VENV"
  "$VENV/bin/pip" install -q transformers datasets
fi
source "$VENV/bin/activate"

# The weights and the shard are already local; fail loudly rather than reaching out.
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

cd "$(dirname "$0")/../.."

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B}"
SHARD="${SHARD:-paper/results/e3/tokens.bin}"
if [ ! -f "$SHARD" ]; then
  echo "no token shard at $SHARD -- run prepare_data.py on a login node first" >&2
  exit 1
fi

nvidia-smi --query-gpu=index,name,memory.total --format=csv

python paper/e3/sweep.py   --model "$MODEL" --tokens "$SHARD" --ranks 2
python paper/e3/scaling.py --model "$MODEL" --tokens "$SHARD" --worlds 1,2
