#!/bin/bash
#SBATCH --job-name=cifar100_runtime
#SBATCH --partition=h200fast
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --output=cifar100_runtime.out

# Runs cifar100_runtime.py: a plain sequential sweep (algorithm x constraint-count x
# init_seed), NOT a Hydra multirun/submitit grid driver, so this script requests the
# GPU directly instead of going through hydra/launcher=slurm_*. Same partition/GPU/mem
# as the cifar100 `opt` launcher (conf/hydra/launcher/slurm_h200.yaml, README's E6).
#
# IMPORTANT: CIFAR100 must already be downloaded under new_bench/data/ before this runs
# on a compute node (which may lack internet access) -- see the README's "make a dummy
# training run to load the data" step, e.g. from an interactive node:
#   python run.py data=cifar100 task=cifar100_loss algorithm=adam approach=opt n_epochs=1
#
# Run from benchmark/new_bench/:   sbatch scripts/cifar100_runtime.sh
set -euo pipefail

ml PyTorch/2.10.0-foss-2025b-CUDA-12.9.1
ml Hydra/1.3.2-GCCcore-14.3.0
ml torchvision/0.25.0-foss-2025b-CUDA-12.9.1

python3 -u cifar100_runtime.py
