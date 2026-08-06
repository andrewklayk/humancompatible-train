#!/bin/bash
#SBATCH --job-name=grid_weight
#SBATCH --error=./results/logs/grid_weight%a.err
#SBATCH --output=./results/logs/grid_weight%a.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=h200fast
#SBATCH --array=0-4

# run this (running)

ml PyTorch/2.10.0-foss-2025b-CUDA-12.9.1
ml Hydra/1.3.2-GCCcore-14.3.0
ml torchvision/0.25.0-foss-2025b-CUDA-12.9.1
source ../../env_humancompatible/bin/activate

python run.py data=cifar100 task=cifar10_loss algorithm=adam approach=opt n_epochs=1