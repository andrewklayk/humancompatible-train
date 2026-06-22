#!/bin/bash
#SBATCH --job-name=grid_helmholtz
#SBATCH --error=./results/logs/grid_helmholtz_%a.err
#SBATCH --output=./results/logs/grid_helmholtz_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=4:00:00
#SBATCH --partition=h200fast
#SBATCH --array=0-2

ml PyTorch/2.10.0-foss-2025b-CUDA-12.9.1
ml Hydra/1.3.2-GCCcore-14.3.0
ml torchvision/0.25.0-foss-2025b-CUDA-12.9.1
source ../env_humancompatible/bin/activate

python3 -u run_helmholtz.py task=helmholtz seed=$SLURM_ARRAY_TASK_ID