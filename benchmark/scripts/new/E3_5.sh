#!/bin/bash
#SBATCH --job-name=grid_income
#SBATCH --error=./results/logs/grid_income_%a.err
#SBATCH --output=./results/logs/grid_income_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=amdgpufast
#SBATCH --array=0-4

# run this from the benchmarking folder
ml PyTorch/2.10.0-foss-2025b-CUDA-12.9.1
ml Hydra/1.3.2-GCCcore-14.3.0
ml torchvision/0.25.0-foss-2025b-CUDA-12.9.1
source ../env_humancompatible/bin/activate

python3 -u run_gridsearch.py task=folktables_positive_rate_pair data=income n_epochs=30 task.seed=$SLURM_ARRAY_TASK_ID task.algorithms="[ssg]"