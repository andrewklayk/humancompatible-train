#!/bin/bash

#SBATCH --job-name=grid_cifar10_ssg
#SBATCH --error=grid_cifar10_ssg.err
#SBATCH --output=grid_cifar10_ssg.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=h200fast

ml PyTorch/2.9.1-foss-2025b-CUDA-12.9.1
ml Hydra/1.3.2-GCCcore-14.3.0
ml torchvision/0.25.0-foss-2025b-CUDA-12.9.1

python -m pip install -e ../.

python -u run_gridsearch.py task=cifar10_loss task.algorithms=['ssg'] data=dutch n_epochs=10