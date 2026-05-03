#!/bin/bash

#SBATCH --job-name=grid_cifar10_pbm_dimin
#SBATCH --error=grid_cifar10_pbm_dimin.err
#SBATCH --output=grid_cifar10_pbm_dimin.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=h200fast

ml PyTorch/2.9.1-foss-2025b-CUDA-12.9.1
ml Hydra/1.3.2-GCCcore-14.3.0
ml torchvision/0.25.0-foss-2025b-CUDA-12.9.1

python -m pip install -e ../.

python -u run_gridsearch.py task=cifar10_loss "task.algorithms=['pbm']" +task.alg_version='dimin' data=dutch n_epochs=10 "+task.seeds=[0, 1, 2]"