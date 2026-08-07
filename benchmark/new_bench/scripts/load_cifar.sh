#!/bin/bash
#SBATCH --job-name=load_cifar
#SBATCH --partition=cpulong
#SBATCH --output=load_cifar.out

ml PyTorch/2.10.0-foss-2025b-CUDA-12.9.1
ml Hydra/1.3.2-GCCcore-14.3.0
ml torchvision/0.25.0-foss-2025b-CUDA-12.9.1
ml Optuna/4.6.0-foss-2025b

python -u run.py data=cifar10 task=cifar10_loss algorithm=adam approach=opt n_epochs=1
python -u run.py data=cifar100 task=cifar100_loss algorithm=adam approach=opt n_epochs=1