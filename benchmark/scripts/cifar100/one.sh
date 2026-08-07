#!/bin/bash

#SBATCH --output=out/cifar100/one.out
#SBATCH --nodes=1
#SBATCH --partition=h200fast
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=64G

ml PyTorch/2.9.1-foss-2025b-CUDA-12.9.1
ml Hydra/1.3.2-GCCcore-14.3.0
ml torchvision/0.25.0-foss-2025b-CUDA-12.9.1

cat conf/experiment.yaml

# python -m pip install -e ../.
python -m pip install tinyimagenet
python -m pip install hydra-submitit-launcher

python -u run_single_experiment.py algorithm=pbm_dimin \
    algorithm.primal__lr=0.002 \
    +algorithm.primal__weight_decay=0.01 \
    algorithm.dual__penalty_mult=0.99 \
    algorithm.dual__penalty_update=const \
    algorithm.dual__pbf=quadratic_reciprocal \
    algorithm.dual__gamma=0.95 \
    algorithm.dual__delta=1.0 \
    algorithm.moreau__mu=2. \
    seed=0 \
    task=cifar100_loss \
    data=dutch \
