#!/bin/bash

#SBATCH --output=out/burgers/one.out
#SBATCH --nodes=1
#SBATCH --partition=h200fast
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=10G

ml PyTorch/2.9.1-foss-2025b-CUDA-12.9.1
ml Hydra/1.3.2-GCCcore-14.3.0
ml torchvision/0.25.0-foss-2025b-CUDA-12.9.1

python -m pip install hydra-submitit-launcher


# python -u PDE_Viscous_Burgers.py algorithm=pbm \
#     algorithm.primal__lr=0.001 \
#     +algorithm.primal__weight_decay=0.01 \
#     algorithm.dual__penalty_mult=0.9999 \
#     algorithm.dual__penalty_update=dimin_adapt \
#     algorithm.dual__pbf=quadratic_logarithmic \
#     algorithm.dual__gamma=0.999 \
#     algorithm.dual__delta=1.0 \
#     algorithm.moreau__mu=1. \
#     seed=0 \
#     task=burgers \
#     data=burgers \
#     model=deep_narrow \
#     +algorithm.beta=1 \


python -u PDE_Viscous_Burgers.py algorithm=alm \
    algorithm.primal__lr=0.0005 \
    +algorithm.primal__weight_decay=0.01 \
    algorithm.dual__lr=0.0005 \
    algorithm.dual__penalty=0. \
    algorithm.moreau__mu=1. \
    seed=0 \
    task=burgers \
    data=burgers \
    model=deep_narrow \
    +algorithm.beta=1 \