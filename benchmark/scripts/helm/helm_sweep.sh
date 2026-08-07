#!/bin/bash

#SBATCH --output=out/helmholtz/sweep.out
#SBATCH --nodes=1
#SBATCH --partition=amd
#SBATCH --mem-per-cpu=10G

ml PyTorch/2.9.1-foss-2025b-CUDA-12.9.1
ml Hydra/1.3.2-GCCcore-14.3.0
ml torchvision/0.25.0-foss-2025b-CUDA-12.9.1

python -m pip install hydra-submitit-launcher

# python -u PDE_Helmholtz.py -m algorithm=adam \
#     algorithm.primal__lr=0.0005,0.001,0.005 \
#     +algorithm.primal__weight_decay=0.01 \
#     +algorithm.beta=0.1,1,2,5 \
#     +model=deep_narrow \
#     seed=0,1,2 \
#     task=helmholtz \
#     data=helmholtz \

python -u PDE_Helmholtz.py -m algorithm=pbm \
    algorithm.primal__lr=0.0005,0.001,0.005 \
    +algorithm.primal__weight_decay=0.01 \
    algorithm.dual__penalty_mult=0.99,0.999,0.9999,1. \
    algorithm.dual__penalty_update=dimin \
    algorithm.dual__pbf=quadratic_logarithmic \
    algorithm.dual__gamma=0.1,0.2,0.9,0.99,0.999 \
    algorithm.dual__delta=1.0 \
    algorithm.moreau__mu=1. \
    seed=0,1,2 \
    task=helmholtz \
    data=helmholtz \
    model=deep_narrow \
    +algorithm.beta=1 \


python -u PDE_Helmholtz.py -m algorithm=pbm \
    algorithm.primal__lr=0.0005,0.001,0.005 \
    +algorithm.primal__weight_decay=0.01 \
    algorithm.dual__penalty_mult=0.99,0.999,0.9999 \
    algorithm.dual__penalty_update=dimin_adapt \
    algorithm.dual__pbf=quadratic_logarithmic \
    algorithm.dual__gamma=0.1,0.2,0.9,0.99,0.999 \
    algorithm.dual__delta=1.0 \
    algorithm.moreau__mu=1. \
    seed=0,1,2 \
    task=helmholtz \
    data=helmholtz \
    model=deep_narrow \
    +algorithm.beta=1 \


python -u PDE_Helmholtz.py -m algorithm=alm \
    algorithm.primal__lr=0.0005,0.001,0.005 \
    +algorithm.primal__weight_decay=0.01 \
    algorithm.dual__lr=0.0005,0.001,0.005 \
    algorithm.dual__penalty=0.,1.,2. \
    algorithm.moreau__mu=1. \
    seed=0,1,2 \
    task=helmholtz \
    data=helmholtz \
    model=deep_narrow \
    +algorithm.beta=1 \
