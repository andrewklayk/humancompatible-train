#!/bin/bash

#SBATCH --output=out/tinyimagenet/sweep.out
#SBATCH --nodes=1
#SBATCH --partition=amd

ml PyTorch/2.9.1-foss-2025b-CUDA-12.9.1
ml Hydra/1.3.2-GCCcore-14.3.0
ml torchvision/0.25.0-foss-2025b-CUDA-12.9.1

cat conf/experiment.yaml

# python -m pip install -e ../.
python -m pip install tinyimagenet
python -m pip install hydra-submitit-launcher

python -u run_single_experiment.py -m algorithm=pbm_dimin \
    algorithm.primal__lr=0.0005,0.001 \
    +algorithm.primal__weight_decay=0.01 \
    algorithm.dual__penalty_mult=0.9999,1. \
    algorithm.dual__penalty_update=dimin \
    algorithm.dual__pbf=quadratic_reciprocal \
    algorithm.dual__gamma=0.95,0.999 \
    algorithm.dual__delta=1.0 \
    algorithm.moreau__mu=1. \
    seed=0,1,2 \
    task=tinyimagenet \
    data=income_sex \


python -u run_single_experiment.py -m algorithm=alm_max \
    algorithm.primal__lr=0.0005,0.001 \
    +algorithm.primal__weight_decay=0.01 \
    algorithm.dual__lr=0.0005,0.001 \
    algorithm.dual__penalty=0. \
    algorithm.moreau__mu=1. \
    seed=0,1,2 \
    task=tinyimagenet \
    data=income_sex \

python -u run_single_experiment.py -m algorithm=alm_slack \
    algorithm.primal__lr=0.0005,0.001 \
    +algorithm.primal__weight_decay=0.01 \
    algorithm.dual__lr=0.0005,0.001 \
    algorithm.dual__penalty=0. \
    algorithm.moreau__mu=1. \
    seed=0,1,2 \
    task=tinyimagenet \
    data=income_sex \

