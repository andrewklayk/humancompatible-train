#!/bin/bash

#SBATCH --output=out/dutch/alm_slack.out
#SBATCH --nodes=1
#SBATCH --partition=amdfast

ml PyTorch/2.9.1-foss-2025b-CUDA-12.9.1
ml Hydra/1.3.2-GCCcore-14.3.0

# python -m pip install -e ../.
python -m pip install hydra-submitit-launcher

python -u run_single_experiment.py -m algorithm=alm_slack \
    algorithm.primal__lr=0.0001,0.0005,0.001,0.005 \
    algorithm.dual__lr=0.0001,0.0005,0.001,0.005 \
    algorithm.dual__penalty=0.,1.,2. \
    algorithm.moreau__mu=0,0.1,0.5,1.,2. \
    seed=0,1,2 \
    task=dutch_positive_rate_pair \
    data=dutch