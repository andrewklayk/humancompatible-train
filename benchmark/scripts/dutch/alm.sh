#!/bin/bash

#SBATCH --job-name=grid_dutch_alm
#SBATCH --error=grid_dutch_alm.err
#SBATCH --output=grid_dutch_alm.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=amdfast

ml PyTorch/2.9.1-foss-2025b-CUDA-12.9.1
ml Hydra/1.3.2-GCCcore-14.3.0

python -m pip install -e ../.

python -u run_gridsearch.py task=dutch_positive_rate_pair +task.algorithms=['alm_slack'] data=dutch n_epochs=10