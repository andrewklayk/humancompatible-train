#!/bin/bash

#SBATCH --job-name=grid_income_vec_alm_max
#SBATCH --error=grid_income_vec_alm_max.err
#SBATCH --output=grid_income_vec_alm_max.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=amd

ml PyTorch/2.9.1-foss-2025b-CUDA-12.9.1
ml Hydra/1.3.2-GCCcore-14.3.0

python -m pip install -e ../.

python -u run_gridsearch.py task=folktables_positive_rate_vec_alm_max data=income_sex n_epochs=10