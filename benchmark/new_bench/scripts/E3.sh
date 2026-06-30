#!/bin/bash
#SBATCH --job-name=income

# run this from the benchmarking folder
ml PyTorch/2.10.0-foss-2025b-CUDA-12.9.1
ml Hydra/1.3.2-GCCcore-14.3.0
ml torchvision/0.25.0-foss-2025b-CUDA-12.9.1
source ../env_humancompatible/bin/activate
pip install hydra-submitit-launcher

python3 -u run.py -m hydra/launcher=slurm_gpu algorithm=alm_proj +sweep=alm_proj \
  data=income task=folktables_positive_rate_pair \
  n_folds=5 cv_seed=0 fold=0,1,2,3,4 init_seed=0,1,2