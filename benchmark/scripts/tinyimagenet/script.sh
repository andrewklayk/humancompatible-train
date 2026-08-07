#!/bin/bash

#SBATCH --output=out/tinyimagenet/sweep.out
#SBATCH --nodes=1
#SBATCH --partition=h200fast
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=128G

ml PyTorch/2.9.1-foss-2025b-CUDA-12.9.1
ml Hydra/1.3.2-GCCcore-14.3.0
ml torchvision/0.25.0-foss-2025b-CUDA-12.9.1

cat conf/experiment.yaml

# python -m pip install -e ../.
python -m pip install tinyimagenet
python -m pip install hydra-submitit-launcher

python -u tiny_image_net.py