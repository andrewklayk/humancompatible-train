#!/bin/bash
#SBATCH --job-name=sparse_llm_train
#SBATCH --error=./results/logs/sparse_llm_train_%a.err
#SBATCH --output=./results/logs/grid_klein_gordon_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=24:00:00
#SBATCH --partition=amdgpu

if command -v ml >/dev/null 2>&1; then
  ml PyTorch/2.11.0-foss-2025b-CUDA-12.9.1
  ml Hydra/1.3.2-GCCcore-14.3.0
  ml torchvision/0.28.0-foss-2025b-CUDA-12.9.1
  ml Optuna/4.6.0-foss-2025b
  ml Transformers/4.57.1-gfbf-2025b gfbf/2025b Python/3.13.5-GCCcore-14.3.0 SciPy-bundle/2025.07-gfbf-2025b PyYAML/6.0.2-GCCcore-14.3.0 tqdm/4.67.1-GCCcore-14.3.0 tokenizers/0.22.1-GCCcore-14.3.0 Safetensors/0.6.2-gfbf-2025b
  source env_empty/bin/activate
  # pip install "datasets<4.5"
  # pip install ../../
  # pip install folktables
  # pip install fairret
  # pip install fairml-datasets
  # pip install scikit-learn
  # pip install 'accelerate>=0.26.0'
fi

python3 train_sparse_llm.py
