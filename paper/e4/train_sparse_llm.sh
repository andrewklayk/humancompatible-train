#!/bin/bash
#SBATCH --job-name=sparse_llm_train
#SBATCH --error=./results/logs/sparse_llm_train_%A_%a.err
#SBATCH --output=./results/logs/sparse_llm_train_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --partition=h200

# One array task = one (grid line x seed). From paper/e4 on a LOGIN node, with bash, not sbatch:
#   GRID=grid.txt SEEDS="0 1 2" bash train_sparse_llm.sh
# Each non-comment line of $GRID is the argument list for train_sparse_llm.py (no spaces inside values).
# `sbatch train_sparse_llm.sh` alone runs the first grid line with the first seed.
# Results land in results/runs/<model>_<method>_seed<s>_<config hash>/{metrics.csv,params.json}.

GRID="${GRID:-grid.txt}"
SEEDS=(${SEEDS:-0})
mapfile -t LINES < <(grep -v -e '^[[:space:]]*#' -e '^[[:space:]]*$' "$GRID")

if [ -z "${SLURM_JOB_ID:-}" ]; then
  mkdir -p results/logs
  exec sbatch --array=0-$(( ${#LINES[@]} * ${#SEEDS[@]} - 1 )) "$0"
fi

TASK="${SLURM_ARRAY_TASK_ID:-0}"
LINE="${LINES[TASK / ${#SEEDS[@]}]}"
SEED="${SEEDS[TASK % ${#SEEDS[@]}]}"

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

echo "+ python3 train_sparse_llm.py $LINE --seed $SEED"
python3 train_sparse_llm.py $LINE --seed "$SEED"
