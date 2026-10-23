
if command -v ml >/dev/null 2>&1; then
  ml PyTorch/2.11.0-foss-2025b-CUDA-12.9.1
  ml Hydra/1.3.2-GCCcore-14.3.0
  ml torchvision/0.28.0-foss-2025b-CUDA-12.9.1
  ml Optuna/4.6.0-foss-2025b
  ml Transformers/5.0.0-gfbf-2025b gfbf/2025b Python/3.13.5-GCCcore-14.3.0 SciPy-bundle/2025.07-gfbf-2025b tokenizers/0.22.2-GCCcore-14.3.0 Safetensors/0.7.0-gfbf-2025b
  source env_empty/bin/activate
  pip install ../../
  pip install folktables
  pip install fairret
  pip install fairml-datasets
  pip install scikit-learn
fi
