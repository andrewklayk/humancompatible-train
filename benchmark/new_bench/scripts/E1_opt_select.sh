#!/bin/bash
# E1-opt selection: aggregates ONLY approach=opt runs (full-data, train-selected,
# with full-batch KKT metrics) into a separate selection dir so they never merge
# with the ml-approach cells. Re-runnable any time; no training, no GPU.
#   Run from benchmark/new_bench/:   bash scripts/E1_opt_select.sh
#   Custom slack levels:             TOLS=1.0,1.05,1.1 bash scripts/E1_opt_select.sh
set -euo pipefail
if [ -f ../env_humancompatible/bin/activate ]; then
  source ../env_humancompatible/bin/activate
fi

if command -v ml >/dev/null 2>&1; then
  ml PyTorch/2.10.0-foss-2025b-CUDA-12.9.1
  ml Hydra/1.3.2-GCCcore-14.3.0
  ml torchvision/0.25.0-foss-2025b-CUDA-12.9.1
  ml Optuna/4.6.0-foss-2025b
fi

: "${TOLS:=1.0,1.1,1.25}"
# Two stages: aggregate (opt runs -> selection/opt/aggregated/) then select.
python3 aggregate.py   --runs multirun/ --out selection/opt --approach opt
python3 select_best.py --agg selection/opt/aggregated --out selection/opt --tols "${TOLS}"
echo "KKT plots:  cd plotting && python plot_kkt.py --agg ../selection/opt/aggregated \\"
echo "              --task <task> --data <data> --mode cdf --metric residual"
