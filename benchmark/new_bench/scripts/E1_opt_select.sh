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
: "${TOLS:=1.0,1.1,1.25}"
python3 select_best.py --runs multirun/ --out selection/opt --approach opt --tols "${TOLS}"
echo "KKT plots:  cd plotting && python plot_kkt.py --agg ../selection/opt/aggregated \\"
echo "              --task <task> --data <data> --mode cdf --metric residual"
