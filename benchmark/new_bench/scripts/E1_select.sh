#!/bin/bash
# E1 selection: threshold-swept feasibility-first winners over whatever runs
# currently live in multirun/. Re-runnable any time; no training, no GPU. Safe to
# run after any subset of the per-algorithm sweeps has finished.
#   Run from benchmark/new_bench/:   bash scripts/E1_select.sh
#   Custom slack levels:             TOLS=1.0,1.05,1.1,1.25 bash scripts/E1_select.sh

ml PyTorch/2.10.0-foss-2025b-CUDA-12.9.1
ml Hydra/1.3.2-GCCcore-14.3.0
ml torchvision/0.25.0-foss-2025b-CUDA-12.9.1
ml Optuna/4.6.0-foss-2025b

set -euo pipefail
if [ -f ../env_humancompatible/bin/activate ]; then
  source ../env_humancompatible/bin/activate
fi
: "${TOLS:=1.0,1.1,1.25}"
# Two stages: aggregate (raw runs -> selection/aggregated/) then select (-> best_*.json).
# --approach ml keeps the CV/val-selected runs separate from opt (E1_opt_select.sh).
python3 aggregate.py   --runs multirun/ --out selection/ --approach opt
python3 select_best.py --agg selection/aggregated --out selection/ --tols "${TOLS}"
