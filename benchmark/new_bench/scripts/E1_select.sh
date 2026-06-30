#!/bin/bash
# E1 selection: threshold-swept feasibility-first winners over whatever runs
# currently live in multirun/. Re-runnable any time; no training, no GPU. Safe to
# run after any subset of the per-algorithm sweeps has finished.
#   Run from benchmark/new_bench/:   bash scripts/E1_select.sh
#   Custom slack levels:             TOLS=1.0,1.05,1.1,1.25 bash scripts/E1_select.sh
set -euo pipefail
if [ -f ../env_humancompatible/bin/activate ]; then
  source ../env_humancompatible/bin/activate
fi
: "${TOLS:=1.0,1.1,1.25}"
python3 select_best.py --runs multirun/ --out selection/ --tols "${TOLS}"
