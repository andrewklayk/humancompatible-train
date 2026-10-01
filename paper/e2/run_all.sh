#!/usr/bin/env bash
# Run the E2 (fairness-constrained learning) experiment.
#
#   bash paper/e2/run_all.sh            # full run, writes paper/results/e2c/
#   bash paper/e2/run_all.sh --quick    # minutes; artifacts go to a temp dir so a
#                                       # smoke run cannot overwrite real results
#   bash paper/e2/run_all.sh --check    # full run, non-zero exit on a failed
#                                       # prediction
#
# Flags are passed through to the script, so --quick --check works too.
#
# NOTE: c_cadence.py reports wall-clock per epoch (B3/C3), so do not run anything
# else CPU-heavy alongside it. It also tunes a small (primal_lr, dual_lr) grid per
# cell before its measurement runs, so expect it to be slow.
#
# a_fairness.py still exists (c_cadence.py reuses its run()/METHODS/fairness.build())
# but is no longer part of the plan and is not run here.
set -euo pipefail

cd "$(dirname "$0")/../.."
ARGS=("$@")

for arg in "${ARGS[@]}"; do
  if [[ "$arg" == "--quick" ]]; then
    export HC_PAPER_RESULTS="${TMPDIR:-/tmp}/hc_paper_results_smoke"
    echo "smoke mode: artifacts -> $HC_PAPER_RESULTS"
  fi
done

status=0
script=paper/e2/c_cadence.py
echo
echo "=============================================================="
echo "  $script ${ARGS[*]:-}"
echo "=============================================================="
if ! python "$script" "${ARGS[@]:-}"; then
  status=1
  echo "!! $script reported a failure"
fi

exit "$status"
