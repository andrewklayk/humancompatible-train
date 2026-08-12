#!/usr/bin/env bash
# Run every E2 (fairness-constrained learning) experiment.
#
#   bash paper/e2/run_all.sh            # full run, writes paper/results/e2*/
#   bash paper/e2/run_all.sh --quick    # minutes; artifacts go to a temp dir so a
#                                       # smoke run cannot overwrite real results
#   bash paper/e2/run_all.sh --check    # full run, non-zero exit on a failed
#                                       # prediction
#
# Flags are passed through to each script, so --quick --check works too.
#
# NOTE: a_fairness.py reports wall-clock per epoch (prediction P8), so do not run
# anything else CPU-heavy alongside it -- including b_parallel.py, which is why
# these run in sequence rather than in parallel.
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
for script in paper/e2/a_fairness.py paper/e2/b_parallel.py; do
  echo
  echo "=============================================================="
  echo "  $script ${ARGS[*]:-}"
  echo "=============================================================="
  # Keep going after a failure so one violated prediction does not hide the rest,
  # then exit non-zero at the end.
  if ! python "$script" "${ARGS[@]:-}"; then
    status=1
    echo "!! $script reported a failure"
  fi
done

exit "$status"
