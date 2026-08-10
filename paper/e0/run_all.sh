#!/usr/bin/env bash
# Run every E0 (validation) experiment.
#
#   bash paper/e0/run_all.sh            # full run, writes paper/results/e0*/
#   bash paper/e0/run_all.sh --quick    # seconds; artifacts go to a temp dir so a
#                                       # smoke run cannot overwrite real results
#   bash paper/e0/run_all.sh --check    # full run, non-zero exit on a failed
#                                       # prediction
#
# Flags are passed through to each script, so --quick --check works too.
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
for script in paper/e0/a_multipliers.py paper/e0/b_nonopt.py paper/e0/d_distributed.py; do
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
