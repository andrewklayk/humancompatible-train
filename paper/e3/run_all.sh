#!/usr/bin/env bash
# Run every E3 (sparsity-constrained LM) experiment.
#
#   bash paper/e3/run_all.sh            # full run, writes paper/results/e3/
#   bash paper/e3/run_all.sh --quick    # seconds, on the stub model and a synthetic
#                                       # shard; artifacts go to a temp dir so a smoke
#                                       # run cannot overwrite real results
#
# Flags are passed through to each script. E3 registers no predictions, so --check is
# accepted for parity and always exits 0.
#
# The real run needs a token shard from prepare_data.py and `transformers` on the path;
# see sbatch_e3.sh. NOTE: scaling.py measures throughput, so do not run anything else on
# the GPUs alongside it -- which is why these run in sequence.
set -euo pipefail

cd "$(dirname "$0")/../.."
ARGS=("$@")

for arg in "${ARGS[@]:-}"; do
  if [[ "$arg" == "--quick" ]]; then
    export HC_PAPER_RESULTS="${TMPDIR:-/tmp}/hc_paper_results_smoke"
    echo "smoke mode: artifacts -> $HC_PAPER_RESULTS"
  fi
done

status=0
for script in paper/e3/sweep.py paper/e3/scaling.py; do
  echo
  echo "=============================================================="
  echo "  $script ${ARGS[*]:-}"
  echo "=============================================================="
  # Keep going after a failure so one broken configuration does not hide the rest,
  # then exit non-zero at the end.
  if ! python "$script" "${ARGS[@]:-}"; then
    status=1
    echo "!! $script reported a failure"
  fi
done

exit "$status"
