#!/bin/bash
# Launch the E1-opt sweeps as a DEPENDENCY CHAIN: each driver starts only after the
# previous one finishes, so the sweeps run strictly one at a time (not all at once).
# Run on the LOGIN NODE -- this only SUBMITS jobs (with --parsable to capture ids) and
# returns immediately; the jobs then run under Slurm's scheduler.
#
#   Run from benchmark/new_bench/:   bash scripts/run_all_opt.sh
#   Pick the driver partition/time:  PARTITION=longcpu TIME=2-00:00:00 bash scripts/run_all_opt.sh
#
# Each driver is a long-lived process that BLOCKS while its submitit GPU array runs, so
# it needs a long walltime on a NON-fast partition (amdgpufast would kill it mid-sweep).
# These vars configure the DRIVER jobs; the GPU CHILDREN get their own partition/gres
# from conf/hydra/launcher/slurm_gpu.yaml, independently.
#
# Env (all optional):
#   PARTITION  driver partition (long/CPU); unset -> cluster default
#   TIME       driver walltime                    (default 1-00:00:00)
#   DEP        afterok (stop chain on failure) | afterany (keep going)   (default afterok)
#   ALGOS      space-separated sweep order  (default: adam alm_proj ssg pbm)
#   SELECT=0   skip the final aggregate + select_best job
# Any sweep vars (DATA, TASK, INIT_SEEDS, ...) set here propagate to the drivers via
# sbatch's default --export=ALL, e.g.:  DATA=dutch bash scripts/run_all_opt.sh
set -euo pipefail

[ -d scripts ] || { echo "run from benchmark/new_bench/ (scripts/ not found)"; exit 1; }

PARTITION=${PARTITION:-cpulong}
DEP=${DEP:-afterany}
ALGOS=${ALGOS:-"adam alm_proj ssg pbm"}
SELECT=${SELECT:-1}

# sbatch options common to every DRIVER job (a lightweight blocking loop: 1 CPU, no GPU).
sb=(--parsable --cpus-per-task=1 --mem=4G )
[ -n "${PARTITION}" ] && sb+=(--partition="${PARTITION}")

jid=""
for algo in ${ALGOS}; do
  script="scripts/E1_opt_${algo}.sh"
  if [ ! -f "${script}" ]; then
    echo "!! ${script} not found -- skipping ${algo}"
    continue
  fi
  if [ -z "${jid}" ]; then
    jid=$(sbatch "${sb[@]}" "${script}")
    echo "submitted ${algo}: job ${jid} (runs first)"
  else
    prev=${jid}
    jid=$(sbatch "${sb[@]}" --dependency="${DEP}:${prev}" "${script}")
    echo "submitted ${algo}: job ${jid} (${DEP}:${prev})"
  fi
done

# Final aggregate + select_best, after the last sweep (no GPU, quick).
if [ "${SELECT}" = "1" ] && [ -n "${jid}" ]; then
  sel_sb=(--parsable --cpus-per-task=1 --mem=4G --time=00:20:00
          --job-name=E1_opt_select --dependency="${DEP}:${jid}")
  [ -n "${PARTITION}" ] && sel_sb+=(--partition="${PARTITION}")
  sel=$(sbatch "${sel_sb[@]}" --wrap="cd '${PWD}' && bash scripts/E1_opt_select.sh")
  echo "submitted select: job ${sel} (${DEP}:${jid})"
fi

echo "chain submitted. watch:  squeue -u \$USER"
