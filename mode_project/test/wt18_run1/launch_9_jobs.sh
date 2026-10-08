#!/bin/bash
# Launch 9 Chow-Liu search jobs (3 eta x 3 alpha) on logical CPUs 7-15.
# This machine has 16 logical CPUs (0-15); there is NO CPU 16.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs_run1

export PYTHONPATH="$PROJ"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

eta_list=(0.005 0.03 0.1)
alpha_list=(0.1 0.5 1.0)

core=7
pids=()

for eta in "${eta_list[@]}"; do
  for alpha in "${alpha_list[@]}"; do
    if [ "$core" -gt 15 ]; then
      echo "ERROR: more than 9 jobs — only logical CPUs 7-15 configured (this host has CPUs 0-15)"
      exit 1
    fi
    log="logs_run1/chow_eta${eta}_alpha${alpha}.log"
    echo "[$(date -Iseconds)] Launching eta=$eta alpha=$alpha on core $core -> $log"
    nice -n 10 taskset -c "$core" python3 -u wt18_run1_models_15.py "$eta" "$alpha" > "$log" 2>&1 &
    pids+=($!)
    core=$((core + 1))
  done
done

echo "Waiting for ${#pids[@]} jobs (PIDs: ${pids[*]})..."
fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    echo "WARN: job pid $pid failed"
    fail=1
  fi
done

echo "[$(date -Iseconds)] All search jobs finished (fail=$fail)."
ls -la "$PROJ/data/wt18_run1/results_run1/"*.csv 2>/dev/null || true
exit $fail
