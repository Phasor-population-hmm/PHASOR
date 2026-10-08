#!/bin/bash
# Launch 9 Chow-Liu search jobs (3 eta x 3 alpha) on CPU cores 1-9 with nice.
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

core=1
pids=()

for eta in "${eta_list[@]}"; do
  for alpha in "${alpha_list[@]}"; do
  if [ "$core" -gt 9 ]; then
    echo "ERROR: more than 9 jobs — only cores 1-9 configured"
    exit 1
  fi
  log="logs_run1/chow_eta${eta}_alpha${alpha}.log"
  echo "Launching eta=$eta alpha=$alpha on core $core -> $log"
  nice -n 10 taskset -c "$core" python3 -u wt17_wash20_models_15.py "$eta" "$alpha" > "$log" 2>&1 &
  pids+=($!)
  core=$((core + 1))
  done
done

echo "Waiting for ${#pids[@]} jobs..."
for pid in "${pids[@]}"; do
  wait "$pid" || echo "WARN: job pid $pid failed"
done

echo "All jobs finished."
ls -la "$PROJ/data/wt17_wash20/results_run1/"
