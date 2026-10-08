#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# launch_9_jobs.sh — wt2_run2 100ms
#
# No core pinning. Set your own before calling:
#   nice -n 10 bash launch_9_jobs.sh
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs_wt2_run2_100ms

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

eta_list=(0.005 0.03 0.1)
alpha_list=(0.1 0.5 1.0)

job=0
for eta in "${eta_list[@]}"; do
  for alpha in "${alpha_list[@]}"; do
    echo "Launching job $job: eta=$eta  alpha=$alpha  [100ms]"
    python3 -u wt2_run2_models_100ms.py "$eta" "$alpha" \
      > logs_wt2_run2_100ms/chow_eta${eta}_alpha${alpha}.log 2>&1 &
    job=$((job + 1))
  done
done

echo "All 9 jobs launched. Waiting..."
wait
echo "All 9 jobs finished (wt2_run2 100ms) — $(date)"
