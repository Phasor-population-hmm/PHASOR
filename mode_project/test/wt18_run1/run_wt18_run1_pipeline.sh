#!/bin/bash
# Full wt18_run1 pipeline: 9-job search -> refit -> analysis -> mirror bundle
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"

MASTER_LOG="logs_run1/master_pipeline.log"
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "============================================================"
echo "WT18 Run1 mode_project pipeline"
echo "Project: $PROJ"
echo "Started: $(date -Iseconds)"
echo "============================================================"

export PYTHONPATH="$PROJ"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

echo ""
echo "=== STEP 1/3: Hyperparameter search (cores 8-16) ==="
bash "$SCRIPT_DIR/launch_9_jobs.sh"

echo ""
echo "=== STEP 2/3: Refit best model ==="
nice -n 10 python3 -u wt18_run1_refit.py 2>&1 | tee logs_run1/refit.log

echo ""
echo "=== STEP 3/3: Final analysis + bundle.npz ==="
nice -n 10 python3 -u wt18_run1_analysis.py 2>&1 | tee logs_run1/analysis.log

BUNDLE="$PROJ/data/wt18_run1/results_run1/final_analysis/bundle.npz"
if [ -n "$PHASOR_BUNDLE_MIRROR" ] && [ -f "$BUNDLE" ]; then
  mkdir -p "$PHASOR_BUNDLE_MIRROR"
  cp -f "$BUNDLE" "$PHASOR_BUNDLE_MIRROR/bundle.npz"
  echo "Mirrored bundle -> $PHASOR_BUNDLE_MIRROR/bundle.npz"
fi

echo ""
echo "Done: $(date -Iseconds)"
echo "Master log: $MASTER_LOG"
