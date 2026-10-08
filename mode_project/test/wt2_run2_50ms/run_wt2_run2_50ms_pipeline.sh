#!/bin/bash
# Full WT2 Run2 50 ms pipeline: 9-job search -> refit -> analysis
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"

MASTER_LOG="logs_run1/master_pipeline.log"
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "============================================================"
echo "WT2 Run2 50 ms mode_project pipeline"
echo "Project: $PROJ"
echo "Started: $(date -Iseconds)"
echo "============================================================"

export PYTHONPATH="$PROJ"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

echo ""
echo "=== STEP 1/3: Hyperparameter search (9 jobs) ==="
bash "$SCRIPT_DIR/launch_9_jobs.sh"

echo ""
echo "=== STEP 2/3: Refit best model ==="
nice -n 10 python3 -u wt2_run2_50ms_refit.py 2>&1 | tee logs_run1/refit.log

echo ""
echo "=== STEP 3/3: Final analysis + bundle.npz ==="
nice -n 10 python3 -u wt2_run2_50ms_analysis.py 2>&1 | tee logs_run1/analysis.log

BUNDLE="$PROJ/data/wt2_run2_50ms/results_run1/final_analysis/bundle.npz"
echo ""
echo "Bundle: $BUNDLE"
echo "Done: $(date -Iseconds)"
echo "Master log: $MASTER_LOG"
