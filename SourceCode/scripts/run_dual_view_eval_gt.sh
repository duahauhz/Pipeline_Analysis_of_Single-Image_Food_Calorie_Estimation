#!/usr/bin/env bash
# Re-run the GT-based dual-view evaluation on ECUSTFD.
# No GPU / YOLOv13n required -- uses ground-truth bbox labels.
#
# Usage:
#   bash scripts/run_dual_view_eval_gt.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

if command -v python3 >/dev/null 2>&1; then
    PY="${PYTHON:-python3}"
else
    PY="${PYTHON:-python}"
fi

"$PY" -u scripts/eval_calorie_dual_view_gt.py \
    --labels-root datasets/ECUSTFD/labels \
    --density-json data/density_processed.json \
    --output runs/dual_view_eval_gt

echo
echo "[OK] Dual-view (GT) evaluation finished. See runs/dual_view_eval_gt/."