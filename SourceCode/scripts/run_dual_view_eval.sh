#!/usr/bin/env bash
# Run the dual-view (top + side) calorie evaluation with YOLOv13n.
# Requires: weights/yolov13n_ecustfd_best.pt + datasets/ECUSTFD/images/test
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

if command -v python3 >/dev/null 2>&1; then
    PY="${PYTHON:-python3}"
else
    PY="${PYTHON:-python}"
fi

"$PY" -u scripts/eval_calorie_dual_view.py \
    --source datasets/ECUSTFD/images/test \
    --weights weights/yolov13n_ecustfd_best.pt \
    --density-json data/density_processed.json \
    --output runs/dual_view_eval

echo
echo "[OK] Dual-view evaluation finished. See runs/dual_view_eval/."