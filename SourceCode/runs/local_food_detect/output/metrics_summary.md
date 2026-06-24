# YOLOv13n ECUSTFD Local Training Summary

## Training

- Run directory: `runs/local_food_detect/yolov13n_ecustfd_local`
- Epochs completed: 100
- Device: NVIDIA GeForce RTX 4050 Laptop GPU
- Image size: 640
- Batch size: 4
- AMP: false
- Weights:
  - `weights/best.pt`
  - `weights/last.pt`

## Validation Metrics

Final validation metrics from epoch 100 in `results.csv`:

| Metric | Value |
|---|---:|
| Precision | 0.99447 |
| Recall | 0.99734 |
| mAP50 | 0.99234 |
| mAP75 | 0.99038 |
| mAP50-95 | 0.91368 |

## Test Metrics

Metrics from `best.pt` on the test split:

| Metric | Value |
|---|---:|
| Precision | 0.964080 |
| Recall | 0.924766 |
| mAP50 | 0.962435 |
| mAP50-95 | 0.872137 |

## Main Artifacts

- `results.csv`
- `results.png`
- `confusion_matrix.png`
- `confusion_matrix_normalized.png`
- `PR_curve.png`
- `P_curve.png`
- `R_curve.png`
- `F1_curve.png`
- `test_eval/`

