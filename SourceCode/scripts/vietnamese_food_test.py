# Zero-Shot Vietnamese Food Test Protocol

## Muc tieu

Kiem tra xem YOLOv13n da train tren ECUSTFD co the generalize sang thuc pham Viet Nam hay khong, bang cach chay inference tren tap anh thuc pham Viet (chuoi, cam, trung, ca chua, xoai) voi dong xu 25mm lam tham chieu.

## Thuc pham Viet can test

| Lop (ECUSTFD mapping) | Thuc pham Viet | Ly do chon |
|---|---|---|
| banana | Chuoi | Hinh dang gan giong banana |
| orange | Cam | Hinh dang trung tinh gan giong orange |
| egg | Trung ga | Hinh dang ellipsoid gan giong |
| tomato | Ca chua | Hinh dang trung tinh ellipsoid gan giong |
| mango | Xoai | Hinh dang ellipsoid gan giong |

## Thu tu thuc hien

### Buoc 1: Chuan bi anh

1. Chuan bi 5-10 mon thuc pham Viet (chuoi, cam, trung, ca chua, xoai)
2. Dat dong xu 25mm cung trong khung hinh (neu co the)
3. Chup anh tu phia tren (top-down view) — goi chinh xac nhat
4. Dat dong xu 25mm (neu khong co, dung tham so `--no-coin`)

### Buoc 2: Chay inference

```bash
python scripts/calorie_estimator.py \
    --source path/to/vietnamese_food_images/ \
    --weights runs/local_food_detect/output/weights/best.pt \
    --max-images 50 \
    --output runs/vietnamese_food_results/
```

### Buoc 3: Danh gia ket qua

```bash
python scripts/vietnamese_food_eval.py \
    --predictions runs/vietnamese_food_results/ \
    --ground-truth vietnamese_food_gt.csv \
    --output runs/vietnamese_food_analysis/
```

## Ket qua mong muon

- **Chuoi**: MAPE ~50-80% (nhu ECUSTFD banana)
- **Cam**: MAPE ~35-45% (nhu ECUSTFD orange)
- **Trung**: MAPE ~20-30% (tot hon banana)
- **Ca chua**: MAPE ~40-50% (nhu ECUSTFD tomato)
- **Xoai**: MAPE ~40-50% (nhu ECUSTFD mango)

## Ket qua khong mong muon

- Model khong phat hien duoc thuc pham
- MAPE > 100% cho bat ky lop nao
- Nhieu false positive

## Ghi nhan quan sat

Neu MAPE rat cao (>100%) cho tat ca cac lop:
- **Ket luan**: Domain gap giua ECUSTFD (Trung Quoc lab) va thuc pham Viet rat lon
- **De xuat**: Can train lai tren tap du lieu Viet Nam

Neu MAPE xap xi nhu ECUSTFD:
- **Ket luan**: Model co the generalize tot sang thuc pham Viet
- **De xuat**: Thu nghiem them voi nhieu loai thuc pham

## File can tao

- `scripts/vietnamese_food_eval.py` — tinh MAPE tu danh sach GT
- `vietnamese_food_gt.csv` — ground truth dinh dang:

```csv
image,class,weight_g,calorie_kcal
chuoi_001.jpg,banana,120,107
cam_001.jpg,orange,180,85
trung_001.jpg,egg,60,93
cachua_001.jpg,tomato,150,27
xoai_001.jpg,mango,200,120
```

## Noi dung script `vietnamese_food_eval.py`

```python
"""Evaluate Vietnamese food zero-shot performance."""

import csv
import json
from pathlib import Path

# Load predictions from calorie_estimator output
# Load ground truth from vietnamese_food_gt.csv
# Compute MAPE, MAE per class
# Report findings
```

## Ket luan va khuyen nghi

- Neu khop VnExpress/food blog dataset, co the mo rong thanh:
  - Dataset thuc pham Viet (50+ loai, 500+ anh)
  - Fine-tune YOLOv13 tren dataset Viet
  - Bao cao domain gap analysis

- Neu khong, ghi nhan lai han che:
  > "Model chi danh gia tren ECUSTFD (Trung Quoc lab). Zero-shot test cho thay
  > domain gap lon voi thuc pham Viet (MAPE > 100% doi voi chuoi).
  > Can co tap du lieu Viet Nam de danh gia tong quat hoa."
```

## Script: `scripts/vietnamese_food_eval.py`

```python
"""Vietnamese food zero-shot evaluation script.

Usage:
    python scripts/vietnamese_food_eval.py \\
        --predictions runs/vietnamese_food_results/food_estimates.csv \\
        --ground-truth data/vietnamese_food_gt.csv \\
        --output runs/vietnamese_food_analysis/
"""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_gt(path):
    gt = {}
    with Path(path).open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            gt[row["image"]] = row
    return gt


def load_predictions(path):
    preds = []
    with Path(path).open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            preds.append(row)
    return preds


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions")
    parser.add_argument("--ground-truth")
    parser.add_argument("--output", default="runs/vietnamese_food_analysis")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    gt = load_gt(args.ground_truth)
    preds = load_predictions(args.predictions)

    by_class = defaultdict(list)
    for pred in preds:
        img = Path(pred.get("image", "")).name
        if img not in gt:
            continue
        gt_row = gt[img]
        try:
            gt_cal = float(gt_row["calorie_kcal"])
            pred_cal = float(pred.get("calorie_kcal", 0))
            mape = abs(pred_cal - gt_cal) / gt_cal
            by_class[gt_row["class"]].append({
                "image": img,
                "gt_cal": gt_cal,
                "pred_cal": pred_cal,
                "mape": mape,
                "abs_err": abs(pred_cal - gt_cal),
            })
        except (TypeError, ValueError):
            continue

    print("=" * 60)
    print("VIETNAMESE FOOD ZERO-SHOT RESULTS")
    print("=" * 60)
    print(f"\n{'Class':<15} {'n':>4} {'MAPE':>10} {'MAE (kcal)':>12}")
    print("-" * 45)
    all_pcts = []
    all_errs = []
    for cls in sorted(by_class.keys()):
        rows = by_class[cls]
        pcts = [r["mape"] for r in rows]
        errs = [r["abs_err"] for r in rows]
        all_pcts.extend(pcts)
        all_errs.extend(errs)
        print(f"{cls:<15} {len(rows):>4} {np.mean(pcts):>10.2%} {np.mean(errs):>12.1f}")

    if all_pcts:
        overall_mape = np.mean(all_pcts)
        overall_mae = np.mean(all_errs)
        print("-" * 45)
        print(f"{'Overall':<15} {len(all_pcts):>4} {overall_mape:>10.2%} {overall_mae:>12.1f}")

    # Save results
    results = {
        "by_class": {cls: {
            "n": len(rows),
            "mape": float(np.mean([r["mape"] for r in rows])),
            "mae_kcal": float(np.mean([r["abs_err"] for r in rows])),
        } for cls, rows in by_class.items()},
        "overall": {
            "n": len(all_pcts),
            "mape": float(overall_mape),
            "mae_kcal": float(overall_mae),
        } if all_pcts else {},
    }
    (output / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {output / 'results.json'}")


if __name__ == "__main__":
    main()
```
