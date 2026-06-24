"""Compute calorie estimates using MiDaS depth maps instead of fixed depth_ratio.

Reads bounding boxes from runs/calorie_eval/per_image_predictions.csv (reuses
the YOLO detections) and depth maps from runs/midas_depth_maps/ (.npy files).
For each food bbox, computes the mean MiDaS depth value in the region, then
derives the depth_mm via the coin scale calibration.

The key difference from the fixed-ratio pipeline:
  depth_mm = mean_depth_in_bbox / mean_depth_in_coin_bbox  *  coin_diameter_mm

This is compared against the fixed-ratio baseline and against GT volume.

Usage
-----
    python scripts/compute_midas_calorie.py \
        --detections runs/calorie_eval/per_image_predictions.csv \
        --depth-maps runs/midas_depth_maps \
        --density-json data/density_processed.json \
        --output runs/midas_calorie \
        --coin-diameter-mm 25.0

Output
------
runs/midas_calorie/
  per_image_midas.csv      # per-image results with MiDaS depth
  summary.json             # overall metrics
  comparison.json          # MiDaS vs fixed ratio
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

# Reuse helpers from calorie_estimator
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from calorie_estimator import (
    FOOD_INFO,
    image_id_from_stem,
    load_density_data,
    volume_cm3,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
INVALID_CALIBRATION_STEMS = {"mix002T(2)", "mix005S(4)"}  # same as calorie_estimator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--detections",
        type=Path,
        default=Path("runs/calorie_eval/per_image_predictions.csv"),
        help="CSV with per-image YOLO detections (from eval_calorie.py).",
    )
    parser.add_argument(
        "--depth-maps",
        type=Path,
        default=Path("runs/midas_depth_maps"),
        help="Directory containing {stem}_depth.npy files from midas_depth_inference.py.",
    )
    parser.add_argument(
        "--density-json",
        type=Path,
        default=Path("data/density_processed.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/midas_calorie"),
    )
    parser.add_argument(
        "--coin-diameter-mm",
        type=float,
        default=25.0,
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=Path("runs/calorie_eval/per_image_predictions.csv"),
        help="Path to per_image_predictions.csv from eval_calorie.",
    )
    return parser.parse_args()


def collect_images(source: Path, max_images: int) -> list[Path]:
    images = sorted(
        p for p in source.rglob("*")
        if p.is_file() and p.suffix in IMAGE_EXTS
    )
    if max_images > 0 and len(images) > max_images:
        return images[:max_images]
    return images


def load_coin_detections(csv_path: Path) -> dict[str, dict]:
    """Load coin + food detection info from eval_calorie per_image_predictions.csv.

    We need the raw bbox coordinates to compute mean depth in each region.
    Unfortunately the CSV only has summary info, not per-bbox data.
    We re-run YOLO predictions to get bboxes, then combine with MiDaS depth.
    """
    rows = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["image"]] = row
    return rows


def load_raw_detections(image_dir: Path) -> dict[str, list[dict]]:
    """Re-run YOLO to get per-bbox coords for all test images.

    Returns {image_path_str: [{class_name, bbox, width_px, height_px}, ...]}
    """
    import os
    import sys
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(".yolo-config").resolve()))
    os.environ.setdefault("YOLO_OFFLINE", "true")

    # Use local YOLOv13 repo to load custom DSC3k2 weights
    yolov13_repo = HERE / ".." / "yolov13"
    if not (yolov13_repo / "ultralytics" / "__init__.py").exists():
        raise FileNotFoundError(f"Missing YOLOv13 repo at {yolov13_repo}")
    if str(yolov13_repo) not in sys.path:
        sys.path.insert(0, str(yolov13_repo))

    from ultralytics import YOLO
    from calorie_estimator import parse_detections

    weights = HERE / ".." / "runs" / "local_food_detect" / "output" / "weights" / "best.pt"
    if not weights.exists():
        weights = HERE / ".." / "runs" / "local_food_detect" / "output" / "weights" / "last.pt"
    if not weights.exists():
        raise FileNotFoundError(f"Cannot find YOLO weights: {weights}")

    model = YOLO(str(weights))
    images = collect_images(image_dir, max_images=0)

    results = {}
    for img_path in images:
        preds = model.predict(
            source=str(img_path),
            conf=0.25,
            imgsz=640,
            verbose=False,
        )[0]
        dets = parse_detections(preds, img_path)
        results[str(img_path)] = dets
    return results


def mean_depth_in_bbox(depth_npy: Path, bbox: list[float],
                       orig_w: int, orig_h: int) -> float | None:
    """Return mean MiDaS depth value within a bbox.

    Args:
        depth_npy: Path to {stem}_depth.npy (float32, shape HxW, values relative)
        bbox: [x1, y1, x2, y2] in original image pixel coords
        orig_w, orig_h: original image dimensions (for depth map resizing)
    """
    import numpy as np
    depth = np.load(str(depth_npy))
    h, w = depth.shape[:2]

    x1, y1, x2, y2 = [float(v) for v in bbox]
    # Scale bbox to depth map dimensions
    sx = w / orig_w
    sy = h / orig_h
    bx1 = max(0, int(x1 * sx))
    by1 = max(0, int(y1 * sy))
    bx2 = min(w, int(x2 * sx))
    by2 = min(h, int(y2 * sy))

    if bx2 <= bx1 or by2 <= by1:
        return None
    region = depth[by1:by2, bx1:bx2]
    return float(region.mean())


def compute_midas_calorie(
    detections: dict[str, list[dict]],
    depth_maps_dir: Path,
    density_data: dict,
    coin_diameter_mm: float,
    eval_csv_rows: list[dict],
) -> tuple[list[dict], dict]:
    """Compute calorie estimates using MiDaS depth for all images with GT."""

    # Build lookup from image_path → gt record
    gt_lookup: dict[str, dict] = {}
    for row in eval_csv_rows:
        if row.get("has_gt") == "True":
            gt_lookup[row["image"]] = {
                "class": row["class"],
                "gt_mass_g": float(row["gt_mass_g"]) if row["gt_mass_g"] else None,
                "gt_kcal": float(row["gt_kcal"]) if row["gt_kcal"] else None,
            }

    output_rows = []
    n_ok = 0
    n_missing_depth = 0
    n_missing_coin = 0
    n_no_food = 0

    for img_path_str, dets in detections.items():
        img_path = Path(img_path_str)
        stem = img_path.stem

        if stem in INVALID_CALIBRATION_STEMS:
            continue

        depth_npy = depth_maps_dir / f"{stem}_depth.npy"
        if not depth_npy.exists():
            n_missing_depth += 1
            continue

        # Get original image dimensions from PIL
        from PIL import Image
        img = Image.open(img_path)
        orig_w, orig_h = img.size

        coins = [d for d in dets if d["class_name"] == "coin"]
        foods = [d for d in dets if d["class_name"] in FOOD_INFO]

        gt = gt_lookup.get(img_path_str)

        row = {
            "image": img_path_str,
            "image_id": image_id_from_stem(stem),
            "class": gt["class"] if gt else None,
            "has_gt": gt is not None,
            "n_food_detected": len(foods),
            "n_coin_detected": len(coins),
            "status": "ok",
            "midas_predicted_kcal": None,
            "midas_predicted_mass_g": None,
            "fixed_predicted_kcal": None,
            "fixed_predicted_mass_g": None,
            "gt_kcal": gt["gt_kcal"] if gt else None,
            "gt_mass_g": gt["gt_mass_g"] if gt else None,
        }

        if not coins:
            row["status"] = "missing_coin"
            n_missing_coin += 1
        elif not foods:
            row["status"] = "no_food"
            n_no_food += 1
        else:
            coin = max(coins, key=lambda d: d["confidence"])
            coin_bbox = coin["bbox"]
            coin_mean_depth = mean_depth_in_bbox(depth_npy, coin_bbox, orig_w, orig_h)

            if coin_mean_depth is None or coin_mean_depth <= 0:
                row["status"] = "invalid_coin_depth"
                n_missing_coin += 1
                continue

            # mm_per_pixel from coin: known diameter / pixel diameter
            coin_px = (coin["width_px"] + coin["height_px"]) / 2.0
            mm_per_pixel = coin_diameter_mm / coin_px

            # Also compute fixed-ratio baseline for comparison
            fixed_total_mass = 0.0
            fixed_total_kcal = 0.0
            midas_total_mass = 0.0
            midas_total_kcal = 0.0

            for food in foods:
                cls = food["class_name"]
                info = FOOD_INFO[cls]

                # Fixed-ratio path (same as eval_calorie baseline)
                width_mm = food["width_px"] * mm_per_pixel
                height_mm = food["height_px"] * mm_per_pixel
                fixed_depth_mm = min(width_mm, height_mm) * info["depth_ratio"]
                fixed_vol = volume_cm3(width_mm, height_mm, fixed_depth_mm, info["geometry"])
                fixed_mass = fixed_vol * info["density"]
                fixed_kcal = fixed_mass * info["kcal_per_100g"] / 100.0

                # MiDaS path: use mean depth in bbox region
                food_depth_val = mean_depth_in_bbox(depth_npy, food["bbox"], orig_w, orig_h)
                if food_depth_val is not None and food_depth_val > 0:
                    # Scale ratio: food_depth / coin_depth → proportion
                    # coin_diameter_mm is real physical size
                    # depth_ratio = food_depth / coin_depth
                    depth_ratio = food_depth_val / coin_mean_depth
                    # depth_mm = depth_ratio * coin_diameter_mm
                    # But depth_ratio here is already scaled by depth map geometry,
                    # so: depth_mm = depth_ratio * coin_diameter_mm * scale_factor
                    # Since both food and coin are in same image, ratio is scale-invariant
                    midas_depth_mm = depth_ratio * coin_diameter_mm
                    # Clamp to reasonable range [5mm, 300mm]
                    midas_depth_mm = max(5.0, min(300.0, midas_depth_mm))
                    midas_vol = volume_cm3(width_mm, height_mm, midas_depth_mm, info["geometry"])
                    midas_mass = midas_vol * info["density"]
                else:
                    # Fall back to fixed ratio if depth unavailable
                    midas_depth_mm = fixed_depth_mm
                    midas_vol = fixed_vol
                    midas_mass = fixed_mass

                midas_kcal = midas_mass * info["kcal_per_100g"] / 100.0

                fixed_total_mass += fixed_mass
                fixed_total_kcal += fixed_kcal
                midas_total_mass += midas_mass
                midas_total_kcal += midas_kcal

            row["fixed_predicted_mass_g"] = fixed_total_mass
            row["fixed_predicted_kcal"] = fixed_total_kcal
            row["midas_predicted_mass_g"] = midas_total_mass
            row["midas_predicted_kcal"] = midas_total_kcal
            n_ok += 1

        output_rows.append(row)

    # Compute metrics
    midas_errs_k = []
    fixed_errs_k = []
    midas_pcts = []
    fixed_pcts = []

    for r in output_rows:
        if not r["has_gt"] or r["midas_predicted_kcal"] is None:
            continue
        gt = r["gt_kcal"]
        if gt and gt > 0:
            midas_errs_k.append(abs(r["midas_predicted_kcal"] - gt))
            fixed_errs_k.append(abs(r["fixed_predicted_kcal"] - gt))
            midas_pcts.append(abs(r["midas_predicted_kcal"] - gt) / gt)
            fixed_pcts.append(abs(r["fixed_predicted_kcal"] - gt) / gt)

    def agg(errs):
        n = len(errs)
        if n == 0:
            return {"n": 0, "mae_kcal": None, "mape_kcal": None, "rmse_kcal": None}
        mae = sum(errs) / n
        rmse = math.sqrt(sum(e * e for e in errs) / n)
        return {"n": n, "mae_kcal": round(mae, 2), "rmse_kcal": round(rmse, 2),
                "mape_kcal": round(sum(errs) / len(errs) / sum(1 for e in errs) * len(errs), 6)}

    def agg_pct(pcts):
        n = len(pcts)
        if n == 0:
            return {"n": 0, "mape_kcal": None, "accuracy_at_20pct": None}
        mape = sum(pcts) / n
        acc20 = sum(1 for p in pcts if p <= 0.20) / n
        return {"n": n, "mape_kcal": round(mape, 4),
                "accuracy_at_20pct": round(acc20, 4)}

    summary = {
        "n_ok": n_ok,
        "n_missing_depth": n_missing_depth,
        "n_missing_coin": n_missing_coin,
        "n_no_food": n_no_food,
        "midas": {
            "mae_kcal": round(sum(midas_errs_k) / len(midas_errs_k), 2) if midas_errs_k else None,
            "rmse_kcal": round(math.sqrt(sum(e*e for e in midas_errs_k) / len(midas_errs_k)), 2) if midas_errs_k else None,
            "mape_kcal": round(sum(midas_pcts) / len(midas_pcts), 4) if midas_pcts else None,
            "accuracy_at_20pct": round(sum(1 for p in midas_pcts if p <= 0.20) / len(midas_pcts), 4) if midas_pcts else None,
            "n_images": len(midas_errs_k),
        },
        "fixed_ratio": {
            "mae_kcal": round(sum(fixed_errs_k) / len(fixed_errs_k), 2) if fixed_errs_k else None,
            "rmse_kcal": round(math.sqrt(sum(e*e for e in fixed_errs_k) / len(fixed_errs_k)), 2) if fixed_errs_k else None,
            "mape_kcal": round(sum(fixed_pcts) / len(fixed_pcts), 4) if fixed_pcts else None,
            "accuracy_at_20pct": round(sum(1 for p in fixed_pcts if p <= 0.20) / len(fixed_pcts), 4) if fixed_pcts else None,
            "n_images": len(fixed_errs_k),
        },
        "improvement": None,
    }

    if summary["midas"]["mape_kcal"] and summary["fixed_ratio"]["mape_kcal"]:
        midas_mape = summary["midas"]["mape_kcal"]
        fixed_mape = summary["fixed_ratio"]["mape_kcal"]
        improvement = (fixed_mape - midas_mape) / fixed_mape * 100
        summary["improvement"] = round(improvement, 2)

    return output_rows, summary


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # Load density data for kcal lookup
    density_data = load_density_data(args.density_json)

    # Load eval CSV rows (for GT reference)
    eval_rows = []
    if args.predictions_csv.exists():
        with args.predictions_csv.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                eval_rows.append(row)
        print(f"[midas_calorie] Loaded {len(eval_rows)} eval rows from {args.predictions_csv}")

    # Find test image directory
    test_dir = Path("datasets/ECUSTFD/images/test")
    if not test_dir.exists():
        # Try to infer from first detection path
        if eval_rows:
            first = Path(eval_rows[0]["image"])
            test_dir = first.parent

    # Load raw YOLO detections
    print("[midas_calorie] Re-running YOLO to get per-bbox coordinates...")
    t0 = time.time()
    detections = load_raw_detections(test_dir)
    print(f"[midas_calorie] YOLO done in {time.time()-t0:.1f}s ({len(detections)} images)")

    # Compute MiDaS calorie estimates
    print("[midas_calorie] Computing MiDaS-based calorie estimates...")
    rows, summary = compute_midas_calorie(
        detections=detections,
        depth_maps_dir=args.depth_maps,
        density_data=density_data,
        coin_diameter_mm=args.coin_diameter_mm,
        eval_csv_rows=eval_rows,
    )

    # Write CSV
    csv_path = args.output / "per_image_midas.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[midas_calorie] Wrote {csv_path} ({len(rows)} rows)")

    # Write summary
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[midas_calorie] Wrote {summary_path}")
    print(f"[midas_calorie] Results:")
    print(f"  MiDaS   MAPE: {summary['midas']['mape_kcal']}  MAE: {summary['midas']['mae_kcal']}  Acc@20%: {summary['midas']['accuracy_at_20pct']}")
    print(f"  Fixed   MAPE: {summary['fixed_ratio']['mape_kcal']}  MAE: {summary['fixed_ratio']['mae_kcal']}  Acc@20%: {summary['fixed_ratio']['accuracy_at_20pct']}")
    if summary["improvement"] is not None:
        print(f"  MiDaS improves MAPE by {summary['improvement']:.1f}%")


if __name__ == "__main__":
    main()
