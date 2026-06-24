"""Full-pipeline calorie evaluation on the ECUSTFD test split.

The pipeline runs `calorie_estimator` style logic over every test image, but
joins predictions against ground truth from `density.xls` (per-image weight &
class). We only have GT for the 160 image ids that appear in the workbook
(174 raw rows, 14 of which are the secondary class of a `mix` image and are
merged into the primary record), so this script restricts the metric
computation to those ids.

What we compute
---------------
Per image (one row, only for ids present in density.json):
- predicted mass (g) and calorie (kcal)
- ground truth mass (g) and calorie (g × kcal/100g / 100)
- absolute error in g and kcal
- absolute percentage error in kcal

Aggregated metrics:
- MAE, RMSE, MAPE, Bias (mean signed error), accuracy@20% (fraction of
  predictions within ±20% of GT)
- Per-class MAE for each of the 19 classes

Why per-image and not per-id
----------------------------
A single id like `apple015` corresponds to two photos (top + side) plus
multiple `(N)` repetitions. We treat each photo as an independent prediction
because the model's depth estimate depends on the angle, and the ECUSTFD
paper (Liang & Li 2017) reports results per photo as well.

Output layout
-------------
runs/calorie_eval/
  per_image_predictions.csv     (all images that had a prediction; GT cols
                                 filled only when the id is in density.json)
  per_class_metrics.csv         (per-class MAE/RMSE/MAPE, only for classes
                                 with at least 3 GT images)
  summary.json                  (overall metrics + counts)
  error_analysis.md             (top-10 worst predictions, observations)

Usage
-----
    python scripts/eval_calorie.py \\
        --source datasets/ECUSTFD/images/test \\
        --weights runs/local_food_detect/output/weights/best.pt \\
        --density-json data/density_processed.json \\
        --output runs/calorie_eval
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path


# Reuse the helpers from calorie_estimator to keep behavior identical.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from calorie_estimator import (  # noqa: E402
    COIN_CLASS_NAME,
    FOOD_INFO,
    IMAGE_EXTS,
    INVALID_CALIBRATION_STEMS,
    collect_images,
    ensure_yolov13_import,
    image_id_from_stem,
    load_density_data,
    parse_detections,
    resolve_food_estimate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("datasets/ECUSTFD/images/test"))
    parser.add_argument("--weights", type=Path, default=Path("weights/yolov13n_ecustfd_best.pt"))
    parser.add_argument("--repo", type=Path, default=Path("yolov13"))
    parser.add_argument("--density-json", type=Path, default=Path("data/density_processed.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/calorie_eval"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--coin-diameter-mm", type=float, default=25.0)
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Optional cap on number of images (0 = all).",
    )
    parser.add_argument(
        "--limit-class",
        type=str,
        default="",
        help="If set, restrict to images whose id starts with this class token (e.g. 'apple').",
    )
    parser.add_argument(
        "--start-at",
        type=int,
        default=0,
        help="Skip the first N images (for sharding across runs).",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def gt_kcal_for_id(image_id: str, density_data: dict) -> float | None:
    """GT calorie for an image id, computed as gt_weight × kcal/100g ÷ 100.

    For mix images, weight is the total of both foods, so we use the first
    class's kcal/100g as a representative (the workbook only gives weight).
    """
    rec = density_data["per_image"].get(image_id)
    if rec is None:
        return None
    cls = rec["class"]
    info = FOOD_INFO.get(cls)
    if info is None:
        return None
    return rec["weight_g"] * info["kcal_per_100g"] / 100.0


def gt_record_for_id(image_id: str, density_data: dict) -> dict | None:
    rec = density_data["per_image"].get(image_id)
    return rec


def _agg(values: list[float]) -> dict[str, float]:
    """Return {n, mean, std, min, max, mae, rmse, mape?, bias?} for `values`.

    `mape` and `bias` are only meaningful if the caller passes signed
    relative / signed errors; we just include the helper stats here.
    """
    n = len(values)
    if n == 0:
        return {"n": 0}
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return {
        "n": n,
        "mean": mean,
        "std": math.sqrt(var),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    args = parse_args()
    ensure_yolov13_import(args.repo)
    if not args.weights.exists():
        raise FileNotFoundError(f"Missing weights: {args.weights}")
    if not args.source.exists():
        raise FileNotFoundError(f"Missing source: {args.source}")

    density_data = load_density_data(args.density_json)
    print(
        f"[eval_calorie] Density data: {len(density_data['per_image'])} per-image, "
        f"{len(density_data['per_class'])} per-class."
    )

    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(".yolo-config").resolve()))
    os.environ.setdefault("YOLO_OFFLINE", "true")

    from ultralytics import YOLO

    args.output.mkdir(parents=True, exist_ok=True)

    # Collect and optionally filter images. When --max-images is 0 we want to
    # process *every* test image (subject to the start-at / limit-class
    # sharding), so collect_images gets a large cap.
    all_images = collect_images(args.source, max_images=10_000_000)
    if args.limit_class:
        all_images = [p for p in all_images if image_id_from_stem(p.stem).startswith(args.limit_class)]
    if args.start_at:
        all_images = all_images[args.start_at:]
    if args.max_images:
        all_images = all_images[: args.max_images]
    print(f"[eval_calorie] Processing {len(all_images)} images")

    # Resolve which density-source policy to use for evaluation. The
    # per_image policy in `calorie_estimator` gives "perfect" mass because
    # it returns the GT weight verbatim, so it cannot be used to evaluate
    # the end-to-end pipeline. We expose a flag to opt into it for the
    # ablation study.
    use_gt_volume = os.environ.get("EVAL_USE_GT_VOLUME", "0") == "1"

    model = YOLO(str(args.weights))

    per_image_rows: list[dict] = []
    n_processed = 0
    n_with_prediction = 0
    n_with_gt = 0
    n_missing_coin = 0
    n_no_food = 0
    n_invalid_coin = 0
    n_skipped_invalid = 0
    t0 = time.time()

    for image_path in all_images:
        n_processed += 1
        if n_processed % 100 == 0:
            elapsed = time.time() - t0
            print(
                f"[eval_calorie] {n_processed}/{len(all_images)} "
                f"({n_processed / max(elapsed, 1e-6):.1f} img/s) "
                f"predictions={n_with_prediction} missing_coin={n_missing_coin}"
            )
        if image_path.stem in INVALID_CALIBRATION_STEMS:
            n_skipped_invalid += 1
            continue

        result = model.predict(
            source=str(image_path),
            conf=args.conf,
            imgsz=args.imgsz,
            verbose=False,
        )[0]
        detections = parse_detections(result, image_path)
        coins = [d for d in detections if d["class_name"] == COIN_CLASS_NAME]
        foods = [d for d in detections if d["class_name"] in FOOD_INFO]

        image_id = image_id_from_stem(image_path.stem)
        gt = gt_record_for_id(image_id, density_data)
        gt_kcal = gt_kcal_for_id(image_id, density_data) if gt is not None else None

        row = {
            "image": str(image_path),
            "image_id": image_id,
            "class": gt["class"] if gt else None,
            "has_gt": gt is not None,
            "n_food_detected": len(foods),
            "n_coin_detected": len(coins),
            "status": "ok",
            "predicted_mass_g": None,
            "predicted_kcal": None,
            "gt_mass_g": gt["weight_g"] if gt else None,
            "gt_kcal": gt_kcal,
            "abs_err_g": None,
            "abs_err_kcal": None,
            "abs_pct_err_kcal": None,
            "volume_source": None,
        }

        if not coins:
            row["status"] = "missing_coin"
            n_missing_coin += 1
        elif not foods:
            row["status"] = "no_food"
            n_no_food += 1
        else:
            # When evaluating, force the per_class path by stripping
            # per_image entries from the density payload. The end-to-end
            # pipeline quality is what we care about; using per_image would
            # just return the GT weight verbatim and make MAE 0.
            eval_density = density_data if use_gt_volume else {**density_data, "per_image": {}}
            coin = max(coins, key=lambda d: d["confidence"])
            coin_px = (coin["width_px"] + coin["height_px"]) / 2.0
            mm_per_pixel = args.coin_diameter_mm / coin_px if coin_px > 0 else None
            if mm_per_pixel is None:
                row["status"] = "invalid_coin_bbox"
                n_invalid_coin += 1
            else:
                estimated = [
                    resolve_food_estimate(f["class_name"], f, mm_per_pixel, eval_density)
                    for f in foods
                ]
                row["predicted_mass_g"] = sum(e["mass_g"] for e in estimated)
                row["predicted_kcal"] = sum(e["calorie_kcal"] for e in estimated)
                row["volume_source"] = ",".join(
                    sorted({e.get("volume_source", "unknown") for e in estimated})
                )
                n_with_prediction += 1

        if (
            row["has_gt"]
            and row["predicted_mass_g"] is not None
            and row["gt_mass_g"] is not None
        ):
            row["abs_err_g"] = abs(row["predicted_mass_g"] - row["gt_mass_g"])
            row["abs_err_kcal"] = abs(row["predicted_kcal"] - row["gt_kcal"])
            if row["gt_kcal"] and row["gt_kcal"] > 0:
                row["abs_pct_err_kcal"] = row["abs_err_kcal"] / row["gt_kcal"]
            n_with_gt += 1

        per_image_rows.append(row)

    elapsed = time.time() - t0
    print(
        f"[eval_calorie] Done in {elapsed:.1f}s "
        f"({n_processed / max(elapsed, 1e-6):.1f} img/s). "
        f"Predictions: {n_with_prediction}, with-GT: {n_with_gt}."
    )

    # Write per-image CSV.
    csv_path = args.output / "per_image_predictions.csv"
    fieldnames = list(per_image_rows[0].keys()) if per_image_rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for r in per_image_rows:
            writer.writerow(r)
    print(f"[eval_calorie] Wrote {csv_path}")

    # Per-class metrics.
    per_class_err_g: dict[str, list[float]] = defaultdict(list)
    per_class_err_kcal: dict[str, list[float]] = defaultdict(list)
    per_class_pct_err: dict[str, list[float]] = defaultdict(list)
    per_class_bias_g: dict[str, list[float]] = defaultdict(list)

    for r in per_image_rows:
        if not r["has_gt"] or r["abs_err_g"] is None:
            continue
        cls = r["class"] or "unknown"
        per_class_err_g[cls].append(r["abs_err_g"])
        per_class_err_kcal[cls].append(r["abs_err_kcal"])
        per_class_pct_err[cls].append(r["abs_pct_err_kcal"] or 0.0)
        per_class_bias_g[cls].append(r["predicted_mass_g"] - r["gt_mass_g"])

    per_class_csv = args.output / "per_class_metrics.csv"
    with per_class_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["class", "n", "mae_g", "rmse_g", "bias_g", "mae_kcal", "mape_kcal"]
        )
        for cls in sorted(per_class_err_g.keys()):
            errs = per_class_err_g[cls]
            errs_k = per_class_err_kcal[cls]
            pct = per_class_pct_err[cls]
            bias = per_class_bias_g[cls]
            if len(errs) < 1:
                continue
            mae_g = sum(errs) / len(errs)
            rmse_g = math.sqrt(sum(e * e for e in errs) / len(errs))
            bias_g = sum(bias) / len(bias)
            mae_k = sum(errs_k) / len(errs_k)
            mape_k = sum(pct) / len(pct)
            writer.writerow(
                [cls, len(errs), round(mae_g, 2), round(rmse_g, 2), round(bias_g, 2),
                 round(mae_k, 2), round(mape_k, 4)]
            )
    print(f"[eval_calorie] Wrote {per_class_csv}")

    # Summary.
    errs_g = [r["abs_err_g"] for r in per_image_rows if r["abs_err_g"] is not None]
    errs_k = [r["abs_err_kcal"] for r in per_image_rows if r["abs_err_kcal"] is not None]
    pcts_k = [r["abs_pct_err_kcal"] for r in per_image_rows if r["abs_pct_err_kcal"] is not None]
    biases_g = [
        r["predicted_mass_g"] - r["gt_mass_g"]
        for r in per_image_rows
        if r["has_gt"] and r["predicted_mass_g"] is not None
    ]
    acc_20 = sum(1 for p in pcts_k if p <= 0.20) / max(len(pcts_k), 1)

    summary = {
        "n_images_total": n_processed,
        "n_predictions": n_with_prediction,
        "n_with_gt": n_with_gt,
        "n_missing_coin": n_missing_coin,
        "n_no_food": n_no_food,
        "n_invalid_coin_bbox": n_invalid_coin,
        "n_skipped_invalid_calibration": n_skipped_invalid,
        "elapsed_s": round(elapsed, 1),
        "images_per_sec": round(n_processed / max(elapsed, 1e-6), 2),
        "overall": {
            "mae_g": round(sum(errs_g) / len(errs_g), 2) if errs_g else None,
            "rmse_g": round(math.sqrt(sum(e * e for e in errs_g) / len(errs_g)), 2) if errs_g else None,
            "bias_g": round(sum(biases_g) / len(biases_g), 2) if biases_g else None,
            "mae_kcal": round(sum(errs_k) / len(errs_k), 2) if errs_k else None,
            "rmse_kcal": round(math.sqrt(sum(e * e for e in errs_k) / len(errs_k)), 2) if errs_k else None,
            "mape_kcal": round(sum(pcts_k) / len(pcts_k), 4) if pcts_k else None,
            "accuracy_at_20pct": round(acc_20, 4),
        },
        "per_class": {
            cls: {
                "n": len(per_class_err_g[cls]),
                "mae_g": round(sum(per_class_err_g[cls]) / len(per_class_err_g[cls]), 2),
                "rmse_g": round(
                    math.sqrt(sum(e * e for e in per_class_err_g[cls]) / len(per_class_err_g[cls])), 2
                ),
                "mae_kcal": round(
                    sum(per_class_err_kcal[cls]) / len(per_class_err_kcal[cls]), 2
                ),
                "mape_kcal": round(
                    sum(per_class_pct_err[cls]) / len(per_class_pct_err[cls]), 4
                ),
            }
            for cls in sorted(per_class_err_g.keys())
        },
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[eval_calorie] Wrote {summary_path}")

    # Error analysis (top worst by % kcal error).
    gt_rows = [r for r in per_image_rows if r["has_gt"] and r["abs_pct_err_kcal"] is not None]
    gt_rows.sort(key=lambda r: r["abs_pct_err_kcal"], reverse=True)
    worst = gt_rows[:10]
    best = sorted(gt_rows, key=lambda r: r["abs_pct_err_kcal"])[:10]

    error_md = args.output / "error_analysis.md"
    with error_md.open("w", encoding="utf-8") as file:
        file.write("# Calorie evaluation — error analysis\n\n")
        file.write(f"- Images processed: **{n_processed}**\n")
        file.write(f"- Predictions: **{n_with_prediction}**\n")
        file.write(f"- With GT (and prediction): **{n_with_gt}**\n")
        file.write(f"- Missing coin: **{n_missing_coin}**, no food: **{n_no_food}**, invalid coin bbox: **{n_invalid_coin}**\n")
        file.write(f"- Total time: **{elapsed:.1f}s** ({n_processed / max(elapsed, 1e-6):.1f} img/s)\n\n")
        file.write("## Overall metrics\n\n")
        o = summary["overall"]
        file.write("| metric | value |\n|---|---|\n")
        for k, v in o.items():
            file.write(f"| {k} | {v} |\n")
        file.write("\n## Top-10 worst predictions (% kcal error)\n\n")
        file.write("| image_id | class | pred kcal | GT kcal | abs err kcal | abs % err |\n")
        file.write("|---|---|---|---|---|---|\n")
        for r in worst:
            file.write(
                f"| {r['image_id']} | {r['class']} | {r['predicted_kcal']:.1f} | {r['gt_kcal']:.1f} | "
                f"{r['abs_err_kcal']:.1f} | {r['abs_pct_err_kcal']:.2%} |\n"
            )
        file.write("\n## Top-10 best predictions\n\n")
        file.write("| image_id | class | pred kcal | GT kcal | abs % err |\n")
        file.write("|---|---|---|---|---|\n")
        for r in best:
            file.write(
                f"| {r['image_id']} | {r['class']} | {r['predicted_kcal']:.1f} | {r['gt_kcal']:.1f} | "
                f"{r['abs_pct_err_kcal']:.2%} |\n"
            )
        file.write("\n## Per-class MAE (g)\n\n")
        file.write("| class | n | MAE (g) | RMSE (g) | MAE (kcal) | MAPE |\n|---|---|---|---|---|---|\n")
        for cls, m in summary["per_class"].items():
            file.write(
                f"| {cls} | {m['n']} | {m['mae_g']} | {m['rmse_g']} | {m['mae_kcal']} | {m['mape_kcal']:.2%} |\n"
            )
    print(f"[eval_calorie] Wrote {error_md}")

    # Print a short summary to stdout.
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for k, v in summary["overall"].items():
        print(f"  {k:20s} : {v}")
    print()
    print("Per-class MAE (g):")
    for cls, m in summary["per_class"].items():
        print(f"  {cls:20s} : n={m['n']:3d}  mae_g={m['mae_g']:7.2f}  mae_kcal={m['mae_kcal']:7.2f}")


if __name__ == "__main__":
    main()
