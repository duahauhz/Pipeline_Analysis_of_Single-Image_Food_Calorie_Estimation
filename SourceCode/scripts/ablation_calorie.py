"""Calorie pipeline ablation on ECUSTFD test split.

We compare three density-source policies for the geometric mass estimator
in `calorie_estimator.resolve_food_estimate`:

  1. **per_image**  — when the image id is in density.xls, use the
     workbook's exact volume & weight (sanity check / upper bound).
  2. **per_class**  — use the class mean density from density.xls
     combined with the geometric model. Current production default.
  3. **geometry_fallback** — ignore density.xls entirely and use the
     hard-coded `FOOD_INFO` density + depth ratio. Baseline before we
     introduced density.xls.

For each policy we recompute the per-image predictions, then aggregate
MAE/RMSE/MAPE on the 160 image ids that have GT. Output goes to
`runs/calorie_ablation/`.

Why this matters for the paper
-----------------------------
- per_image vs per_class: shows the gap between perfect GT lookup and the
  realistic per-class prior (most production food-image systems will not
  have per-image ground truth).
- per_class vs geometry_fallback: isolates the contribution of replacing
  the hard-coded density/depth values with values measured from
  density.xls.

Usage
-----
    python scripts/ablation_calorie.py \\
        --weights runs/local_food_detect/output/weights/best.pt \\
        --output runs/calorie_ablation
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

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

POLICIES = ["per_image", "per_class", "geometry_fallback"]


def predict_for_policy(
    images: list[Path],
    density_data: dict,
    weights_path: Path,
    repo_dir: Path,
    conf: float,
    imgsz: int,
    coin_diameter_mm: float,
    policy: str,
) -> list[dict]:
    """Run the full pipeline with the given density-source policy.

    `policy` is forwarded to `resolve_food_estimate` via the density
    payload: per_image entries are kept/removed depending on the policy.
    """
    from ultralytics import YOLO

    if policy == "per_image":
        eval_density = density_data
    elif policy == "per_class":
        eval_density = {**density_data, "per_image": {}}
    elif policy == "geometry_fallback":
        eval_density = {"per_image": {}, "per_class": {}, "source_file": "synthetic", "classes": []}
    else:
        raise ValueError(f"Unknown policy: {policy}")

    model = YOLO(str(weights_path))
    rows: list[dict] = []
    t0 = time.time()
    for i, image_path in enumerate(images):
        if image_path.stem in INVALID_CALIBRATION_STEMS:
            continue
        if (i + 1) % 200 == 0:
            print(f"  [{policy}] {i+1}/{len(images)} ({(i+1)/max(time.time()-t0,1e-6):.1f} img/s)")
        result = model.predict(
            source=str(image_path), conf=conf, imgsz=imgsz, verbose=False
        )[0]
        detections = parse_detections(result, image_path)
        coins = [d for d in detections if d["class_name"] == COIN_CLASS_NAME]
        foods = [d for d in detections if d["class_name"] in FOOD_INFO]
        image_id = image_id_from_stem(image_path.stem)
        gt = density_data["per_image"].get(image_id)

        row = {
            "image": str(image_path),
            "image_id": image_id,
            "class": gt["class"] if gt else None,
            "has_gt": gt is not None,
            "policy": policy,
            "predicted_mass_g": None,
            "predicted_kcal": None,
            "gt_mass_g": gt["weight_g"] if gt else None,
            "gt_kcal": None,
            "abs_err_g": None,
            "abs_err_kcal": None,
            "abs_pct_err_kcal": None,
        }
        if gt is not None and gt["class"] in FOOD_INFO:
            row["gt_kcal"] = gt["weight_g"] * FOOD_INFO[gt["class"]]["kcal_per_100g"] / 100.0

        if not coins or not foods:
            rows.append(row)
            continue

        coin = max(coins, key=lambda d: d["confidence"])
        coin_px = (coin["width_px"] + coin["height_px"]) / 2.0
        mm_per_pixel = coin_diameter_mm / coin_px if coin_px > 0 else None
        if mm_per_pixel is None:
            rows.append(row)
            continue

        estimated = [
            resolve_food_estimate(f["class_name"], f, mm_per_pixel, eval_density)
            for f in foods
        ]
        row["predicted_mass_g"] = sum(e["mass_g"] for e in estimated)
        row["predicted_kcal"] = sum(e["calorie_kcal"] for e in estimated)

        if row["has_gt"] and row["predicted_mass_g"] is not None and row["gt_mass_g"] is not None:
            row["abs_err_g"] = abs(row["predicted_mass_g"] - row["gt_mass_g"])
            row["abs_err_kcal"] = abs(row["predicted_kcal"] - row["gt_kcal"])
            if row["gt_kcal"] and row["gt_kcal"] > 0:
                row["abs_pct_err_kcal"] = row["abs_err_kcal"] / row["gt_kcal"]

        rows.append(row)
    print(f"  [{policy}] Done in {time.time()-t0:.1f}s")
    return rows


def aggregate(rows: list[dict]) -> dict:
    errs_g = [r["abs_err_g"] for r in rows if r["abs_err_g"] is not None]
    errs_k = [r["abs_err_kcal"] for r in rows if r["abs_err_kcal"] is not None]
    pcts = [r["abs_pct_err_kcal"] for r in rows if r["abs_pct_err_kcal"] is not None]
    biases = [
        r["predicted_mass_g"] - r["gt_mass_g"]
        for r in rows
        if r["has_gt"] and r["predicted_mass_g"] is not None
    ]
    acc_20 = sum(1 for p in pcts if p <= 0.20) / max(len(pcts), 1)
    return {
        "n": len(rows),
        "n_with_gt": len(errs_g),
        "mae_g": round(sum(errs_g) / len(errs_g), 2) if errs_g else None,
        "rmse_g": round(math.sqrt(sum(e * e for e in errs_g) / len(errs_g)), 2) if errs_g else None,
        "bias_g": round(sum(biases) / len(biases), 2) if biases else None,
        "mae_kcal": round(sum(errs_k) / len(errs_k), 2) if errs_k else None,
        "rmse_kcal": round(math.sqrt(sum(e * e for e in errs_k) / len(errs_k)), 2) if errs_k else None,
        "mape_kcal": round(sum(pcts) / len(pcts), 4) if pcts else None,
        "accuracy_at_20pct": round(acc_20, 4),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("datasets/ECUSTFD/images/test"))
    parser.add_argument("--weights", type=Path, default=Path("weights/yolov13n_ecustfd_best.pt"))
    parser.add_argument("--repo", type=Path, default=Path("yolov13"))
    parser.add_argument("--density-json", type=Path, default=Path("data/density_processed.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/calorie_ablation"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--coin-diameter-mm", type=float, default=25.0)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--policies", nargs="+", default=POLICIES, choices=POLICIES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_yolov13_import(args.repo)
    if not args.weights.exists():
        raise FileNotFoundError(f"Missing weights: {args.weights}")
    density_data = load_density_data(args.density_json)
    print(
        f"[ablation] Density data: {len(density_data['per_image'])} per-image, "
        f"{len(density_data['per_class'])} per-class."
    )

    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(".yolo-config").resolve()))
    os.environ.setdefault("YOLO_OFFLINE", "true")

    images = collect_images(args.source, max_images=10_000_000)
    if args.max_images:
        images = images[: args.max_images]
    print(f"[ablation] {len(images)} images, policies={args.policies}")

    args.output.mkdir(parents=True, exist_ok=True)

    all_rows: dict[str, list[dict]] = {}
    summaries: dict[str, dict] = {}
    for policy in args.policies:
        rows = predict_for_policy(
            images,
            density_data,
            args.weights,
            args.repo,
            args.conf,
            args.imgsz,
            args.coin_diameter_mm,
            policy,
        )
        all_rows[policy] = rows
        summaries[policy] = aggregate(rows)

    # Per-policy CSV.
    for policy, rows in all_rows.items():
        path = args.output / f"predictions_{policy}.csv"
        with path.open("w", newline="", encoding="utf-8") as file:
            if not rows:
                continue
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"[ablation] Wrote {path}")

    # Comparison table.
    summary_path = args.output / "comparison.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"[ablation] Wrote {summary_path}")

    # Per-class comparison: for each class, how does each policy do?
    per_class: dict[str, dict[str, dict]] = defaultdict(dict)
    for policy, rows in all_rows.items():
        by_class: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            if r["class"] is None or r["abs_err_g"] is None:
                continue
            by_class[r["class"]].append(r["abs_err_g"])
        for cls, errs in by_class.items():
            per_class[cls][policy] = {
                "n": len(errs),
                "mae_g": round(sum(errs) / len(errs), 2),
            }
    pc_path = args.output / "per_class_comparison.csv"
    with pc_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        header = ["class"]
        for policy in args.policies:
            header += [f"n_{policy}", f"mae_g_{policy}"]
        writer.writerow(header)
        for cls in sorted(per_class.keys()):
            row = [cls]
            for policy in args.policies:
                entry = per_class[cls].get(policy, {})
                row.append(entry.get("n", ""))
                row.append(entry.get("mae_g", ""))
            writer.writerow(row)
    print(f"[ablation] Wrote {pc_path}")

    # Markdown summary.
    md = args.output / "ablation_summary.md"
    with md.open("w", encoding="utf-8") as file:
        file.write("# Calorie pipeline ablation\n\n")
        file.write(f"- Weights: `{args.weights}`\n")
        file.write(f"- Source: `{args.source}`\n")
        file.write(f"- Images: {len(images)}\n")
        file.write(f"- Density source: `{args.density_json}`\n\n")
        file.write("## Overall metrics\n\n")
        file.write("| policy | n | MAE (g) | RMSE (g) | bias (g) | MAE (kcal) | MAPE | acc@20% |\n")
        file.write("|---|---|---|---|---|---|---|---|\n")
        for policy, m in summaries.items():
            file.write(
                f"| {policy} | {m['n_with_gt']} | {m['mae_g']} | {m['rmse_g']} | {m['bias_g']} | "
                f"{m['mae_kcal']} | {m['mape_kcal']:.2%} | {m['accuracy_at_20pct']:.2%} |\n"
            )
        file.write("\n## Per-class MAE (g)\n\n")
        file.write("| class | n | per_image | per_class | geometry_fallback |\n")
        file.write("|---|---|---|---|---|\n")
        for cls in sorted(per_class.keys()):
            cells = [cls]
            for policy in args.policies:
                e = per_class[cls].get(policy, {})
                cells.append(str(e.get("n", "")))
                cells.append(str(e.get("mae_g", "")))
            file.write("| " + " | ".join(cells) + " |\n")
    print(f"[ablation] Wrote {md}")

    # Print to stdout.
    print()
    print("=" * 60)
    print("ABLATION SUMMARY")
    print("=" * 60)
    for policy, m in summaries.items():
        print(f"\n[{policy}]")
        for k, v in m.items():
            if isinstance(v, float) and k != "n" and k != "n_with_gt":
                print(f"  {k:20s} : {v:.4f}")
            else:
                print(f"  {k:20s} : {v}")


if __name__ == "__main__":
    main()
