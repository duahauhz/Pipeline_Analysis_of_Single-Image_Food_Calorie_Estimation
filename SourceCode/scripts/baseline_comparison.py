"""Baseline Comparisons — Evaluate 3 naive baselines for calorie estimation.

This script computes 3 simple baselines to contextualize the pipeline's MAPE.

Baseline 1 — Random: predict the overall mean calorie from the test set.
Baseline 2 — Class-mean (GT class): predict class mean calorie from train set.
Baseline 3 — Predicted-class mean: use detected class, not volume estimation.

Usage
-----
    python scripts/baseline_comparison.py

Output
------
    runs/baseline_comparison/
        baselines_summary.csv     # MAPE, MAE for all baselines
        comparison_table.csv      # formatted table for paper
"""

import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent.parent
DENSITY_JSON = HERE / "data" / "density_processed.json"
PRED_CSV = HERE / "runs" / "calorie_eval" / "per_image_predictions.csv"
OUTPUT_DIR = HERE / "runs" / "baseline_comparison"

# kcal per 100g lookup (same as FOOD_INFO in calorie_estimator.py)
KCAL_PER_100G = {
    "apple": 52, "banana": 89, "bread": 265, "bun": 223,
    "doughnut": 452, "egg": 155, "fired_dough_twist": 450,
    "grape": 69, "lemon": 29, "litchi": 66, "mango": 60,
    "mooncake": 420, "orange": 47, "peach": 39, "pear": 57,
    "plum": 46, "qiwi": 61, "sachima": 450, "tomato": 18,
}


def load_density():
    with DENSITY_JSON.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_predictions():
    rows = {}
    if not PRED_CSV.exists():
        print(f"[WARN] {PRED_CSV} not found; returning empty dict")
        return rows
    with PRED_CSV.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows[row["image"]] = row
    return rows


def load_train_calories(density_data):
    """Compute per-class mean calorie from train split via density.xls."""
    per_class = defaultdict(list)
    for img_id, rec in density_data["per_image"].items():
        cls = rec["class"]
        vol = rec["volume_cm3"]
        dens = rec["density_g_cm3"]
        kcal = vol * dens * KCAL_PER_100G.get(cls, 50) / 100
        per_class[cls].append(kcal)

    class_mean = {}
    for cls, vals in per_class.items():
        class_mean[cls] = sum(vals) / len(vals)
    return class_mean


def mean_calorie(predictions):
    """Overall mean calorie in test set."""
    vals = []
    for row in predictions.values():
        gt = row.get("gt_kcal", "")
        try:
            vals.append(float(gt))
        except (TypeError, ValueError):
            pass
    return sum(vals) / len(vals) if vals else 0.0


def compute_metrics(errors):
    """Compute MAE, RMSE, MAPE, Bias from a list of (pred, gt) pairs."""
    if not errors:
        return {}
    abs_errors_g = [abs(p - t) for p, t in errors]
    pct_errors = [abs(p - t) / max(t, 1e-6) for p, t in errors]
    signed_errors_g = [p - t for p, t in errors]
    n = len(errors)
    mae = sum(abs_errors_g) / n
    rmse = math.sqrt(sum(e * e for e in abs_errors_g) / n)
    mape = sum(pct_errors) / n
    bias = sum(signed_errors_g) / n
    acc20 = sum(1 for p in pct_errors if p <= 0.20) / n
    return {
        "n": n,
        "mae_g": round(mae, 2),
        "rmse_g": round(rmse, 2),
        "mape": round(mape, 4),
        "bias_g": round(bias, 2),
        "acc@20": round(acc20, 4),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    density_data = load_density()
    predictions = load_predictions()

    print(f"[baseline] Loaded {len(predictions)} predictions from {PRED_CSV}")
    print(f"[baseline] Loaded {len(density_data['per_image'])} per-image density records")

    # ── Build ground-truth lists ────────────────────────────────────────────
    # errors = list of (predicted_kcal, gt_kcal) for images with GT
    test_rows = [
        (float(r["predicted_kcal"]), float(r["gt_kcal"]))
        for r in predictions.values()
        if r.get("gt_kcal") and r.get("predicted_kcal")
           and r["gt_kcal"] != ""
           and r["predicted_kcal"] != ""
        for _ in [None]  # flatten generator
    ]
    # Rewrite to be readable:
    test_rows = []
    for r in predictions.values():
        try:
            pred = float(r["predicted_kcal"])
            gt = float(r["gt_kcal"])
            if gt > 0:
                test_rows.append((pred, gt))
        except (TypeError, ValueError):
            continue

    class_mean_kcal = load_train_calories(density_data)
    overall_mean = mean_calorie(predictions)

    print(f"[baseline] Test set: {len(test_rows)} images with GT")
    print(f"[baseline] Overall mean calorie: {overall_mean:.2f} kcal")

    # ── Baseline 1: Random ───────────────────────────────────────────────
    errors_b1 = [(overall_mean, gt) for _, gt in test_rows]
    metrics_b1 = compute_metrics(errors_b1)
    metrics_b1["name"] = "Baseline 1 (Random)"

    # ── Baseline 2: Class mean (GT class) ───────────────────────────────
    errors_b2 = []
    for pred, gt in test_rows:
        # Find the class of this prediction
        cls = None
        for r in predictions.values():
            try:
                if float(r["predicted_kcal"]) == pred:
                    cls = r.get("class", "")
                    break
            except (TypeError, ValueError):
                continue
        if cls and cls in class_mean_kcal:
            errors_b2.append((class_mean_kcal[cls], gt))
        else:
            errors_b2.append((overall_mean, gt))
    # Better: match by image
    errors_b2 = []
    for r in predictions.values():
        try:
            gt = float(r["gt_kcal"])
            cls = r.get("class", "")
            if cls and cls in class_mean_kcal:
                pred_kcal = class_mean_kcal[cls]
            else:
                pred_kcal = overall_mean
            if gt > 0:
                errors_b2.append((pred_kcal, gt))
        except (TypeError, ValueError):
            continue

    metrics_b2 = compute_metrics(errors_b2)
    metrics_b2["name"] = "Baseline 2 (Class-mean, GT class)"

    # ── Baseline 3: Predicted-class mean (no volume) ─────────────────────
    errors_b3 = []
    for r in predictions.values():
        try:
            gt = float(r["gt_kcal"])
            pred = float(r["predicted_kcal"])
            if gt <= 0:
                continue
            # Use the predicted class (class_name column)
            cls = r.get("class_name", r.get("class", ""))
            if cls and cls in class_mean_kcal:
                pred_kcal = class_mean_kcal[cls]
            else:
                pred_kcal = overall_mean
            errors_b3.append((pred_kcal, gt))
        except (TypeError, ValueError):
            continue

    metrics_b3 = compute_metrics(errors_b3)
    metrics_b3["name"] = "Baseline 3 (Predicted-class mean)"

    # ── Pipeline (full) ───────────────────────────────────────────────
    errors_pipeline = [(pred, gt) for pred, gt in test_rows]
    metrics_pipeline = compute_metrics(errors_pipeline)
    metrics_pipeline["name"] = "Pipeline (full, per-class density)"

    # ── Print summary ─────────────────────────────────────────────────
    all_metrics = [metrics_b1, metrics_b2, metrics_b3, metrics_pipeline]

    print("\n" + "=" * 80)
    print("BASELINE COMPARISON RESULTS")
    print("=" * 80)
    print(f"\n{'Method':<45} {'n':>6} {'MAE (g)':>10} {'MAPE':>10} {'Acc@20%':>10}")
    print("-" * 85)
    for m in all_metrics:
        print(f"{m['name']:<45} {m['n']:>6} {m['mae_g']:>10.2f} "
              f"{m['mape']:>10.2%} {m['acc@20']:>10.2%}")

    print("\n--- Statistical interpretation ---")
    p = metrics_pipeline["mape"]
    b1 = metrics_b1["mape"]
    b2 = metrics_b2["mape"]
    b3 = metrics_b3["mape"]
    print(f"  Pipeline MAPE = {p:.2%}")
    print(f"  Baseline 1 (Random) MAPE = {b1:.2%}")
    print(f"  Baseline 2 (Class-mean GT) MAPE = {b2:.2%}")
    print(f"  Baseline 3 (Predicted-class mean) MAPE = {b3:.2%}")
    print(f"  Pipeline vs Random: {-p/b1*100:.1f}% better" if b1 > 0 else "")
    print(f"  Pipeline vs Baseline 3: {-p/b3*100:.1f}% better" if b3 > 0 else "")

    if p >= b2:
        print("\n  WARNING: Pipeline is WORSE than class-mean baseline!")
        print("  → Volume estimation adds NO value over knowing the class alone.")
    elif p < b2 and p >= b3:
        print("\n  Pipeline beats class-mean but ≈ predicted-class mean.")
        print("  → Detector is doing the main work, volume estimation helps marginally.")
    elif p < b3:
        print("\n  Pipeline meaningfully beats predicted-class mean.")
        print("  → Volume estimation DOES contribute value.")

    # ── Save outputs ──────────────────────────────────────────────────
    summary_csv = OUTPUT_DIR / "baselines_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["name", "n", "mae_g", "rmse_g", "mape", "bias_g", "acc@20"])
        writer.writeheader()
        for m in all_metrics:
            writer.writerow({k: v for k, v in m.items() if k != "name"})
    print(f"\n[Saved] {summary_csv}")

    # Paper-ready comparison table
    table_rows = []
    for m in all_metrics:
        table_rows.append({
            "Method": m["name"],
            "$n$": m["n"],
            "MAE (g)": m["mae_g"],
            "RMSE (g)": m["rmse_g"],
            "MAPE": f"{m['mape']:.2%}",
            "Bias (g)": m["bias_g"],
            "Acc@20%": f"{m['acc@20']:.2%}",
        })

    comp_csv = OUTPUT_DIR / "comparison_table.csv"
    with comp_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=table_rows[0].keys())
        writer.writeheader()
        writer.writerows(table_rows)
    print(f"[Saved] {comp_csv}")

    # Per-class breakdown for baseline 2 vs pipeline
    per_class_comparison = defaultdict(lambda: {"n": 0, "b2_errors": [], "pipe_errors": []})
    for r in predictions.values():
        try:
            gt = float(r["gt_kcal"])
            pred_pipe = float(r["predicted_kcal"])
            cls = r.get("class", "")
            if gt <= 0 or not cls:
                continue
            b2_pred = class_mean_kcal.get(cls, overall_mean)
            per_class_comparison[cls]["n"] += 1
            per_class_comparison[cls]["b2_errors"].append(abs(b2_pred - gt))
            per_class_comparison[cls]["pipe_errors"].append(abs(pred_pipe - gt))
        except (TypeError, ValueError):
            continue

    per_cls_rows = []
    for cls in sorted(per_class_comparison.keys()):
        d = per_class_comparison[cls]
        n = d["n"]
        if n == 0:
            continue
        b2_mae = sum(d["b2_errors"]) / n
        pipe_mae = sum(d["pipe_errors"]) / n
        improvement = (b2_mae - pipe_mae) / max(b2_mae, 1e-6)
        per_cls_rows.append({
            "class": cls, "n": n,
            "MAE_baseline2": round(b2_mae, 2),
            "MAE_pipeline": round(pipe_mae, 2),
            "Improvement": f"{improvement:+.1%}",
        })

    cls_csv = OUTPUT_DIR / "per_class_baseline2_vs_pipeline.csv"
    with cls_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=per_cls_rows[0].keys())
        writer.writeheader()
        writer.writerows(per_cls_rows)
    print(f"[Saved] {cls_csv}")

    print(f"\nOutput: {OUTPUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
