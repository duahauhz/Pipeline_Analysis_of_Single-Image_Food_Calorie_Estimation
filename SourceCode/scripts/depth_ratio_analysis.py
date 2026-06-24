"""Depth Ratio Analysis — Derive depth ratios from density.xls ground truth.

This script computes the actual depth ratio for each image in ECUSTFD from the
ground-truth volume and bounding-box dimensions, then:

1. Reports per-class depth-ratio statistics (median, std, IQR, n).
2. Compares learned ratios vs the hard-coded ratios currently in Table 7.
3. Draws a boxplot of per-class depth-ratio distributions.
4. Runs 5-fold cross-validation: derive ratio from 4 folds, test on the 5th.
5. Runs sensitivity analysis: perturb ratio ±10%, ±20%, ±30% and measure MAPE.

Usage
-----
    python scripts/depth_ratio_analysis.py

Output
------
    runs/depth_ratio_analysis/
        depth_ratio_stats.json       # per-class stats
        depth_ratio_comparison.csv  # learned vs hard-coded
        depth_ratio_boxplot.png     # boxplot by class
        crossval_results.json       # 5-fold MAPE mean±std
        sensitivity_results.json    # MAPE vs perturbation
        sensitivity_table.csv     # tabulated sensitivity
"""

import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# ── paths ──────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent.parent
DENSITY_JSON = HERE / "data" / "density_processed.json"
OUTPUT_DIR = HERE / "runs" / "depth_ratio_analysis"

# ── geometric model (must match calorie_estimator.py) ────────────────────────
GEOMETRY = {
    "apple": "ellipsoid", "banana": "ellipsoid", "bread": "box",
    "bun": "ellipsoid", "doughnut": "cylinder", "egg": "ellipsoid",
    "fired_dough_twist": "cylinder", "grape": "ellipsoid",
    "lemon": "ellipsoid", "litchi": "ellipsoid", "mango": "ellipsoid",
    "mooncake": "cylinder", "orange": "ellipsoid", "peach": "ellipsoid",
    "pear": "ellipsoid", "plum": "ellipsoid", "qiwi": "ellipsoid",
    "sachima": "box", "tomato": "ellipsoid",
}


def ellipsoid_volume(w, h, d):
    return (4 / 3) * math.pi * (w / 2) * (h / 2) * (d / 2)


def cylinder_volume(w, h, d):
    return math.pi * (w / 2) * (d / 2) * h


def box_volume(w, h, d):
    return w * h * d


def compute_depth_from_gt(gt_vol_cm3, w_mm, h_mm, geometry):
    """Invert volume formula to recover depth (mm) from ground-truth volume."""
    w, h = w_mm, h_mm
    if geometry == "ellipsoid":
        # V = (4/3)*pi*(w/2)*(h/2)*(d/2)  =>  d = 6*V / (pi*w*h)
        d_cm = 6 * gt_vol_cm3 / (math.pi * w * h)
    elif geometry == "cylinder":
        # V = pi*(w/2)*(d/2)*h  =>  d = 4*V / (pi*w*h)
        d_cm = 4 * gt_vol_cm3 / (math.pi * w * h)
    else:  # box
        # V = w*h*d  =>  d = V / (w*h)
        d_cm = gt_vol_cm3 / (w * h)
    return d_cm * 10  # cm -> mm


def derive_depth_ratios(per_image: dict, geometry: dict) -> dict[str, list[float]]:
    """Derive depth_ratio = D_actual / min(W, H) for every image in per_image."""
    ratios_by_class = defaultdict(list)
    for img_id, rec in per_image.items():
        cls = rec["class"]
        vol_cm3 = rec["volume_cm3"]
        geo = geometry.get(cls, "ellipsoid")

        # We need pixel dimensions of the bounding box.
        # density.xls doesn't store them, so we approximate: we use the mean
        # class bounding-box size from the YOLO labels (or a placeholder).
        # → NOTE: this derivation is only meaningful when we also have bbox sizes.
        # The per_image block in density_processed.json doesn't have bbox dims.
        #
        # Workaround: we skip actual derivation here and use the formula
        # directly from the known dataset: for each class, the GT volume was
        # measured from water displacement.  We CANNOT recover D without W,H.
        # So we MUST use bounding-box sizes from the YOLO labels.
        pass

    return dict(ratios_by_class)


def load_yolo_labels(label_dir: Path) -> dict[str, dict]:
    """Load all YOLO label files → {image_id: {class: {x_center, y_center, w, h}}}.
    Returns pixel dimensions (w_px, h_px) per (image_id, class).
    We use image dimensions from the original images if available, otherwise 640×480.
    """
    label_files = list(label_dir.rglob("*.txt"))
    results = {}

    # Try to infer image size from label dir structure
    # ECUSTFD labels are in datasets/ECUSTFD/labels/{train,val,test}/
    # Images are in datasets/ECUSTFD/images/{train,val,test}/
    # We'll use a default 640×480 (ECUSTFD images are typically 640×480)
    DEFAULT_W, DEFAULT_H = 640, 480

    for lbl_file in label_files:
        # image_id from filename: strip suffix
        img_id = lbl_file.stem
        # Remove S(1), T(2) suffix pattern used in ECUSTFD
        import re as _re
        clean = _re.sub(r"[ST]\(\d+\)$", "", img_id)
        # also strip trailing letter S/T
        clean = _re.sub(r"[ST]$", "", clean)

        entries = []
        with lbl_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:5])
                w_px = bw * DEFAULT_W
                h_px = bh * DEFAULT_H
                entries.append({"class_id": cls_id, "w_px": w_px, "h_px": h_px})
        results[clean] = entries
    return results


def load_predictions_csv(csv_path: Path) -> dict[str, dict]:
    """Load runs/calorie_eval/per_image_predictions.csv."""
    rows = {}
    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows[row["image"]] = row
    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ─────────────────────────────────────────────────────
    with DENSITY_JSON.open("r", encoding="utf-8") as fh:
        density = json.load(fh)
    per_image = density["per_image"]
    per_class = density["per_class"]

    # YOLO label dir: try train+val combined
    label_dirs = [
        HERE / "datasets" / "ECUSTFD" / "labels" / "test",
        HERE / "datasets" / "ECUSTFD" / "labels" / "train",
        HERE / "datasets" / "ECUSTFD" / "labels" / "val",
    ]
    all_labels = {}
    for d in label_dirs:
        if d.exists():
            all_labels.update(load_yolo_labels(d))

    print(f"[depth_ratio] Loaded {len(per_image)} per-image records")
    print(f"[depth_ratio] Loaded {len(all_labels)} label entries")

    # ── 2. Derive depth ratio per image ──────────────────────────────────
    # We need to map image_id in density.xls → label entries.
    # ECUSTFD naming: apple015, banana001, etc.
    # YOLO label names match: apple015, apple015S(1), etc.
    # Strategy: for each per_image entry, find matching label by prefix.
    import re as _re
    ratios_by_class = defaultdict(list)
    ratio_records = []  # {image_id, class, w_px, h_px, depth_ratio, gt_vol}

    for img_id, rec in per_image.items():
        cls = rec["class"]
        gt_vol = rec["volume_cm3"]  # cm³
        geo = GEOMETRY.get(cls, "ellipsoid")

        # Find matching label entry (use bare id without S/T suffix)
        bare_id = _re.sub(r"[ST]\(\d+\)$", "", img_id)
        bare_id = _re.sub(r"[ST]$", "", bare_id)

        label_entries = all_labels.get(bare_id, [])
        if not label_entries:
            # Try exact match
            label_entries = all_labels.get(img_id, [])

        # Find entry for this class
        cls_entry = None
        # Map class name → class_id (class 0-18, 19=coin)
        CLASS_NAMES = [
            "apple", "banana", "bread", "bun", "doughnut", "egg",
            "fired_dough_twist", "grape", "lemon", "litchi",
            "mango", "mooncake", "orange", "peach", "pear", "plum",
            "qiwi", "sachima", "tomato", "coin",
        ]
        cls_id = CLASS_NAMES.index(cls) if cls in CLASS_NAMES else -1

        for entry in label_entries:
            if entry["class_id"] == cls_id:
                cls_entry = entry
                break

        if cls_entry is None:
            print(f"  [WARN] No label for {img_id} ({cls}), skipping")
            continue

        w_px = cls_entry["w_px"]
        h_px = cls_entry["h_px"]
        w_mm = w_px * 0.264583  # Approximate: ECUSTFD ~96 DPI → 96/362.5 px/mm
        h_mm = h_px * 0.264583
        # Better: use coin diameter 25mm to calibrate
        # Find coin entry in same label
        coin_entry = None
        for entry in label_entries:
            if entry["class_id"] == 19:  # coin
                coin_entry = entry
                break
        if coin_entry:
            coin_w_px = coin_entry["w_px"]
            coin_h_px = coin_entry["h_px"]
            coin_avg_px = (coin_w_px + coin_h_px) / 2
            mm_per_px = 25.0 / coin_avg_px if coin_avg_px > 0 else 0.264583
            w_mm = w_px * mm_per_px
            h_mm = h_px * mm_per_px

        d_mm = compute_depth_from_gt(gt_vol, w_mm, h_mm, geo)
        ratio = d_mm / min(w_mm, h_mm) if min(w_mm, h_mm) > 0 else 0.0

        if 0 < ratio < 5:  # sanity filter
            ratios_by_class[cls].append(ratio)
            ratio_records.append({
                "image_id": img_id,
                "class": cls,
                "w_mm": w_mm,
                "h_mm": h_mm,
                "d_mm": d_mm,
                "depth_ratio": ratio,
                "gt_vol_cm3": gt_vol,
            })

    # ── 3. Per-class statistics ─────────────────────────────────────────
    stats_out = {}
    for cls, ratios in sorted(ratios_by_class.items()):
        arr = np.array(ratios)
        stats_out[cls] = {
            "n": int(len(ratios)),
            "median": float(np.median(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "q25": float(np.percentile(arr, 25)),
            "q75": float(np.percentile(arr, 75)),
        }

    # Save JSON
    stats_path = OUTPUT_DIR / "depth_ratio_stats.json"
    stats_path.write_text(json.dumps(stats_out, indent=2), encoding="utf-8")
    print(f"\n[depth_ratio] Saved {stats_path}")

    # ── 4. Compare with hard-coded Table 7 ───────────────────────────────
    HARDCODED = {
        "apple": 0.85, "banana": 0.55, "bread": 0.45, "bun": 0.45,
        "doughnut": 0.35, "egg": 0.75, "fired_dough_twist": 0.35,
        "grape": 0.90, "lemon": 0.85, "litchi": 0.90, "mango": 0.70,
        "mooncake": 0.35, "orange": 0.90, "peach": 0.85, "pear": 0.80,
        "plum": 0.85, "qiwi": 0.80, "sachima": 0.40, "tomato": 0.85,
    }

    comparison_rows = []
    for cls in sorted(HARDCODED.keys()):
        s = stats_out.get(cls, {})
        if not s:
            continue
        hard = HARDCODED[cls]
        learned = s["median"]
        diff = learned - hard
        diff_pct = (diff / hard * 100) if hard > 0 else 0
        comparison_rows.append({
            "class": cls,
            "hard_coded": hard,
            "learned_median": round(learned, 4),
            "learned_mean": round(s["mean"], 4),
            "learned_std": round(s["std"], 4),
            "difference": round(diff, 4),
            "diff_pct": round(diff_pct, 2),
            "n": s["n"],
        })

    comp_path = OUTPUT_DIR / "depth_ratio_comparison.csv"
    with comp_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=comparison_rows[0].keys())
        writer.writeheader()
        writer.writerows(comparison_rows)
    print(f"[depth_ratio] Saved {comp_path}")

    # ── 5. Boxplot ─────────────────────────────────────────────────────
    classes_ordered = sorted(ratios_by_class.keys())
    data = [ratios_by_class[c] for c in classes_ordered]
    # Sort classes by median ratio
    class_medians = [(c, np.median(ratios_by_class[c])) for c in classes_ordered]
    class_medians.sort(key=lambda x: x[1])
    sorted_classes = [c for c, _ in class_medians]
    sorted_data = [ratios_by_class[c] for c in sorted_classes]

    fig, ax = plt.subplots(figsize=(14, 6))
    bp = ax.boxplot(sorted_data, labels=sorted_classes, patch_artist=True,
                    medianprops={"color": "red", "linewidth": 2})
    colors = plt.cm.tab20(np.linspace(0, 1, len(sorted_classes)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Depth Ratio  D / min(W, H)")
    ax.set_xlabel("Food Class")
    ax.set_title("Depth Ratio Distribution by Class (Derived from GT Volume)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    # Add horizontal line at median of hard-coded values
    hard_vals = [HARDCODED.get(c, 0) for c in sorted_classes]
    ax.scatter(range(1, len(sorted_classes) + 1), hard_vals,
               color="blue", marker="D", s=30, label="Hard-coded ratio", zorder=5)
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "depth_ratio_boxplot.png", dpi=150)
    plt.close(fig)
    print(f"[depth_ratio] Saved depth_ratio_boxplot.png")

    # ── 6. Cross-validation (5-fold) ──────────────────────────────────
    print("\n[depth_ratio] Running 5-fold cross-validation...")
    import re as _re2
    # Rebuild ratio_records indexed by class
    by_class_records = defaultdict(list)
    for rec in ratio_records:
        by_class_records[rec["class"]].append(rec)

    cv_results = {}
    for cls in sorted(by_class_records.keys()):
        records = by_class_records[cls]
        if len(records) < 5:
            cv_results[cls] = {"note": f"Too few samples ({len(records)})"}
            continue
        ratios = [r["depth_ratio"] for r in records]
        n = len(ratios)
        k = 5
        fold_size = n // k
        fold_mapes = []
        geo = GEOMETRY.get(cls, "ellipsoid")
        density_cls = per_class.get(cls, {}).get("mean_density_g_cm3", 1.0)
        kcal_per_100g = 52  # placeholder — use FOOD_INFO

        # Use FOOD_INFO kcal
        kcal_map = {
            "apple": 52, "banana": 89, "bread": 265, "bun": 223,
            "doughnut": 452, "egg": 155, "fired_dough_twist": 450,
            "grape": 69, "lemon": 29, "litchi": 66, "mango": 60,
            "mooncake": 420, "orange": 47, "peach": 39, "pear": 57,
            "plum": 46, "qiwi": 61, "sachima": 450, "tomato": 18,
        }
        kcal_per_100g = kcal_map.get(cls, 50)

        for fold in range(k):
            start = fold * fold_size
            end = start + fold_size if fold < k - 1 else n
            train_ratios = ratios[:start] + ratios[end:]
            test_records = records[start:end]
            if not train_ratios:
                continue
            median_ratio = np.median(train_ratios)

            # Compute MAPE on test fold
            abs_errors = []
            for rec in test_records:
                w, h = rec["w_mm"], rec["h_mm"]
                d = min(w, h) * median_ratio
                if geo == "ellipsoid":
                    vol = ellipsoid_volume(w, h, d)
                elif geo == "cylinder":
                    vol = cylinder_volume(w, h, d)
                else:
                    vol = box_volume(w, h, d)
                mass = vol * density_cls / 10  # cm³ * g/cm³
                pred_kcal = mass * kcal_per_100g / 100
                gt_kcal = rec["gt_vol_cm3"] * density_cls * kcal_per_100g / 100
                abs_errors.append(abs(pred_kcal - gt_kcal) / max(gt_kcal, 1e-6))
            fold_mape = np.mean(abs_errors) if abs_errors else 0
            fold_mapes.append(fold_mape)

        if fold_mapes:
            cv_results[cls] = {
                "n": n,
                "fold_mapes": [round(x, 4) for x in fold_mapes],
                "mean_mape": round(float(np.mean(fold_mapes)), 4),
                "std_mape": round(float(np.std(fold_mapes)), 4),
                "derived_ratio": round(float(np.median(ratios)), 4),
            }

    cv_path = OUTPUT_DIR / "crossval_results.json"
    cv_path.write_text(json.dumps(cv_results, indent=2), encoding="utf-8")
    print(f"[depth_ratio] Saved {cv_path}")

    # ── 7. Sensitivity analysis ─────────────────────────────────────────
    print("\n[depth_ratio] Running sensitivity analysis...")

    # Load per-image predictions to get GT calories
    pred_csv = HERE / "runs" / "calorie_eval" / "per_image_predictions.csv"
    if pred_csv.exists():
        preds = load_predictions_csv(pred_csv)
    else:
        preds = {}

    sensitivity_results = {}
    for cls in sorted(by_class_records.keys()):
        records = by_class_records[cls]
        if len(records) < 3:
            continue
        base_ratio = stats_out[cls]["median"]
        geo = GEOMETRY.get(cls, "ellipsoid")
        density_cls = per_class.get(cls, {}).get("mean_density_g_cm3", 1.0)
        kcal_map = {
            "apple": 52, "banana": 89, "bread": 265, "bun": 223,
            "doughnut": 452, "egg": 155, "fired_dough_twist": 450,
            "grape": 69, "lemon": 29, "litchi": 66, "mango": 60,
            "mooncake": 420, "orange": 47, "peach": 39, "pear": 57,
            "plum": 46, "qiwi": 61, "sachima": 450, "tomato": 18,
        }
        kcal_per_100g = kcal_map.get(cls, 50)

        perturbations = [-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30]
        mape_by_pert = {}
        for pert in perturbations:
            ratio = base_ratio * (1 + pert)
            abs_errors = []
            for rec in records:
                w, h = rec["w_mm"], rec["h_mm"]
                d = min(w, h) * ratio
                if geo == "ellipsoid":
                    vol = ellipsoid_volume(w, h, d)
                elif geo == "cylinder":
                    vol = cylinder_volume(w, h, d)
                else:
                    vol = box_volume(w, h, d)
                mass = vol * density_cls / 10
                pred_kcal = mass * kcal_per_100g / 100
                gt_vol = rec["gt_vol_cm3"]
                gt_kcal = gt_vol * density_cls * kcal_per_100g / 100
                abs_errors.append(abs(pred_kcal - gt_kcal) / max(gt_kcal, 1e-6))
            mape_by_pert[f"{pert*100:+.0f}%"] = round(float(np.mean(abs_errors)), 4)

        sensitivity_results[cls] = {
            "base_ratio": round(base_ratio, 4),
            "mape_by_perturbation": mape_by_pert,
        }

    sens_path = OUTPUT_DIR / "sensitivity_results.json"
    sens_path.write_text(json.dumps(sensitivity_results, indent=2), encoding="utf-8")

    # CSV summary
    sens_rows = []
    for cls, res in sensitivity_results.items():
        row = {"class": cls, "base_ratio": res["base_ratio"]}
        row.update(res["mape_by_perturbation"])
        sens_rows.append(row)
    if sens_rows:
        sens_csv_path = OUTPUT_DIR / "sensitivity_table.csv"
        with sens_csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=sens_rows[0].keys())
            writer.writeheader()
            writer.writerows(sens_rows)
        print(f"[depth_ratio] Saved {sens_csv_path}")

    # ── 8. Print summary ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DEPTH RATIO ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"\n{'Class':<20} {'N':>5} {'Learned':>10} {'Hardcoded':>10} {'Diff':>10} {'Diff%':>8}")
    print("-" * 65)
    for row in comparison_rows:
        print(f"{row['class']:<20} {row['n']:>5} "
              f"{row['learned_median']:>10.4f} {row['hard_coded']:>10.4f} "
              f"{row['difference']:>+10.4f} {row['diff_pct']:>+7.2f}%")

    print("\n--- Cross-validation MAPE (mean ± std) ---")
    for cls, res in cv_results.items():
        if "mean_mape" in res:
            print(f"  {cls:<20} n={res['n']:>3}  "
                  f"MAPE={res['mean_mape']:.2%} ± {res['std_mape']:.2%}")

    print("\n--- Sensitivity (base → ±30%) ---")
    for cls, res in sensitivity_results.items():
        m0 = res["mape_by_perturbation"].get("+0%", "N/A")
        m30 = res["mape_by_perturbation"].get("+30%", "N/A")
        m_n30 = res["mape_by_perturbation"].get("-30%", "N/A")
        print(f"  {cls:<20} base={res['base_ratio']:.4f}  "
              f"MAPE(ratio±0%)={m0}  ±30%={m30}  -30%={m_n30}")

    print(f"\nOutput: {OUTPUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
