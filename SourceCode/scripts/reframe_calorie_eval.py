"""Reframe Calorie Results — Grouped MAPE + uncertainty quantification.

Re-organizes the calorie evaluation into three groups:
  - Geometric-friendly   (MAPE < 25%): litchi, egg, plum, pear, lemon
  - Geometric-moderate   (25% < MAPE < 50%): apple, peach, orange, ...
  - Geometric-hostile    (MAPE > 50%): bread, grape, banana, doughnut, qiwi

Also computes per-class bootstrap CI and proposes a routing/confidence system.

Usage
-----
    python scripts/reframe_calorie_eval.py

Output
------
    runs/reframe_calorie_eval/
        grouped_results.csv
        grouped_summary.json
        per_class_uncertainty.csv
"""

import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

# ── paths ──────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent.parent
PRED_CSV = HERE / "runs" / "calorie_eval" / "per_image_predictions.csv"
OUTPUT_DIR = HERE / "runs" / "reframe_calorie_eval"

# ── Groups ──────────────────────────────────────────────────────────────────
GEOMETRIC_FRIENDLY = {"litchi", "egg", "plum", "pear", "lemon"}
GEOMETRIC_HOSTILE = {"bread", "grape", "banana", "doughnut", "qiwi"}
# GEOMETRIC_MODERATE = all remaining classes
ROUTING_CLASSES = GEOMETRIC_HOSTILE  # classes that get "low confidence" flag


def classify_group(cls):
    if cls in GEOMETRIC_FRIENDLY:
        return "geometric-friendly"
    if cls in GEOMETRIC_HOSTILE:
        return "geometric-hostile"
    return "geometric-moderate"


def load_predictions():
    rows = []
    if not PRED_CSV.exists():
        print(f"[WARN] {PRED_CSV} not found"); return rows
    with PRED_CSV.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    return rows


def paired_abs_err(predictions):
    out = {}
    for row in predictions:
        gt = row.get("gt_kcal", "")
        pred = row.get("predicted_kcal", "")
        img = row.get("image", "")
        try:
            out[img] = abs(float(pred) - float(gt))
        except (TypeError, ValueError):
            pass
    return out


def bootstrap_ci(values, n_resamples=5000, ci=0.95, rng_seed=42):
    if not values:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(rng_seed)
    arr = np.asarray(values, dtype=float)
    boot = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, len(arr), size=len(arr))
        boot[i] = np.mean(arr[idx])
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot, [alpha, 1 - alpha])
    return float(np.mean(arr)), float(lo), float(hi)


def compute_mape_kcal(predictions):
    """Compute MAPE from predictions list of dicts."""
    pcts = []
    for row in predictions:
        try:
            gt = float(row["gt_kcal"])
            pred = float(row["predicted_kcal"])
            if gt > 0:
                pcts.append(abs(pred - gt) / gt)
        except (TypeError, ValueError):
            continue
    return pcts


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(42)
    np.random.seed(42)

    predictions = load_predictions()
    print(f"[reframe] Loaded {len(predictions)} predictions")

    # ── Per-image data ─────────────────────────────────────────────────
    all_rows = []
    for row in predictions:
        try:
            gt = float(row["gt_kcal"])
            pred = float(row["predicted_kcal"])
            cls = row.get("class", row.get("class_name", "unknown"))
            if gt <= 0:
                continue
            all_rows.append({
                "image": row["image"],
                "class": cls,
                "group": classify_group(cls),
                "gt_kcal": gt,
                "pred_kcal": pred,
                "abs_err_kcal": abs(pred - gt),
                "abs_pct_err": abs(pred - gt) / gt,
                "signed_err": pred - gt,
                "routing": "low_conf" if cls in ROUTING_CLASSES else "standard",
            })
        except (TypeError, ValueError):
            continue

    # ── Grouped results ──────────────────────────────────────────────────
    groups = defaultdict(list)
    for r in all_rows:
        groups[r["group"]].append(r)

    group_summary = {}
    for group, rows in groups.items():
        pcts = [r["abs_pct_err"] for r in rows]
        errs_kcal = [r["abs_err_kcal"] for r in rows]
        mae_kcal_pt, ci_lo, ci_hi = bootstrap_ci(errs_kcal)
        mape_pt, mape_lo, mape_hi = bootstrap_ci(pcts)
        acc20 = sum(1 for p in pcts if p <= 0.20) / len(pcts)

        group_summary[group] = {
            "n": len(rows),
            "mae_kcal": round(mae_kcal_pt, 2),
            "mae_ci": f"[{ci_lo:.2f}, {ci_hi:.2f}]",
            "mape": round(mape_pt, 4),
            "mape_ci": f"[{mape_lo:.2%}, {mape_hi:.2%}]",
            "acc_at_20pct": round(acc20, 4),
        }

    # ── Routing system: exclude hostile classes ────────────────────────────
    routed_rows = [r for r in all_rows if r["routing"] == "standard"]
    routed_pcts = [r["abs_pct_err"] for r in routed_rows]
    routed_errs = [r["abs_err_kcal"] for r in routed_rows]
    routed_mae_pt, r_ci_lo, r_ci_hi = bootstrap_ci(routed_errs)
    routed_mape_pt, rm_lo, rm_hi = bootstrap_ci(routed_pcts)
    routed_acc20 = sum(1 for p in routed_pcts if p <= 0.20) / len(routed_pcts)

    routed_summary = {
        "n": len(routed_rows),
        "mae_kcal": round(routed_mae_pt, 2),
        "mae_ci": f"[{r_ci_lo:.2f}, {r_ci_hi:.2f}]",
        "mape": round(routed_mape_pt, 4),
        "mape_ci": f"[{rm_lo:.2%}, {rm_hi:.2%}]",
        "acc_at_20pct": round(routed_acc20, 4),
        "note": "Routing excludes geometric-hostile classes (bread, grape, banana, doughnut, qiwi)",
    }

    # ── Per-class uncertainty ────────────────────────────────────────────────
    by_class = defaultdict(list)
    for r in all_rows:
        by_class[r["class"]].append(r)

    class_uncertainty = {}
    for cls, rows in sorted(by_class.items()):
        pcts = [r["abs_pct_err"] for r in rows]
        errs = [r["abs_err_kcal"] for r in rows]
        signed = [r["signed_err"] for r in rows]

        mae_pt, mae_lo, mae_hi = bootstrap_ci(errs)
        mape_pt, mape_lo, mape_hi = bootstrap_ci(pcts)

        class_uncertainty[cls] = {
            "n": len(rows),
            "group": rows[0]["group"] if rows else "unknown",
            "mae_kcal": round(mae_pt, 2),
            "mae_ci_95": f"[{mae_lo:.2f}, {mae_hi:.2f}]",
            "mae_pm": round((mae_hi - mae_lo) / 2, 2),
            "mape": round(mape_pt, 4),
            "mape_ci_95": f"[{mape_lo:.2%}, {mape_hi:.2%}]",
            "mape_pm": round((mape_hi - mape_lo) / 2, 2),
            "bias_kcal": round(float(np.mean(signed)), 2),
            "acc_at_20pct": round(sum(1 for p in pcts if p <= 0.20) / len(pcts), 4),
            "routing": rows[0]["routing"],
        }

    # ── Print summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("REFREAMED CALORIE RESULTS — GROUPED ANALYSIS")
    print("=" * 75)

    print("\n--- Grouped MAPE ---")
    print(f"{'Group':<25} {'n':>6} {'MAE (kcal)':>15} {'MAPE':>12} {'Acc@20%':>10}")
    print("-" * 72)
    for group in ["geometric-friendly", "geometric-moderate", "geometric-hostile"]:
        if group not in group_summary:
            continue
        g = group_summary[group]
        print(f"{group:<25} {g['n']:>6} "
              f"{g['mae_kcal']:>10.2f} {g['mae_ci']:>15} "
              f"{g['mape']:>12.2%} {g['acc_at_20pct']:>10.2%}")

    print("\n--- Routing System (exclude hostile classes) ---")
    print(f"  n = {routed_summary['n']}")
    print(f"  MAE = {routed_summary['mae_kcal']:.2f} kcal {routed_summary['mae_ci']}")
    print(f"  MAPE = {routed_summary['mape']:.2%} {routed_summary['mape_ci']}")
    print(f"  Acc@20% = {routed_summary['acc_at_20pct']:.2%}")

    print("\n--- Per-Class Uncertainty (sample) ---")
    print(f"{'Class':<20} {'Group':<22} {'n':>4} {'MAE':>8} {'MAE 95%CI':>18} {'MAPE':>8}")
    print("-" * 84)
    for cls, u in list(class_uncertainty.items())[:8]:
        print(f"{cls:<20} {u['group']:<22} {u['n']:>4} "
              f"{u['mae_kcal']:>8.1f} {u['mae_ci_95']:>18} {u['mape']:>8.2%}")

    # ── Save ────────────────────────────────────────────────────────────────
    # Grouped CSV
    grp_csv = OUTPUT_DIR / "grouped_results.csv"
    with grp_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Group", "n", "MAE (kcal)", "MAE 95% CI", "MAPE", "MAPE 95% CI", "Acc@20%"])
        for group in ["geometric-friendly", "geometric-moderate", "geometric-hostile"]:
            if group not in group_summary:
                continue
            g = group_summary[group]
            writer.writerow([
                group, g["n"], g["mae_kcal"], g["mae_ci"],
                f"{g['mape']:.2%}", g["mape_ci"], f"{g['acc_at_20pct']:.2%}",
            ])
        writer.writerow([])
        writer.writerow(["Routing (exclude hostile)", routed_summary["n"],
                        routed_summary["mae_kcal"], routed_summary["mae_ci"],
                        f"{routed_summary['mape']:.2%}", routed_summary["mape_ci"],
                        f"{routed_summary['acc_at_20pct']:.2%}"])
    print(f"\n[Saved] {grp_csv}")

    # Per-class CSV
    cls_csv = OUTPUT_DIR / "per_class_uncertainty.csv"
    fieldnames = ["class", "group", "n", "mae_kcal", "mae_ci_95", "mape",
                  "mape_ci_95", "bias_kcal", "acc_at_20pct", "routing"]
    with cls_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for cls, u in class_uncertainty.items():
            writer.writerow({
                "class": cls,
                "group": u["group"],
                "n": u["n"],
                "mae_kcal": u["mae_kcal"],
                "mae_ci_95": u["mae_ci_95"],
                "mape": f"{u['mape']:.2%}",
                "mape_ci_95": u["mape_ci_95"],
                "bias_kcal": u["bias_kcal"],
                "acc_at_20pct": f"{u['acc_at_20pct']:.2%}",
                "routing": u["routing"],
            })
    print(f"[Saved] {cls_csv}")

    # JSON summary
    out_json = OUTPUT_DIR / "grouped_summary.json"
    out_json.write_text(json.dumps({
        "groups": group_summary,
        "routing": routed_summary,
        "per_class": {k: {kk: vv for kk, vv in v.items() if kk != "group"}
                     for k, v in class_uncertainty.items()},
    }, indent=2), encoding="utf-8")
    print(f"[Saved] {out_json}")

    # Bar chart
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Grouped MAPE bar
    group_names = ["geometric-friendly", "geometric-moderate", "geometric-hostile"]
    group_mapes = [group_summary.get(g, {}).get("mape", 0) for g in group_names]
    group_n = [group_summary.get(g, {}).get("n", 0) for g in group_names]
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    bars = ax1.bar(group_names, [m * 100 for m in group_mapes], color=colors, alpha=0.8)
    ax1.set_ylabel("MAPE (%)")
    ax1.set_title("MAPE by Geometric Friendliness Group")
    ax1.axhline(y=20, color="gray", linestyle="--", alpha=0.5, label="20% threshold")
    ax1.legend()
    for bar, n in zip(bars, group_n):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"n={n}", ha="center", va="bottom", fontsize=9)

    # Per-class MAPE with CI
    sorted_cls = sorted(class_uncertainty.items(), key=lambda x: x[1]["mape"])
    cls_names = [c for c, _ in sorted_cls]
    cls_mapes = [u["mape"] * 100 for _, u in sorted_cls]
    cls_mape_pm = [u["mape_pm"] * 100 for _, u in sorted_cls]
    bar_colors = ["#2ecc71" if u["group"] == "geometric-friendly"
                 else "#f39c12" if u["group"] == "geometric-moderate"
                 else "#e74c3c"
                 for _, u in sorted_cls]
    ax2.barh(cls_names, cls_mapes, xerr=cls_mape_pm, color=bar_colors, alpha=0.8,
              capsize=3)
    ax2.axvline(x=20, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("MAPE (%)")
    ax2.set_title("Per-Class MAPE with 95% CI (colored by group)")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "grouped_mape_chart.png", dpi=150)
    plt.close(fig)
    print(f"[Saved] grouped_mape_chart.png")

    print(f"\nOutput: {OUTPUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
