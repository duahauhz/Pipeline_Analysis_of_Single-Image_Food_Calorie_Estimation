"""Statistical analysis for the ECUSTFD calorie evaluation.

Inputs are per-image prediction CSVs from
  - `runs/calorie_eval/per_image_predictions.csv`             (YOLOv13)
  - `runs/calorie_eval_yolov8n/per_image_predictions.csv`     (YOLOv8n, after training)
  - `runs/calorie_ablation_v13/predictions_{policy}.csv`     (ablation)

The script computes:
  1. Bootstrap 95% confidence interval for MAE, MAPE, RMSE, bias
     (5000 resamples, paired per image id).
  2. Paired Wilcoxon signed-rank test on the per-image absolute
     calorie error between two prediction sets (e.g. YOLOv13 vs
     YOLOv8n). Non-parametric: we don't trust normality with 1728
     images that include heavy-tailed outliers.
  3. Effect size (matched-pairs rank-biserial correlation) so we can
     report practical significance, not just p-values.
  4. A markdown report with all of the above plus per-class CIs.

Usage
-----
    python scripts/statistical_analysis.py \\
        --baseline runs/calorie_eval/per_image_predictions.csv \\
        --comparison runs/calorie_eval_yolov8n/per_image_predictions.csv \\
        --output runs/statistical_analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev

import numpy as np
from scipy import stats


def load_predictions(path: Path) -> dict[str, dict]:
    """Load a per-image CSV. We key by the full image path so that multiple
    photos of the same food id (e.g. apple015T(1), apple015S(1)) are kept
    as separate entries — this matters for the paired Wilcoxon test.
    """
    out: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            key = row.get("image")
            if not key:
                continue
            out[key] = row
    return out


def to_float(x) -> float | None:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def paired_abs_err(predictions: dict[str, dict]) -> dict[str, float]:
    """{image_id: |predicted_kcal - gt_kcal|} for images that have both."""
    out: dict[str, float] = {}
    for k, row in predictions.items():
        gt = to_float(row.get("gt_kcal"))
        pred = to_float(row.get("predicted_kcal"))
        if gt is not None and pred is not None and gt > 0:
            out[k] = abs(pred - gt)
    return out


def bootstrap_ci(
    values: list[float],
    n_resamples: int = 5000,
    ci: float = 0.95,
    statistic=np.mean,
    rng_seed: int = 42,
) -> tuple[float, float, float]:
    """Return (point_estimate, ci_low, ci_high) using percentile bootstrap."""
    if not values:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(rng_seed)
    arr = np.asarray(values)
    point = statistic(arr)
    boot = np.empty(n_resamples, dtype=float)
    n = len(arr)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boot[i] = statistic(arr[idx])
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot, [alpha, 1 - alpha])
    return float(point), float(lo), float(hi)


def paired_wilcoxon(
    baseline: dict[str, float], comparison: dict[str, float]
) -> dict[str, float]:
    """Run Wilcoxon signed-rank on the intersection of two error dicts.

    `baseline` and `comparison` are keyed by image path, not image id.
    """
    common = sorted(set(baseline) & set(comparison))
    if not common:
        return {"n": 0}
    a = np.array([baseline[k] for k in common], dtype=float)
    b = np.array([comparison[k] for k in common], dtype=float)
    diffs = a - b
    # Drop zero diffs (required by scipy).
    nonzero = diffs[diffs != 0]
    if len(nonzero) < 10:
        return {
            "n": int(len(common)),
            "n_nonzero_diffs": int(len(nonzero)),
            "note": "Too few non-zero diffs for Wilcoxon (likely identical inputs).",
            "mean_diff": float(np.mean(diffs)),
            "median_diff": float(np.median(diffs)),
            "mean_baseline": float(np.mean(a)),
            "mean_comparison": float(np.mean(b)),
        }
    try:
        stat, p = stats.wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided")
    except ValueError as exc:
        return {"n": int(len(common)), "note": f"Wilcoxon failed: {exc}"}
    # Effect size: matched-pairs rank-biserial correlation.
    n_pos = int(np.sum(nonzero > 0))
    n_neg = int(np.sum(nonzero < 0))
    r_rb = (n_pos - n_neg) / (n_pos + n_neg) if (n_pos + n_neg) > 0 else 0.0
    return {
        "n": int(len(common)),
        "n_nonzero_diffs": int(len(nonzero)),
        "mean_diff": float(np.mean(diffs)),
        "median_diff": float(np.median(diffs)),
        "wilcoxon_stat": float(stat),
        "p_value": float(p),
        "rank_biserial": float(r_rb),
        "mean_baseline": float(np.mean(a)),
        "mean_comparison": float(np.mean(b)),
    }


def per_class_metrics(
    predictions: dict[str, dict],
) -> dict[str, dict]:
    """MAE/RMSE/MAPE per class over images that have GT."""
    by_class: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in predictions.values():
        gt = to_float(row.get("gt_kcal"))
        pred = to_float(row.get("predicted_kcal"))
        gt_mass = to_float(row.get("gt_mass_g"))
        pred_mass = to_float(row.get("predicted_mass_g"))
        cls = row.get("class")
        if cls is None or cls == "" or cls == "None":
            continue
        if gt is None or pred is None or gt_mass is None or pred_mass is None:
            continue
        by_class[cls].append(
            {
                "abs_err_g": abs(pred_mass - gt_mass),
                "abs_err_kcal": abs(pred - gt),
                "abs_pct_err_kcal": abs(pred - gt) / gt if gt > 0 else 0.0,
                "signed_err_g": pred_mass - gt_mass,
            }
        )
    out: dict[str, dict] = {}
    for cls, records in by_class.items():
        errs = [r["abs_err_g"] for r in records]
        errs_k = [r["abs_err_kcal"] for r in records]
        pcts = [r["abs_pct_err_kcal"] for r in records]
        bias = [r["signed_err_g"] for r in records]
        out[cls] = {
            "n": len(records),
            "mae_g": mean(errs),
            "rmse_g": math.sqrt(sum(e * e for e in errs) / len(errs)),
            "bias_g": mean(bias),
            "mae_kcal": mean(errs_k),
            "mape_kcal": mean(pcts),
        }
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("runs/calorie_eval/per_image_predictions.csv"),
        help="Per-image CSV for the baseline (YOLOv13).",
    )
    parser.add_argument(
        "--comparison",
        type=Path,
        default=Path("runs/calorie_eval_yolov8n/per_image_predictions.csv"),
        help="Per-image CSV for the comparison model (YOLOv8n).",
    )
    parser.add_argument(
        "--ablation-dir",
        type=Path,
        default=Path("runs/calorie_ablation_v13"),
        help="Directory with predictions_{policy}.csv files (optional).",
    )
    parser.add_argument("--output", type=Path, default=Path("runs/statistical_analysis"))
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    args.output.mkdir(parents=True, exist_ok=True)

    if not args.baseline.exists():
        print(f"[stat] Baseline not found: {args.baseline}")
        baseline = {}
    else:
        baseline = load_predictions(args.baseline)
        print(f"[stat] Baseline: {len(baseline)} rows from {args.baseline}")

    if not args.comparison.exists():
        print(f"[stat] Comparison not found: {args.comparison}")
        comparison = {}
    else:
        comparison = load_predictions(args.comparison)
        print(f"[stat] Comparison: {len(comparison)} rows from {args.comparison}")

    if not baseline and not comparison:
        raise SystemExit("No prediction files found. Run eval_calorie first.")

    # 1. Bootstrap CI for baseline.
    base_errs = paired_abs_err(baseline)
    base_errs_g = [
        to_float(row.get("abs_err_g"))
        for row in baseline.values()
        if to_float(row.get("abs_err_g")) is not None
    ]
    print(f"[stat] Baseline paired abs_err (kcal): n={len(base_errs)}")

    base_ci = {
        "mae_kcal": bootstrap_ci(list(base_errs.values()), n_resamples=args.n_boot),
        "mae_g": bootstrap_ci(base_errs_g, n_resamples=args.n_boot),
    }
    pcts_base = [
        v / max(1e-6, to_float(baseline[k].get("gt_kcal")) or 0)
        for k, v in base_errs.items()
    ]
    base_ci["mape_kcal"] = bootstrap_ci(pcts_base, n_resamples=args.n_boot)

    summary: dict = {
        "baseline_file": str(args.baseline),
        "comparison_file": str(args.comparison),
        "n_baseline": len(baseline),
        "n_comparison": len(comparison),
        "baseline_bootstrap_ci": {
            k: {"point": v[0], "ci_low": v[1], "ci_high": v[2]} for k, v in base_ci.items()
        },
    }

    # 2. Per-class CIs for baseline.
    pc = per_class_metrics(baseline)
    pc_with_ci = {}
    for cls, m in pc.items():
        cls_rows = [
            row for row in baseline.values()
            if row.get("class") == cls and to_float(row.get("abs_err_g")) is not None
        ]
        cls_errs = [to_float(r["abs_err_g"]) for r in cls_rows]
        cls_pcts = [
            (to_float(r["abs_err_kcal"])) / max(1e-6, to_float(r["gt_kcal"]) or 0)
            for r in cls_rows
            if to_float(r.get("abs_err_kcal")) is not None and to_float(r.get("gt_kcal")) is not None
        ]
        cls_errs = [e for e in cls_errs if e is not None]
        ci_mae = bootstrap_ci(cls_errs, n_resamples=args.n_boot)
        ci_mape = bootstrap_ci(cls_pcts, n_resamples=args.n_boot)
        m["mae_g_ci"] = {"point": ci_mae[0], "ci_low": ci_mae[1], "ci_high": ci_mae[2]}
        m["mape_kcal_ci"] = {"point": ci_mape[0], "ci_low": ci_mape[1], "ci_high": ci_mape[2]}
        pc_with_ci[cls] = m
    summary["per_class"] = pc_with_ci

    # 3. Paired test baseline vs comparison.
    if comparison:
        comp_errs = paired_abs_err(comparison)
        wilcoxon = paired_wilcoxon(base_errs, comp_errs)
        summary["paired_test_baseline_vs_comparison"] = wilcoxon

    # 4. Ablation Wilcoxon (if dir exists).
    if args.ablation_dir.exists():
        policies = ["per_image", "per_class", "geometry_fallback"]
        ablation_results: dict[str, dict] = {}
        for p in policies:
            p_path = args.ablation_dir / f"predictions_{p}.csv"
            if not p_path.exists():
                continue
            p_preds = load_predictions(p_path)
            p_errs = paired_abs_err(p_preds)
            ablation_results[p] = {"n": len(p_errs)}
        # Compare per_class vs geometry_fallback (the meaningful pair).
        if "per_class" in ablation_results and "geometry_fallback" in ablation_results:
            pc_preds = load_predictions(args.ablation_dir / "predictions_per_class.csv")
            gf_preds = load_predictions(args.ablation_dir / "predictions_geometry_fallback.csv")
            pc_errs = paired_abs_err(pc_preds)
            gf_errs = paired_abs_err(gf_preds)
            summary["paired_test_per_class_vs_geometry_fallback"] = paired_wilcoxon(
                pc_errs, gf_errs
            )
        # And per_class vs per_image.
        if "per_image" in ablation_results and "per_class" in ablation_results:
            pi_preds = load_predictions(args.ablation_dir / "predictions_per_image.csv")
            pc_preds = load_predictions(args.ablation_dir / "predictions_per_class.csv")
            pi_errs = paired_abs_err(pi_preds)
            pc_errs = paired_abs_err(pc_preds)
            summary["paired_test_per_image_vs_per_class"] = paired_wilcoxon(
                pi_errs, pc_errs
            )
        summary["ablation_files"] = {
            p: str(args.ablation_dir / f"predictions_{p}.csv") for p in policies
        }

    out_json = args.output / "statistical_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[stat] Wrote {out_json}")

    # 5. Markdown report.
    md = args.output / "statistical_report.md"
    with md.open("w", encoding="utf-8") as file:
        file.write("# Statistical analysis of the calorie pipeline\n\n")
        file.write(f"- Baseline file: `{args.baseline}`\n")
        file.write(f"- Comparison file: `{args.comparison}`\n")
        file.write(f"- Bootstrap resamples: {args.n_boot}\n\n")

        file.write("## 1. Baseline bootstrap 95% CIs\n\n")
        file.write("| metric | point | CI low | CI high |\n|---|---|---|---|\n")
        for metric, ci in summary["baseline_bootstrap_ci"].items():
            file.write(
                f"| {metric} | {ci['point']:.4f} | {ci['ci_low']:.4f} | {ci['ci_high']:.4f} |\n"
            )

        file.write("\n## 2. Per-class bootstrap 95% CIs (baseline)\n\n")
        file.write("| class | n | MAE (g) | MAE CI | MAPE | MAPE CI |\n|---|---|---|---|---|---|\n")
        for cls, m in summary["per_class"].items():
            file.write(
                f"| {cls} | {m['n']} | {m['mae_g']:.2f} | "
                f"[{m['mae_g_ci']['ci_low']:.2f}, {m['mae_g_ci']['ci_high']:.2f}] | "
                f"{m['mape_kcal']:.2%} | "
                f"[{m['mape_kcal_ci']['ci_low']:.2%}, {m['mape_kcal_ci']['ci_high']:.2%}] |\n"
            )

        if "paired_test_baseline_vs_comparison" in summary:
            file.write("\n## 3. Paired Wilcoxon test: baseline vs comparison\n\n")
            w = summary["paired_test_baseline_vs_comparison"]
            file.write(f"- n common: {w.get('n', '?')}\n")
            file.write(f"- mean abs err (baseline): {w.get('mean_baseline', float('nan')):.4f}\n")
            file.write(
                f"- mean abs err (comparison): {w.get('mean_comparison', float('nan')):.4f}\n"
            )
            file.write(f"- mean paired diff (baseline - comparison): {w.get('mean_diff', float('nan')):.4f}\n")
            file.write(f"- Wilcoxon statistic: {w.get('wilcoxon_stat', float('nan')):.4f}\n")
            file.write(f"- p-value: {w.get('p_value', float('nan')):.4e}\n")
            file.write(f"- rank-biserial effect size: {w.get('rank_biserial', float('nan')):.4f}\n")
            file.write(
                "\nA positive `mean_diff` means baseline error > comparison error "
                "(i.e. baseline is worse on calorie MAE). A positive `rank_biserial` "
                "indicates the comparison has more pairs with smaller error.\n"
            )

        if "paired_test_per_class_vs_geometry_fallback" in summary:
            file.write("\n## 4. Ablation paired Wilcoxon: per_class vs geometry_fallback\n\n")
            w = summary["paired_test_per_class_vs_geometry_fallback"]
            file.write(f"- n common: {w.get('n', '?')}\n")
            file.write(f"- mean abs err (per_class): {w.get('mean_baseline', float('nan')):.4f}\n")
            file.write(
                f"- mean abs err (geometry_fallback): {w.get('mean_comparison', float('nan')):.4f}\n"
            )
            file.write(f"- p-value: {w.get('p_value', float('nan')):.4e}\n")
            file.write(f"- rank-biserial effect size: {w.get('rank_biserial', float('nan')):.4f}\n")

        if "paired_test_per_image_vs_per_class" in summary:
            file.write("\n## 5. Ablation paired Wilcoxon: per_image vs per_class\n\n")
            w = summary["paired_test_per_image_vs_per_class"]
            file.write(f"- n common: {w.get('n', '?')}\n")
            file.write(f"- mean abs err (per_image): {w.get('mean_baseline', float('nan')):.4f}\n")
            file.write(f"- mean abs err (per_class): {w.get('mean_comparison', float('nan')):.4f}\n")
            file.write(f"- p-value: {w.get('p_value', float('nan')):.4e}\n")
            file.write(f"- rank-biserial effect size: {w.get('rank_biserial', float('nan')):.4f}\n")

    print(f"[stat] Wrote {md}")
    print()
    print("=" * 60)
    print("STATISTICAL SUMMARY")
    print("=" * 60)
    for metric, ci in summary["baseline_bootstrap_ci"].items():
        print(
            f"  {metric:20s} : {ci['point']:.4f}  CI=[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]"
        )
    if "paired_test_baseline_vs_comparison" in summary:
        w = summary["paired_test_baseline_vs_comparison"]
        print()
        print(f"  Paired test (baseline vs comparison):")
        print(f"    n = {w.get('n', '?')}")
        print(f"    p = {w.get('p_value', float('nan')):.4e}")
        print(f"    rank-biserial = {w.get('rank_biserial', float('nan')):.4f}")


if __name__ == "__main__":
    main()
