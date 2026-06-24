"""Calorie MAPE using *ground-truth* bboxes (no detector involved).

This script is a controlled ablation: it replaces the YOLOv13n detection
with the actual YOLO-format labels from
``datasets/ECUSTFD/labels/{val,test}/`` and re-uses the same dual-view
geometric pipeline as ``eval_calorie_dual_view.py``.

Why this script exists
----------------------
* The local environment blocks the venv Python via Device Guard, so the
  full detector-based pipeline cannot run end-to-end here.
* But the labels are the *perfect detector* — they tell us where the food
  and the coin actually are, so the only thing left to measure is the
  geometric / volume / calorie model itself.
* It also answers a sharper research question than the YOLO-based one:
  **"if the detector were perfect, how much would adding a side view
  improve MAPE calorie?"** This isolates the contribution of the geometry
  from the contribution of the detector.
* When the detector run becomes possible, ``eval_calorie_dual_view.py``
  will repeat the same numbers with real YOLO output, and the delta
  between the two scripts is exactly the detector's contribution.

Pipeline (identical to the detector version, just with GT bboxes)
-----------------------------------------------------------------
For every pair id that has both ``<id>T(N).JPG`` and ``<id>S(N).JPG``
labels in the chosen split:

1. Read GT bboxes from both labels files.
2. Compute mm/px from the coin bbox (25 mm diameter).
3. 1-view (top only): volume from top bbox with depth =
   ``min(W,H) * depth_ratio`` (per-class ratio from FOOD_INFO).
4. 2-view: real L×W×D from top + side silhouettes.
5. Mass = volume × density (per-class from density_processed.json).
6. Calorie = mass × kcal/100g. Compare against the GT record in
   ``density_processed.json``.

Output (same layout as the detector version)
-------------------------------------------
runs/dual_view_eval_gt/
    per_image_predictions.csv
    per_class_metrics_v1.csv / per_class_metrics_v2.csv
    comparison_1v_vs_2v.csv
    comparison_1v_vs_2v.md
    error_analysis.md
    summary.json
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


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from calorie_estimator import (  # noqa: E402
    FOOD_INFO,
    image_id_from_stem,
)


# ECUSTFD images are all 3000×4000. Hard-coding is safe because every
# test/val sample in the repo was shot with the same fixed camera.
IMG_W = 3000
IMG_H = 4000
COIN_DIAMETER_MM = 25.0
FOOD_CLASS_NAMES = {i: name for i, name in enumerate(
    ["apple","banana","bread","bun","doughnut","egg","fired_dough_twist",
     "grape","lemon","litchi","mango","mooncake","orange","peach","pear",
     "plum","qiwi","sachima","tomato"]
)}
COIN_CLASS_ID = 19


# --------------------------------------------------------------------------- #
# YOLO-label parsing
# --------------------------------------------------------------------------- #


VIEW_TRAIL_RE = re.compile(r"([TS])\(\d+\)$", re.IGNORECASE)


def view_of_stem(stem: str) -> str | None:
    m = VIEW_TRAIL_RE.search(stem)
    return m.group(1).upper() if m else None


def pair_id_of_stem(stem: str) -> str:
    return image_id_from_stem(stem)


def parse_yolo_label(path: Path) -> list[dict]:
    """YOLO label format: ``cls cx cy w h`` (normalized).

    Returns one dict per line with ``cls``, ``cx``, ``cy``, ``w``, ``h``
    in normalized 0..1.
    """
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            rows.append({
                "cls": int(float(parts[0])),
                "cx": float(parts[1]),
                "cy": float(parts[2]),
                "w": float(parts[3]),
                "h": float(parts[4]),
            })
        except ValueError:
            continue
    return rows


def split_food_coin(records: list[dict]) -> tuple[dict | None, dict | None]:
    food = None
    coin = None
    for r in records:
        if r["cls"] == COIN_CLASS_ID:
            coin = r
        elif r["cls"] in FOOD_CLASS_NAMES:
            # prefer the largest non-coin bbox
            if food is None or r["w"] * r["h"] > food["w"] * food["h"]:
                food = r
    return food, coin


def bbox_px(rec: dict) -> tuple[float, float, float, float]:
    """Convert normalized YOLO bbox to absolute pixels."""
    w_px = rec["w"] * IMG_W
    h_px = rec["h"] * IMG_H
    cx_px = rec["cx"] * IMG_W
    cy_px = rec["cy"] * IMG_H
    return cx_px - w_px / 2.0, cy_px - h_px / 2.0, cx_px + w_px / 2.0, cy_px + h_px / 2.0


# --------------------------------------------------------------------------- #
# Geometry (mirrors the detector version)
# --------------------------------------------------------------------------- #


def volume_cm3(length_mm: float, width_mm: float, depth_mm: float, geometry: str) -> float:
    L = length_mm / 10.0
    W = width_mm / 10.0
    D = depth_mm / 10.0
    if geometry == "box":
        return L * W * D
    if geometry == "cylinder":
        return math.pi * (W / 2.0) * (D / 2.0) * L
    return (4.0 / 3.0) * math.pi * (L / 2.0) * (W / 2.0) * (D / 2.0)


def one_view_dims(food_px_w: float, food_px_h: float, mm_per_px: float,
                  depth_ratio: float) -> dict:
    L_mm = max(food_px_w, food_px_h) * mm_per_px
    W_mm = min(food_px_w, food_px_h) * mm_per_px
    D_mm = min(L_mm, W_mm) * depth_ratio
    return {"length_mm": L_mm, "width_mm": W_mm, "depth_mm": D_mm}


def two_view_dims(top_food_px: tuple[float, float],
                  side_food_px: tuple[float, float],
                  top_mm_per_px: float, side_mm_per_px: float,
                  depth_ratio: float) -> dict:
    """Estimate 3D dimensions from top + side silhouettes.

    The smaller of the two side-view axes is the real depth of the food;
    we use it directly (with a sanity floor) instead of the per-class
    depth-ratio guess used in the 1-view pipeline.
    """
    top_x = top_food_px[0] * top_mm_per_px
    top_y = top_food_px[1] * top_mm_per_px
    side_x = side_food_px[0] * side_mm_per_px
    side_y = side_food_px[1] * side_mm_per_px

    top_axes = sorted([top_x, top_y], reverse=True)
    side_axes = sorted([side_x, side_y], reverse=True)
    L = top_axes[0]
    W = top_axes[1]
    H = side_axes[0]
    D = side_axes[1]

    # Sanity-floor: side view may be end-on (a banana seen from the tip
    # would give a very small diameter). In that case trust the geometric
    # prior rather than a noisy measurement.
    if D > W * 1.05:
        D = W
    if D < 0.25 * W:
        D = min(L, W) * depth_ratio
    return {
        "length_mm": L,
        "width_mm": W,
        "height_mm": H,
        "depth_mm": D,
    }
    top_x = top_food_px[0] * top_mm_per_px
    top_y = top_food_px[1] * top_mm_per_px
    side_x = side_food_px[0] * side_mm_per_px
    side_y = side_food_px[1] * side_mm_per_px

    top_axes = sorted([top_x, top_y], reverse=True)
    side_axes = sorted([side_x, side_y], reverse=True)
    L = top_axes[0]
    W = top_axes[1]
    H = side_axes[0]
    D = side_axes[1]

    # Heuristic: the depth should never be drastically larger than the width
    # (it would mean the side view caught the food end-on and the silhouette
    # is a thin slice). Cap D at W and floor it at 0.25 * W to stay inside a
    # realistic range for food geometry.
    if D > W:
        D = min(D, W * 1.05)  # tiny tolerance
    if D < 0.25 * W:
        # Side view too thin to be the depth — fall back to the geometric prior
        D = min(L, W) * depth_ratio
    return {
        "length_mm": L,
        "width_mm": W,
        "height_mm": H,
        "depth_mm": D,
    }


# --------------------------------------------------------------------------- #
# Main evaluation
# --------------------------------------------------------------------------- #


def per_class_density(class_name: str, density_data: dict) -> float | None:
    pc = density_data.get("per_class", {}).get(class_name, {})
    if pc.get("mean_density_g_cm3", 0) > 0:
        return pc["mean_density_g_cm3"]
    info = FOOD_INFO.get(class_name)
    return info["density"] if info else None


def discover_pairs(labels_root: Path) -> dict[str, dict[str, Path]]:
    """Find all pair ids that have both T and S label files in any subdir.

    Walks ``labels_root`` recursively (so it can cover ``labels/{train,val,test}``
    in one go) and returns ``{pair_id: {'T': path, 'S': path}}``.
    """
    pairs: dict[str, dict[str, Path]] = {}
    for path in sorted(labels_root.rglob("*.txt")):
        view = view_of_stem(path.stem)
        if view is None:
            continue
        pair_id = pair_id_of_stem(path.stem)
        pairs.setdefault(pair_id, {})[view] = path
    return {pid: v for pid, v in pairs.items() if "T" in v and "S" in v}


def evaluate_pair(pair_id: str, paths: dict[str, Path],
                  density_data: dict) -> dict:
    top_records = parse_yolo_label(paths["T"])
    side_records = parse_yolo_label(paths["S"])
    top_food, top_coin = split_food_coin(top_records)
    side_food, side_coin = split_food_coin(side_records)

    gt = density_data.get("per_image", {}).get(pair_id)
    gt_class = gt["class"] if gt else None
    gt_kcal = None
    if gt:
        info = FOOD_INFO.get(gt_class)
        if info:
            gt_kcal = gt["weight_g"] * info["kcal_per_100g"] / 100.0

    row = {
        "pair_id": pair_id,
        "class": gt_class,
        "top_label": str(paths["T"]),
        "side_label": str(paths["S"]),
        "has_gt": gt is not None,
        "gt_mass_g": gt["weight_g"] if gt else None,
        "gt_volume_cm3": gt["volume_cm3"] if gt else None,
        "gt_kcal": gt_kcal,
        # 1-view
        "v1_predicted_mass_g": None,
        "v1_predicted_kcal": None,
        "v1_predicted_volume_cm3": None,
        "v1_abs_err_kcal": None,
        "v1_abs_pct_err_kcal": None,
        "v1_status": "ok",
        # 2-view
        "v2_predicted_mass_g": None,
        "v2_predicted_kcal": None,
        "v2_predicted_volume_cm3": None,
        "v2_abs_err_kcal": None,
        "v2_abs_pct_err_kcal": None,
        "v2_status": "ok",
        # diagnostics
        "top_food_class": (FOOD_CLASS_NAMES.get(top_food["cls"]) if top_food else None),
        "side_food_class": (FOOD_CLASS_NAMES.get(side_food["cls"]) if side_food else None),
        "top_coin_px": None,
        "side_coin_px": None,
        "length_mm_v2": None,
        "width_mm_v2": None,
        "depth_mm_v2": None,
        "height_mm_v2": None,
    }

    if top_food is None or top_coin is None:
        row["v1_status"] = "missing_top_food_or_coin"
    if top_food is None or top_coin is None or side_food is None or side_coin is None:
        row["v2_status"] = "missing_coin_or_food"

    if top_coin is not None:
        cw = top_coin["w"] * IMG_W
        ch = top_coin["h"] * IMG_H
        row["top_coin_px"] = (cw + ch) / 2.0
    if side_coin is not None:
        cw = side_coin["w"] * IMG_W
        ch = side_coin["h"] * IMG_H
        row["side_coin_px"] = (cw + ch) / 2.0

    # ---- 1-view (top only) ----
    if top_food is not None and top_coin is not None and gt_class:
        info = FOOD_INFO.get(gt_class)
        if info is not None and row["top_coin_px"] > 0:
            mm_per_px = COIN_DIAMETER_MM / row["top_coin_px"]
            food_w_px = top_food["w"] * IMG_W
            food_h_px = top_food["h"] * IMG_H
            dims = one_view_dims(food_w_px, food_h_px, mm_per_px, info["depth_ratio"])
            volume = volume_cm3(dims["length_mm"], dims["width_mm"], dims["depth_mm"], info["geometry"])
            density = per_class_density(gt_class, density_data) or info["density"]
            mass = volume * density
            kcal = mass * info["kcal_per_100g"] / 100.0
            row["v1_predicted_volume_cm3"] = volume
            row["v1_predicted_mass_g"] = mass
            row["v1_predicted_kcal"] = kcal

    # ---- 2-view (top + side) ----
    if (top_food is not None and side_food is not None
            and top_coin is not None and side_coin is not None and gt_class):
        info = FOOD_INFO.get(gt_class)
        if info is not None and row["top_coin_px"] > 0 and row["side_coin_px"] > 0:
            top_mm_per_px = COIN_DIAMETER_MM / row["top_coin_px"]
            side_mm_per_px = COIN_DIAMETER_MM / row["side_coin_px"]
            top_food_px = (top_food["w"] * IMG_W, top_food["h"] * IMG_H)
            side_food_px = (side_food["w"] * IMG_W, side_food["h"] * IMG_H)
            dims = two_view_dims(top_food_px, side_food_px, top_mm_per_px, side_mm_per_px,
                                 info["depth_ratio"])
            volume = volume_cm3(dims["length_mm"], dims["width_mm"], dims["depth_mm"], info["geometry"])
            density = per_class_density(gt_class, density_data) or info["density"]
            mass = volume * density
            kcal = mass * info["kcal_per_100g"] / 100.0
            row["v2_predicted_volume_cm3"] = volume
            row["v2_predicted_mass_g"] = mass
            row["v2_predicted_kcal"] = kcal
            row["length_mm_v2"] = dims["length_mm"]
            row["width_mm_v2"] = dims["width_mm"]
            row["depth_mm_v2"] = dims["depth_mm"]
            row["height_mm_v2"] = dims["height_mm"]

    # ---- Errors vs GT ----
    if row["has_gt"] and row["gt_kcal"] and row["gt_kcal"] > 0:
        if row["v1_predicted_kcal"] is not None:
            row["v1_abs_err_kcal"] = abs(row["v1_predicted_kcal"] - row["gt_kcal"])
            row["v1_abs_pct_err_kcal"] = row["v1_abs_err_kcal"] / row["gt_kcal"]
        if row["v2_predicted_kcal"] is not None:
            row["v2_abs_err_kcal"] = abs(row["v2_predicted_kcal"] - row["gt_kcal"])
            row["v2_abs_pct_err_kcal"] = row["v2_abs_err_kcal"] / row["gt_kcal"]
    return row


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def aggregate(rows: list[dict], key: str) -> dict:
    pcts = [r[f"{key}_abs_pct_err_kcal"] for r in rows
            if r[f"{key}_abs_pct_err_kcal"] is not None]
    mass_errs = []
    for r in rows:
        if r["has_gt"] and r["gt_mass_g"] and r[f"{key}_predicted_mass_g"] is not None:
            mass_errs.append(abs(r[f"{key}_predicted_mass_g"] - r["gt_mass_g"]))
    n = len(pcts)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "mape_kcal": round(sum(pcts) / n, 4),
        "mae_g": round(sum(mass_errs) / len(mass_errs), 2) if mass_errs else None,
        "accuracy_at_20pct": round(sum(1 for p in pcts if p <= 0.20) / n, 4),
    }


def aggregate_per_class(rows: list[dict], key: str) -> dict:
    out: dict[str, dict] = {}
    by_cls: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r["class"] is None:
            continue
        pct = r[f"{key}_abs_pct_err_kcal"]
        if pct is None:
            continue
        by_cls[r["class"]].append(pct)
    for cls, pcts in sorted(by_cls.items()):
        out[cls] = {
            "n": len(pcts),
            "mape_kcal": round(sum(pcts) / len(pcts), 4),
            "accuracy_at_20pct": round(sum(1 for p in pcts if p <= 0.20) / len(pcts), 4),
        }
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels-root", type=Path,
                   default=Path("datasets/ECUSTFD/labels"),
                   help="Root containing {train,val,test} subdirs. All subdirs are walked.")
    p.add_argument("--labels-dir", type=Path, default=None,
                   help="(Deprecated) single labels dir. Use --labels-root instead.")
    p.add_argument("--density-json", type=Path, default=Path("data/density_processed.json"))
    p.add_argument("--output", type=Path, default=Path("runs/dual_view_eval_gt"))
    p.add_argument("--max-pairs", type=int, default=0)
    return p.parse_args()


def _fmt_pct(v) -> str:
    return "—" if v is None else f"{v * 100:.2f}%"


def main() -> None:
    args = parse_args()
    labels_root = args.labels_dir or args.labels_root
    if not labels_root.exists():
        raise FileNotFoundError(f"Missing labels root: {labels_root}")
    density_data = json.loads(args.density_json.read_text(encoding="utf-8"))
    print(f"[dual_view_gt] density per_image={len(density_data['per_image'])} "
          f"per_class={len(density_data['per_class'])}")

    pairs = discover_pairs(labels_root)
    # Keep only pairs that have a density GT record
    pairs = {pid: v for pid, v in pairs.items() if pid in density_data["per_image"]}
    if args.max_pairs:
        pairs = dict(list(pairs.items())[: args.max_pairs])
    print(f"[dual_view_gt] Found {len(pairs)} T+S pairs with density GT under {labels_root}")

    args.output.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    rows: list[dict] = []
    for i, (pid, paths) in enumerate(pairs.items(), start=1):
        if i % 100 == 0 or i == len(pairs):
            elapsed = time.time() - t0
            print(f"[dual_view_gt] {i}/{len(pairs)} pairs "
                  f"({i / max(elapsed, 1e-6):.1f} pair/s)")
        rows.append(evaluate_pair(pid, paths, density_data))
    elapsed = time.time() - t0
    print(f"[dual_view_gt] Done in {elapsed:.1f}s")

    # per-pair CSV
    if rows:
        csv_path = args.output / "per_image_predictions.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[dual_view_gt] Wrote {csv_path}")

    # per-class
    for prefix in ("v1", "v2"):
        per_cls = aggregate_per_class(rows, prefix)
        cls_csv = args.output / f"per_class_metrics_{prefix}.csv"
        with cls_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["class", "n", "mape_kcal", "accuracy_at_20pct"])
            for cls, m in per_cls.items():
                w.writerow([cls, m["n"], m["mape_kcal"], m["accuracy_at_20pct"]])
        print(f"[dual_view_gt] Wrote {cls_csv}")

    # overall
    v1 = aggregate(rows, "v1")
    v2 = aggregate(rows, "v2")
    per_class_v1 = aggregate_per_class(rows, "v1")
    per_class_v2 = aggregate_per_class(rows, "v2")

    summary = {
        "n_pairs": len(rows),
        "labels_dir": str(args.labels_dir),
        "v1_single_image_top_only": v1,
        "v2_dual_image_top_side": v2,
        "per_class_v1": per_class_v1,
        "per_class_v2": per_class_v2,
        "elapsed_s": round(elapsed, 1),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[dual_view_gt] Wrote {args.output / 'summary.json'}")

    # comparison CSV
    cmp_csv = args.output / "comparison_1v_vs_2v.csv"
    with cmp_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["setup", "n", "mape_kcal", "mae_g", "accuracy_at_20pct", "notes"])
        w.writerow([
            "1_view_top_only", v1["n"], v1.get("mape_kcal"),
            v1.get("mae_g"), v1.get("accuracy_at_20pct"),
            "GT bbox + depth-ratio (per class) → ellipsoid/box/cyl volume.",
        ])
        w.writerow([
            "2_view_top_side", v2["n"], v2.get("mape_kcal"),
            v2.get("mae_g"), v2.get("accuracy_at_20pct"),
            "GT bbox on both views → real L×W×D silhouette volume.",
        ])
        w.writerow([
            "ECUSTFD_paper_baseline", 297, 0.189, None, None,
            "Liang & Li 2017, dual-camera + GrabCut. Reference only.",
        ])
    print(f"[dual_view_gt] Wrote {cmp_csv}")

    # comparison MD
    md = []
    md.append("# So sánh 1-ảnh (top) vs 2-ảnh (top + side) trên ECUSTFD\n\n")
    md.append(f"- Số cặp ảnh T+S có cả label top & side: **{len(rows)}**\n")
    md.append(f"- Detector: **ground-truth bbox** (loại trừ nhiễu detection để cô lập đóng góp của geometry)\n")
    md.append(f"- Density source: `data/density_processed.json` (per-class)\n")
    md.append(f"- Thời gian: **{elapsed:.1f}s**\n\n")
    md.append("## Overall metrics\n\n")
    md.append("| Setup | n | MAPE (kcal) | MAE (g) | Acc@20% | Ghi chú |\n")
    md.append("|---|---:|---:|---:|---:|---|\n")
    md.append(
        f"| 1 view (top only) | {v1['n']} | {_fmt_pct(v1.get('mape_kcal'))} | "
        f"{v1.get('mae_g', '—')} | {_fmt_pct(v1.get('accuracy_at_20pct'))} | "
        f"depth = min(W,H) × depth_ratio (per class) |\n"
    )
    md.append(
        f"| 2 views (top + side) | {v2['n']} | {_fmt_pct(v2.get('mape_kcal'))} | "
        f"{v2.get('mae_g', '—')} | {_fmt_pct(v2.get('accuracy_at_20pct'))} | "
        f"real L×W×D từ 2 silhouette |\n"
    )
    md.append(
        f"| ECUSTFD paper baseline (Liang & Li 2017) | 297 | 18.9% | — | — | "
        f"Dual-camera + GrabCut |\n"
    )
    md.append("\n## Per-class MAPE\n\n")
    md.append("| class | n (1v) | MAPE 1v | n (2v) | MAPE 2v | Δ (2v − 1v) |\n")
    md.append("|---|---:|---:|---:|---:|---:|\n")
    all_classes = sorted(set(per_class_v1) | set(per_class_v2))
    for cls in all_classes:
        m1 = per_class_v1.get(cls, {})
        m2 = per_class_v2.get(cls, {})
        delta = ""
        if m1.get("mape_kcal") is not None and m2.get("mape_kcal") is not None:
            delta = f"{(m2['mape_kcal'] - m1['mape_kcal']) * 100:+.2f} pp"
        md.append(
            f"| {cls} | {m1.get('n', '—')} | {_fmt_pct(m1.get('mape_kcal'))} | "
            f"{m2.get('n', '—')} | {_fmt_pct(m2.get('mape_kcal'))} | {delta} |\n"
        )
    md.append("\n## Nhận xét\n")
    if v1.get("mape_kcal") and v2.get("mape_kcal"):
        d = (v2["mape_kcal"] - v1["mape_kcal"]) * 100
        md.append(
            f"- 2-view pipeline cải thiện MAPE {d:+.2f} pp so với 1-view "
            f"({_fmt_pct(v1.get('mape_kcal'))} → {_fmt_pct(v2.get('mape_kcal'))}).\n"
        )
    md.append(
        "- Con số này là **upper bound** (detector hoàn hảo). Khi chạy với "
        "YOLOv13n inference thật, MAPE sẽ tệ hơn do sai số detection, nhưng "
        "delta giữa 1-view và 2-view vẫn phản ánh đúng đóng góp của geometry.\n"
    )
    md_path = args.output / "comparison_1v_vs_2v.md"
    md_path.write_text("".join(md), encoding="utf-8")
    print(f"[dual_view_gt] Wrote {md_path}")

    # error analysis
    valid_v2 = [r for r in rows if r["v2_abs_pct_err_kcal"] is not None]
    valid_v2.sort(key=lambda r: r["v2_abs_pct_err_kcal"], reverse=True)
    err_path = args.output / "error_analysis.md"
    with err_path.open("w", encoding="utf-8") as f:
        f.write("# Dual-view (GT bboxes) — error analysis\n\n")
        f.write(f"- Pairs evaluated: **{len(rows)}**\n")
        f.write(f"- v2 valid: **{len(valid_v2)}**\n\n")
        f.write("## Top-10 worst (by 2-view % kcal error)\n\n")
        f.write("| pair_id | class | GT kcal | v1 %err | v2 %err | Δ |\n")
        f.write("|---|---|---:|---:|---:|---:|\n")
        for r in valid_v2[:10]:
            d = ""
            if r["v1_abs_pct_err_kcal"] is not None and r["v2_abs_pct_err_kcal"] is not None:
                d = f"{(r['v2_abs_pct_err_kcal'] - r['v1_abs_pct_err_kcal']) * 100:+.2f} pp"
            f.write(
                f"| {r['pair_id']} | {r['class']} | {r['gt_kcal']:.1f} | "
                f"{_fmt_pct(r['v1_abs_pct_err_kcal'])} | "
                f"{_fmt_pct(r['v2_abs_pct_err_kcal'])} | {d} |\n"
            )
        f.write("\n## Top-10 best\n\n")
        f.write("| pair_id | class | GT kcal | v1 %err | v2 %err | Δ |\n")
        f.write("|---|---|---:|---:|---:|---:|\n")
        for r in sorted(valid_v2, key=lambda r: r["v2_abs_pct_err_kcal"])[:10]:
            d = ""
            if r["v1_abs_pct_err_kcal"] is not None and r["v2_abs_pct_err_kcal"] is not None:
                d = f"{(r['v2_abs_pct_err_kcal'] - r['v1_abs_pct_err_kcal']) * 100:+.2f} pp"
            f.write(
                f"| {r['pair_id']} | {r['class']} | {r['gt_kcal']:.1f} | "
                f"{_fmt_pct(r['v1_abs_pct_err_kcal'])} | "
                f"{_fmt_pct(r['v2_abs_pct_err_kcal'])} | {d} |\n"
            )
    print(f"[dual_view_gt] Wrote {err_path}")

    print()
    print("=" * 60)
    print("DUAL-VIEW (GT BBOX) COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  pairs             : {len(rows)}")
    print(f"  v1 (top) MAPE     : {_fmt_pct(v1.get('mape_kcal'))}")
    print(f"  v2 (top+side) MAPE: {_fmt_pct(v2.get('mape_kcal'))}")
    print(f"  v1 Acc@20%        : {_fmt_pct(v1.get('accuracy_at_20pct'))}")
    print(f"  v2 Acc@20%        : {_fmt_pct(v2.get('accuracy_at_20pct'))}")
    print()


if __name__ == "__main__":
    main()
