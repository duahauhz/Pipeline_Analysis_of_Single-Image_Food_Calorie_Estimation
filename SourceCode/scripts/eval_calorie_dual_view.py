"""Calorie estimation using BOTH top and side views (the original ECUSTFD setup).

Background
----------
The default pipeline in `calorie_estimator.py` works on a single image. The
ECUSTFD dataset, however, ships each fruit in two photos taken simultaneously:
* T (top view)   — looks like the existing 1-image setup
* S (side view)  — shows the vertical profile of the object

The 2017 paper [Liang & Li] used both views to estimate a real 3D volume
instead of guessing the third dimension from a fixed depth ratio. This script
re-implements that dual-view pipeline on top of the existing YOLOv13n
detector and the same `density_processed.json` GT data.

Pipeline
--------
For every `apple015` style id that has both `apple015T(N).JPG` and
`apple015S(N).JPG` files:

1. Run YOLOv13n on each view separately. Match the food box across views by
   class (top has at most one food, side also one). We keep the top-view
   bbox for the (width, height) and the side-view bbox for the (height, depth)
   footprint.
2. Calibrate the pixel↔mm scale on each view independently from the detected
   coin (25mm).
3. Derive `width_mm` and `depth_mm` from the side view (the silhouette width
   on the side is the real horizontal axis; the silhouette height is the real
   vertical axis). Use top view's silhouette for `length_mm` (the
   longest horizontal axis).
4. Build an ellipsoid (or box / cylinder) volume from these real three
   dimensions, *not* from a fixed depth ratio.
5. Compute mass = volume × density (per-class from `density_processed.json`)
   and calorie = mass × kcal/100g.

The script also runs the legacy single-image path on the same ids so the
output table can directly compare 1-view vs 2-view MAPE.

Output
------
runs/dual_view_eval/
  per_image_predictions.csv   — one row per image_id (paired)
  per_class_metrics.csv       — per class
  comparison_1v_vs_2v.csv     — overall comparison + ECUSTFD baseline
  comparison_1v_vs_2v.md       — markdown for the thesis
  summary.json                — all metrics in one place
  error_analysis.md            — top-10 worst / best ids

Usage (after ECUSTFD images are restored)
-----------------------------------------
    python scripts/eval_calorie_dual_view.py \\
        --source datasets/ECUSTFD/images/test \\
        --weights runs/local_food_detect/output/weights/best.pt \\
        --density-json data/density_processed.json \\
        --output runs/dual_view_eval
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


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from calorie_estimator import (  # noqa: E402
    COIN_CLASS_NAME,
    FOOD_INFO,
    IMAGE_EXTS,
    INVALID_CALIBRATION_STEMS,
    ensure_yolov13_import,
    image_id_from_stem,
    load_density_data,
    parse_detections,
    resolve_food_estimate,
    volume_cm3,
)


# --------------------------------------------------------------------------- #
# Pairing logic
# --------------------------------------------------------------------------- #

VIEW_TRAIL_RE = re.compile(r"([TS])\(\d+\)$", re.IGNORECASE)
TRAIL_RE = re.compile(r"([TS]?)\(\d+\)$", re.IGNORECASE)


def view_of_stem(stem: str) -> str | None:
    """Return 'T' (top), 'S' (side) or None if no view suffix."""
    m = VIEW_TRAIL_RE.search(stem)
    if m:
        return m.group(1).upper()
    return None


def pair_id_of_stem(stem: str) -> str:
    """Return the shared id (e.g. `apple015S(1)` → `apple015`)."""
    return image_id_from_stem(stem)


def discover_pairs(test_dir: Path) -> dict[str, dict[str, Path]]:
    """Walk the test tree and return `{pair_id: {'T': path, 'S': path}}`.

    A pair is only kept if BOTH views are present (the whole point of this
    experiment). Stems listed in `INVALID_CALIBRATION_STEMS` are skipped.
    """
    if not test_dir.exists():
        return {}
    pairs: dict[str, dict[str, Path]] = {}
    for path in sorted(test_dir.rglob("*")):
        if not path.is_file() or path.suffix not in IMAGE_EXTS:
            continue
        if path.stem in INVALID_CALIBRATION_STEMS:
            continue
        view = view_of_stem(path.stem)
        if view is None:
            continue
        pair_id = pair_id_of_stem(path.stem)
        pairs.setdefault(pair_id, {})[view] = path
    # keep only those with both
    return {pid: v for pid, v in pairs.items() if "T" in v and "S" in v}


# --------------------------------------------------------------------------- #
# Detection cache — run YOLO once per image, reuse results across both modes
# --------------------------------------------------------------------------- #


def run_detections(model, image_path: Path, conf: float, imgsz: int) -> list[dict]:
    result = model.predict(
        source=str(image_path),
        conf=conf,
        imgsz=imgsz,
        verbose=False,
    )[0]
    return parse_detections(result, image_path)


def pick_coin_px(detections: list[dict]) -> float | None:
    coins = [d for d in detections if d["class_name"] == COIN_CLASS_NAME]
    if not coins:
        return None
    coin = max(coins, key=lambda d: d["confidence"])
    return (coin["width_px"] + coin["height_px"]) / 2.0


def pick_food(detections: list[dict], expected_class: str | None = None) -> dict | None:
    foods = [d for d in detections if d["class_name"] in FOOD_INFO]
    if expected_class:
        same = [d for d in foods if d["class_name"] == expected_class]
        if same:
            foods = same
    if not foods:
        return None
    return max(foods, key=lambda d: d["confidence"])


# --------------------------------------------------------------------------- #
# Dual-view volume model
# --------------------------------------------------------------------------- #


def dual_view_dimensions_mm(
    top_food: dict,
    side_food: dict,
    top_coin_px: float,
    side_coin_px: float,
    coin_diameter_mm: float,
) -> dict:
    """Compute real (length, width, depth) in mm from the two silhouettes.

    Conventions
    -----------
    * Top view: silhouette spans the X (length) and Y (width) axes of the food.
      We use `width_px × height_px` of the top-view bbox as (X, Y) of the
      food after dividing by `mm_per_pixel_top`.
    * Side view: silhouette spans the Z (height) and either X (length) or
      Y (width) axis depending on orientation. The bigger of the two side-view
      axes is matched with the larger of the top-view axes (length) and the
      smaller with the other (width/height), so a banana lying on its side
      gets its long axis from the side view, not the round cross-section.
    * The remaining axis (depth) is the smaller of the side-view axes, which
      is the third dimension the top view can't see.

    Result
    ------
    Returns dict with `length_mm`, `width_mm`, `depth_mm`, both mm_per_pixel
    values, and a `side_orientation` flag describing how the side bbox was
    decomposed (informational, used to compute the side_geometry_axis swap).
    """
    top_mm_per_px = coin_diameter_mm / top_coin_px if top_coin_px > 0 else None
    side_mm_per_px = coin_diameter_mm / side_coin_px if side_coin_px > 0 else None
    if top_mm_per_px is None or side_mm_per_px is None:
        return {
            "ok": False,
            "reason": "missing_coin_scale",
            "top_mm_per_px": top_mm_per_px,
            "side_mm_per_px": side_mm_per_px,
        }

    top_x_mm = top_food["width_px"] * top_mm_per_px
    top_y_mm = top_food["height_px"] * top_mm_per_px
    side_x_mm = side_food["width_px"] * side_mm_per_px
    side_y_mm = side_food["height_px"] * side_mm_per_px

    # The side view height (Z) is whichever side axis is the taller one:
    # a fruit standing up will have the long side axis = Z, a fruit on its
    # side will have the long axis still = Z if it is roughly spherical.
    # In either case, the *vertical* silhouette of the side view is Z. We
    # cannot tell from a single image which side axis is the vertical one, so
    # we take the LARGER side axis as the vertical (height of the food) and
    # the smaller as the depth. This matches the typical setup: a fruit
    # sitting on a plate has a vertical axis (the "height" you see from the
    # side) that is ≥ its depth.
    side_axes = sorted([side_x_mm, side_y_mm], reverse=True)
    side_height_mm = side_axes[0]
    side_depth_mm = side_axes[1]

    # Now match the top-view axes to length and width. The top view's
    # longer axis is the length, the shorter is the width.
    top_axes = sorted([top_x_mm, top_y_mm], reverse=True)
    length_mm = top_axes[0]
    width_mm = top_axes[1]

    return {
        "ok": True,
        "length_mm": length_mm,
        "width_mm": width_mm,
        "depth_mm": side_depth_mm,
        "height_mm": side_height_mm,
        "top_mm_per_px": top_mm_per_px,
        "side_mm_per_px": side_mm_per_px,
        "top_x_mm": top_x_mm,
        "top_y_mm": top_y_mm,
        "side_x_mm": side_x_mm,
        "side_y_mm": side_y_mm,
    }


def dual_view_volume_cm3(dims: dict, geometry: str) -> float:
    """Compute volume from real (length, width, depth) in mm.

    The side-view vertical axis is also passed in but is *not* used to fit the
    geometry — we already have enough information with length/width/depth.
    """
    return volume_cm3(dims["length_mm"], dims["width_mm"], dims["depth_mm"], geometry)


# --------------------------------------------------------------------------- #
# Per-id evaluation
# --------------------------------------------------------------------------- #


def gt_record_for_id(image_id: str, density_data: dict) -> dict | None:
    return density_data["per_image"].get(image_id)


def gt_kcal_for_id(image_id: str, density_data: dict) -> float | None:
    rec = gt_record_for_id(image_id, density_data)
    if rec is None:
        return None
    cls = rec["class"]
    info = FOOD_INFO.get(cls)
    if info is None:
        return None
    return rec["weight_g"] * info["kcal_per_100g"] / 100.0


def evaluate_pair(
    pair_id: str,
    paths: dict[str, Path],
    detections_cache: dict[Path, list[dict]],
    model,
    density_data: dict,
    coin_diameter_mm: float,
    conf: float,
    imgsz: int,
) -> dict:
    """Run both 1-view (top only) and 2-view (top + side) estimates for one pair."""
    top_path = paths["T"]
    side_path = paths["S"]

    # Use cached detections if available (else run)
    if top_path in detections_cache:
        top_dets = detections_cache[top_path]
    else:
        top_dets = run_detections(model, top_path, conf, imgsz)
        detections_cache[top_path] = top_dets
    if side_path in detections_cache:
        side_dets = detections_cache[side_path]
    else:
        side_dets = run_detections(model, side_path, conf, imgsz)
        detections_cache[side_path] = side_dets

    gt = gt_record_for_id(pair_id, density_data)
    expected_class = gt["class"] if gt else None
    gt_kcal = gt_kcal_for_id(pair_id, density_data) if gt else None

    row = {
        "pair_id": pair_id,
        "class": expected_class,
        "top_image": str(top_path),
        "side_image": str(side_path),
        "has_gt": gt is not None,
        "gt_mass_g": gt["weight_g"] if gt else None,
        "gt_volume_cm3": gt["volume_cm3"] if gt else None,
        "gt_kcal": gt_kcal,
        # 1-view results
        "v1_predicted_mass_g": None,
        "v1_predicted_kcal": None,
        "v1_predicted_volume_cm3": None,
        "v1_abs_err_kcal": None,
        "v1_abs_pct_err_kcal": None,
        "v1_status": "ok",
        # 2-view results
        "v2_predicted_mass_g": None,
        "v2_predicted_kcal": None,
        "v2_predicted_volume_cm3": None,
        "v2_abs_err_kcal": None,
        "v2_abs_pct_err_kcal": None,
        "v2_status": "ok",
        # diagnostic
        "top_coin_px": pick_coin_px(top_dets),
        "side_coin_px": pick_coin_px(side_dets),
        "top_food_class": None,
        "side_food_class": None,
        "length_mm": None,
        "width_mm": None,
        "depth_mm": None,
        "height_mm": None,
    }

    # ---- 1-view path (top image only) ----
    top_coin_px = pick_coin_px(top_dets)
    top_food = pick_food(top_dets, expected_class)
    if top_food is not None:
        row["top_food_class"] = top_food["class_name"]
    if top_coin_px is None or top_food is None:
        row["v1_status"] = "missing_coin_or_food"
    else:
        mm_per_px = coin_diameter_mm / top_coin_px
        # Use the existing single-image resolver. We force per-class path by
        # stripping per_image entries — that's what `eval_calorie.py` does to
        # get a real end-to-end number.
        eval_density = {**density_data, "per_image": {}}
        est = resolve_food_estimate(top_food["class_name"], top_food, mm_per_px, eval_density)
        row["v1_predicted_mass_g"] = est["mass_g"]
        row["v1_predicted_kcal"] = est["calorie_kcal"]
        row["v1_predicted_volume_cm3"] = est["volume_cm3"]

    # ---- 2-view path ----
    side_coin_px = pick_coin_px(side_dets)
    side_food = pick_food(side_dets, expected_class)
    if side_food is not None:
        row["side_food_class"] = side_food["class_name"]
    if top_coin_px is None or side_coin_px is None or top_food is None or side_food is None:
        row["v2_status"] = "missing_coin_or_food"
    else:
        # The two views should detect the same class, but if not, fall back
        # to the class with a FOOD_INFO entry.
        cls_name = top_food["class_name"]
        if cls_name != side_food["class_name"] and side_food["class_name"] in FOOD_INFO:
            cls_name = side_food["class_name"]
        info = FOOD_INFO.get(cls_name)
        if info is None:
            row["v2_status"] = "unknown_class"
        else:
            dims = dual_view_dimensions_mm(
                top_food, side_food, top_coin_px, side_coin_px, coin_diameter_mm
            )
            if not dims["ok"]:
                row["v2_status"] = dims.get("reason", "dims_failed")
            else:
                volume = dual_view_volume_cm3(dims, info["geometry"])
                density = info["density"]
                # Prefer per-class density from density_processed.json if present
                per_class = density_data.get("per_class", {}).get(cls_name, {})
                if per_class.get("mean_density_g_cm3", 0) > 0:
                    density = per_class["mean_density_g_cm3"]
                mass_g = volume * density
                kcal = mass_g * info["kcal_per_100g"] / 100.0
                row["v2_predicted_mass_g"] = mass_g
                row["v2_predicted_kcal"] = kcal
                row["v2_predicted_volume_cm3"] = volume
                row["length_mm"] = dims["length_mm"]
                row["width_mm"] = dims["width_mm"]
                row["depth_mm"] = dims["depth_mm"]
                row["height_mm"] = dims["height_mm"]

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


def aggregate(rows: list[dict], key_prefix: str) -> dict:
    pcts = [r[f"{key_prefix}_abs_pct_err_kcal"] for r in rows
            if r[f"{key_prefix}_abs_pct_err_kcal"] is not None]
    errs_g = []
    for r in rows:
        if r["has_gt"] and r["gt_mass_g"] and r[f"{key_prefix}_predicted_mass_g"] is not None:
            errs_g.append(abs(r[f"{key_prefix}_predicted_mass_g"] - r["gt_mass_g"]))
    n = len(pcts)
    if n == 0:
        return {"n": 0}
    mape = sum(pcts) / n
    mae_g = sum(errs_g) / len(errs_g) if errs_g else None
    return {
        "n": n,
        "mape_kcal": round(mape, 4),
        "mae_g": round(mae_g, 2) if mae_g is not None else None,
        "accuracy_at_20pct": round(sum(1 for p in pcts if p <= 0.20) / n, 4),
    }


def aggregate_per_class(rows: list[dict], key_prefix: str) -> dict:
    out: dict[str, dict] = {}
    by_cls: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r["class"] is None:
            continue
        pct = r[f"{key_prefix}_abs_pct_err_kcal"]
        if pct is None:
            continue
        by_cls[r["class"]].append(pct)
    for cls, pcts in sorted(by_cls.items()):
        if not pcts:
            continue
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
    p.add_argument("--source", type=Path, default=Path("datasets/ECUSTFD/images/test"))
    p.add_argument("--weights", type=Path, default=Path("weights/yolov13n_ecustfd_best.pt"))
    p.add_argument("--repo", type=Path, default=Path("yolov13"))
    p.add_argument("--density-json", type=Path, default=Path("data/density_processed.json"))
    p.add_argument("--output", type=Path, default=Path("runs/dual_view_eval"))
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--coin-diameter-mm", type=float, default=25.0)
    p.add_argument("--limit-class", type=str, default="")
    p.add_argument("--max-pairs", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_yolov13_import(args.repo)
    if not args.weights.exists():
        raise FileNotFoundError(f"Missing weights: {args.weights}")
    if not args.source.exists():
        raise FileNotFoundError(
            f"Missing source: {args.source}. Restore the ECUSTFD images under "
            f"datasets/ECUSTFD/images/test/ and re-run."
        )

    density_data = load_density_data(args.density_json)
    print(f"[dual_view] density per_image={len(density_data['per_image'])} "
          f"per_class={len(density_data['per_class'])}")

    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(".yolo-config").resolve()))
    os.environ.setdefault("YOLO_OFFLINE", "true")

    from ultralytics import YOLO
    model = YOLO(str(args.weights))

    pairs = discover_pairs(args.source)
    if args.limit_class:
        pairs = {pid: v for pid, v in pairs.items() if pid.startswith(args.limit_class)}
    if args.max_pairs:
        pairs = dict(list(pairs.items())[: args.max_pairs])

    print(f"[dual_view] Found {len(pairs)} complete T+S pairs")
    args.output.mkdir(parents=True, exist_ok=True)

    detections_cache: dict[Path, list[dict]] = {}
    rows: list[dict] = []
    t0 = time.time()
    for i, (pair_id, paths) in enumerate(pairs.items(), start=1):
        if i % 50 == 0 or i == len(pairs):
            elapsed = time.time() - t0
            print(f"[dual_view] {i}/{len(pairs)} pairs "
                  f"({i / max(elapsed, 1e-6):.1f} pair/s)")
        row = evaluate_pair(
            pair_id, paths, detections_cache, model, density_data,
            args.coin_diameter_mm, args.conf, args.imgsz,
        )
        rows.append(row)
    elapsed = time.time() - t0
    print(f"[dual_view] Done in {elapsed:.1f}s")

    # per-pair CSV
    csv_path = args.output / "per_image_predictions.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[dual_view] Wrote {csv_path}")

    # per-class CSVs (1v and 2v)
    for prefix in ("v1", "v2"):
        per_cls = aggregate_per_class(rows, prefix)
        cls_csv = args.output / f"per_class_metrics_{prefix}.csv"
        with cls_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["class", "n", "mape_kcal", "accuracy_at_20pct"])
            for cls, m in per_cls.items():
                w.writerow([cls, m["n"], m["mape_kcal"], m["accuracy_at_20pct"]])
        print(f"[dual_view] Wrote {cls_csv}")

    # overall comparison
    overall_v1 = aggregate(rows, "v1")
    overall_v2 = aggregate(rows, "v2")
    per_class_v1 = aggregate_per_class(rows, "v1")
    per_class_v2 = aggregate_per_class(rows, "v2")

    summary = {
        "n_pairs": len(rows),
        "v1_single_image": overall_v1,
        "v2_dual_image": overall_v2,
        "per_class_v1": per_class_v1,
        "per_class_v2": per_class_v2,
        "elapsed_s": round(elapsed, 1),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[dual_view] Wrote {args.output / 'summary.json'}")

    # comparison table
    cmp_csv = args.output / "comparison_1v_vs_2v.csv"
    with cmp_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["setup", "n", "mape_kcal", "mae_g", "accuracy_at_20pct", "notes"])
        w.writerow([
            "1_view_top_only", overall_v1["n"], overall_v1.get("mape_kcal"),
            overall_v1.get("mae_g"), overall_v1.get("accuracy_at_20pct"),
            "Reproduces existing pipeline on top view only (paired ids).",
        ])
        w.writerow([
            "2_view_top+side", overall_v2["n"], overall_v2.get("mape_kcal"),
            overall_v2.get("mae_g"), overall_v2.get("accuracy_at_20pct"),
            "New dual-view pipeline using real L×W×D from silhouettes.",
        ])
        w.writerow([
            "ECUSTFD_baseline_paper", 297, 0.189, None, None,
            "Liang & Li 2017, dual-camera + multi-view geometry (reference).",
        ])
    print(f"[dual_view] Wrote {cmp_csv}")

    # comparison markdown
    md = []
    md.append("# So sánh 1-ảnh (top) vs 2-ảnh (top + side) trên ECUSTFD\n")
    md.append(f"- Số cặp ảnh T+S đủ điều kiện: **{len(rows)}**\n")
    md.append(f"- Detector: YOLOv13n (đã train local)\n")
    md.append(f"- Density source: `data/density_processed.json` (per_class path)\n")
    md.append(f"- Thời gian chạy: **{elapsed:.1f}s**\n\n")
    md.append("## Overall metrics\n")
    md.append("| Setup | n | MAPE (kcal) | MAE (g) | Acc@20% | Ghi chú |\n")
    md.append("|---|---:|---:|---:|---:|---|\n")
    md.append(
        f"| 1 view (top only) | {overall_v1['n']} | "
        f"{_fmt_pct(overall_v1.get('mape_kcal'))} | "
        f"{_fmt(overall_v1.get('mae_g'))} | "
        f"{_fmt_pct(overall_v1.get('accuracy_at_20pct'))} | "
        f"Pipeline hiện tại |\n"
    )
    md.append(
        f"| 2 views (top + side) | {overall_v2['n']} | "
        f"{_fmt_pct(overall_v2.get('mape_kcal'))} | "
        f"{_fmt(overall_v2.get('mae_g'))} | "
        f"{_fmt_pct(overall_v2.get('accuracy_at_20pct'))} | "
        f"Dual-view silhouette → thể tích ellipsoid/box/cylinder thật |\n"
    )
    md.append(
        f"| ECUSTFD paper baseline | 297 | 18.9% | — | — | "
        f"Liang & Li 2017, dual-camera chuyên dụng |\n"
    )
    md.append("\n## Per-class MAPE\n")
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
    if overall_v1.get("mape_kcal") and overall_v2.get("mape_kcal"):
        d = (overall_v2["mape_kcal"] - overall_v1["mape_kcal"]) * 100
        md.append(
            f"- 2-view pipeline cải thiện MAPE {d:+.2f} pp so với 1-view "
            f"({_fmt_pct(overall_v1.get('mape_kcal'))} → "
            f"{_fmt_pct(overall_v2.get('mape_kcal'))}).\n"
        )
    md.append(
        "- So sánh với paper gốc (18.9%) là so sánh tham chiếu, không phải "
        "mục tiêu cạnh tranh trực tiếp — paper dùng 2 camera chuyên dụng "
        "chụp đồng thời, hệ thống này dùng detector học được từ dữ liệu.\n"
    )
    md_path = args.output / "comparison_1v_vs_2v.md"
    md_path.write_text("".join(md), encoding="utf-8")
    print(f"[dual_view] Wrote {md_path}")

    # error analysis
    err_path = args.output / "error_analysis.md"
    valid_v2 = [r for r in rows if r["v2_abs_pct_err_kcal"] is not None]
    valid_v2.sort(key=lambda r: r["v2_abs_pct_err_kcal"], reverse=True)
    with err_path.open("w", encoding="utf-8") as f:
        f.write("# Dual-view evaluation — error analysis\n\n")
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
        f.write("\n## Top-10 best (by 2-view % kcal error)\n\n")
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
    print(f"[dual_view] Wrote {err_path}")

    # final summary on stdout
    print()
    print("=" * 60)
    print("DUAL-VIEW COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  pairs             : {len(rows)}")
    print(f"  v1 (top only) MAPE: {_fmt_pct(overall_v1.get('mape_kcal'))}")
    print(f"  v2 (top+side) MAPE: {_fmt_pct(overall_v2.get('mape_kcal'))}")
    print(f"  v1 Acc@20%        : {_fmt_pct(overall_v1.get('accuracy_at_20pct'))}")
    print(f"  v2 Acc@20%        : {_fmt_pct(overall_v2.get('accuracy_at_20pct'))}")
    print()


def _fmt(v) -> str:
    if v is None:
        return "—"
    return f"{v}"


def _fmt_pct(v) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.2f}%"


if __name__ == "__main__":
    main()
