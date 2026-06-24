"""Calorie estimation from YOLOv13 food and coin boxes, using density.xls for
real per-image ground truth where available.

This is the refactored v1 of the prototype. The hard-coded `FOOD_INFO` is kept
as a *fallback for nutrition (kcal/100g)* only — geometry, depth ratio, and
density now come from the workbook-driven `density_processed.json`.

Resolution order for a detected food
------------------------------------
1. If the image id (e.g. `apple015S(1).JPG` → `apple015`) is in
   `per_image` of `density_processed.json`, use its real volume + weight.
2. Otherwise fall back to `per_class` mean volume/density for that class.
3. kcal/100g still comes from `FOOD_INFO`; if the class is missing there, we
   set the calorie to 0 and report `kcal_source="missing"`.

Usage
-----
    python scripts/calorie_estimator.py \\
        --source datasets/ECUSTFD/images/test \\
        --weights runs/local_food_detect/output/weights/best.pt \\
        --max-images 8 \\
        --density-json data/density_processed.json
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
INVALID_CALIBRATION_STEMS = {"mix002T(2)", "mix005S(4)"}
COIN_CLASS_NAME = "coin"


# Nutrition source: kcal/100g + geometric model. The geometry and depth ratio
# are *not* used when density.xls provides real volume/weight per image. They
# are still used as a fallback when an image id is missing from the workbook.
FOOD_INFO = {
    "apple": {"density": 0.84, "kcal_per_100g": 52, "geometry": "ellipsoid", "depth_ratio": 0.85},
    "banana": {"density": 0.94, "kcal_per_100g": 89, "geometry": "ellipsoid", "depth_ratio": 0.55},
    "bread": {"density": 0.27, "kcal_per_100g": 265, "geometry": "box", "depth_ratio": 0.45},
    "bun": {"density": 0.35, "kcal_per_100g": 223, "geometry": "ellipsoid", "depth_ratio": 0.45},
    "doughnut": {"density": 0.33, "kcal_per_100g": 452, "geometry": "cylinder", "depth_ratio": 0.35},
    "egg": {"density": 1.03, "kcal_per_100g": 155, "geometry": "ellipsoid", "depth_ratio": 0.75},
    "fired_dough_twist": {"density": 0.35, "kcal_per_100g": 450, "geometry": "cylinder", "depth_ratio": 0.35},
    "grape": {"density": 0.96, "kcal_per_100g": 69, "geometry": "ellipsoid", "depth_ratio": 0.90},
    "lemon": {"density": 0.96, "kcal_per_100g": 29, "geometry": "ellipsoid", "depth_ratio": 0.85},
    "litchi": {"density": 1.00, "kcal_per_100g": 66, "geometry": "ellipsoid", "depth_ratio": 0.90},
    "mango": {"density": 0.95, "kcal_per_100g": 60, "geometry": "ellipsoid", "depth_ratio": 0.70},
    "mooncake": {"density": 0.65, "kcal_per_100g": 420, "geometry": "cylinder", "depth_ratio": 0.35},
    "orange": {"density": 0.87, "kcal_per_100g": 47, "geometry": "ellipsoid", "depth_ratio": 0.90},
    "peach": {"density": 0.95, "kcal_per_100g": 39, "geometry": "ellipsoid", "depth_ratio": 0.85},
    "pear": {"density": 0.92, "kcal_per_100g": 57, "geometry": "ellipsoid", "depth_ratio": 0.80},
    "plum": {"density": 0.95, "kcal_per_100g": 46, "geometry": "ellipsoid", "depth_ratio": 0.85},
    "qiwi": {"density": 1.00, "kcal_per_100g": 61, "geometry": "ellipsoid", "depth_ratio": 0.80},
    "sachima": {"density": 0.38, "kcal_per_100g": 450, "geometry": "box", "depth_ratio": 0.40},
    "tomato": {"density": 0.95, "kcal_per_100g": 18, "geometry": "ellipsoid", "depth_ratio": 0.85},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("datasets/ECUSTFD/images/test"))
    parser.add_argument("--weights", type=Path, default=Path("weights/yolov13n_ecustfd_best.pt"))
    parser.add_argument("--repo", type=Path, default=Path("yolov13"))
    parser.add_argument("--output", type=Path, default=Path("runs/local_food_detect/calorie_estimation"))
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--coin-diameter-mm", type=float, default=25.0)
    parser.add_argument(
        "--density-json",
        type=Path,
        default=Path("data/density_processed.json"),
        help="Output of scripts/parse_density_xls.py. Per-image and per-class densities.",
    )
    return parser.parse_args()


def class_key(path: Path) -> str:
    match = re.match(r"^[A-Za-z_]+", path.stem)
    return match.group(0) if match else path.stem


def image_id_from_stem(stem: str) -> str:
    """Map `apple015S(1)` → `apple015`, `mix001T(2)` → `mix001`."""
    # Strip the (N) suffix and the trailing single letter S/T.
    cleaned = re.sub(r"\(\d+\)$", "", stem)
    cleaned = re.sub(r"[ST]$", "", cleaned)
    return cleaned


def collect_images(source: Path, max_images: int) -> list[Path]:
    if source.is_file():
        return [source]

    images = [
        path
        for path in sorted(source.rglob("*"))
        if path.is_file() and path.suffix in IMAGE_EXTS and path.stem not in INVALID_CALIBRATION_STEMS
    ]
    if len(images) <= max_images:
        return images

    selected = []
    seen_classes = set()
    for image in images:
        key = class_key(image)
        if key not in seen_classes:
            selected.append(image)
            seen_classes.add(key)
        if len(selected) >= max_images:
            return selected

    for image in images:
        if image not in selected:
            selected.append(image)
        if len(selected) >= max_images:
            break
    return selected


def ensure_yolov13_import(repo: Path) -> None:
    repo = repo.resolve()
    if not (repo / "ultralytics" / "__init__.py").exists():
        raise FileNotFoundError(f"Missing YOLOv13 repo: {repo}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


def load_density_data(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing density JSON: {path}. Run `python scripts/parse_density_xls.py` first."
        )
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def parse_detections(result, image_path: Path) -> list[dict]:
    names = result.names
    rows = []
    if result.boxes is None:
        return rows

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    for idx, (box, conf, cls_id) in enumerate(zip(boxes, confs, classes), start=1):
        class_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
        x1, y1, x2, y2 = [float(value) for value in box]
        rows.append(
            {
                "image": str(image_path),
                "detection_id": idx,
                "class_id": int(cls_id),
                "class_name": class_name,
                "confidence": float(conf),
                "bbox": [x1, y1, x2, y2],
                "width_px": x2 - x1,
                "height_px": y2 - y1,
            }
        )
    return rows


def volume_cm3(width_mm: float, height_mm: float, depth_mm: float, geometry: str) -> float:
    width_cm = width_mm / 10.0
    height_cm = height_mm / 10.0
    depth_cm = depth_mm / 10.0

    if geometry == "box":
        return width_cm * height_cm * depth_cm
    if geometry == "cylinder":
        return math.pi * (width_cm / 2.0) * (depth_cm / 2.0) * height_cm
    return (4.0 / 3.0) * math.pi * (width_cm / 2.0) * (height_cm / 2.0) * (depth_cm / 2.0)


def estimate_food_geometry(
    food_detection: dict, mm_per_pixel: float, info: dict
) -> tuple[float, float]:
    """Estimate volume and mass using the geometric model (no per-image GT)."""
    width_mm = food_detection["width_px"] * mm_per_pixel
    height_mm = food_detection["height_px"] * mm_per_pixel
    depth_mm = min(width_mm, height_mm) * info["depth_ratio"]
    volume = volume_cm3(width_mm, height_mm, depth_mm, info["geometry"])
    mass_g = volume * info["density"]
    return volume, mass_g


def resolve_food_estimate(
    class_name: str,
    food_detection: dict,
    mm_per_pixel: float,
    density_data: dict,
) -> dict:
    """Resolve a food's volume/mass/calorie from density.json (per-image → per-class) + FOOD_INFO for nutrition.

    Returns a dict with extra fields:
      - `volume_source`: "per_image" | "per_class" | "geometry_fallback"
      - `kcal_source`: "food_info" | "missing"
    """
    image_id = image_id_from_stem(Path(food_detection["image"]).stem)
    per_image = density_data.get("per_image", {})
    per_class = density_data.get("per_class", {})

    info = FOOD_INFO.get(class_name)
    if info is None:
        return {
            **food_detection,
            "class": class_name,
            "width_mm": food_detection["width_px"] * mm_per_pixel,
            "height_mm": food_detection["height_px"] * mm_per_pixel,
            "depth_mm": None,
            "geometry": None,
            "density_g_cm3": None,
            "kcal_per_100g": None,
            "volume_cm3": 0.0,
            "mass_g": 0.0,
            "calorie_kcal": 0.0,
            "volume_source": "missing_class",
            "kcal_source": "missing",
        }

    # Per-image path: use the workbook's volume/weight directly.
    if image_id in per_image:
        gt = per_image[image_id]
        volume = gt["volume_cm3"]
        mass_g = gt["weight_g"]
        density = gt["density_g_cm3"]
        kcal = mass_g * info["kcal_per_100g"] / 100.0
        return {
            **food_detection,
            "class": class_name,
            "width_mm": food_detection["width_px"] * mm_per_pixel,
            "height_mm": food_detection["height_px"] * mm_per_pixel,
            "depth_mm": None,
            "geometry": "ground_truth_from_density.xls",
            "density_g_cm3": density,
            "kcal_per_100g": info["kcal_per_100g"],
            "volume_cm3": volume,
            "mass_g": mass_g,
            "calorie_kcal": kcal,
            "volume_source": "per_image",
            "kcal_source": "food_info",
        }

    # Per-class path: use the class's mean volume/weight but re-derive the
    # mass via the detected bbox's mm_per_pixel × class mean density. This
    # gives a per-class prior with a *detected* linear scale, which is closer
    # to what the geometric model does than using the workbook mass directly.
    cls_stats = per_class.get(class_name)
    if cls_stats is not None and cls_stats.get("mean_density_g_cm3", 0) > 0:
        # If we don't have a per-image volume, use the geometric model with
        # the *workbook's* density so the mass is consistent with the class.
        width_mm = food_detection["width_px"] * mm_per_pixel
        height_mm = food_detection["height_px"] * mm_per_pixel
        depth_mm = min(width_mm, height_mm) * info["depth_ratio"]
        volume = volume_cm3(width_mm, height_mm, depth_mm, info["geometry"])
        mass_g = volume * cls_stats["mean_density_g_cm3"]
        kcal = mass_g * info["kcal_per_100g"] / 100.0
        return {
            **food_detection,
            "class": class_name,
            "width_mm": width_mm,
            "height_mm": height_mm,
            "depth_mm": depth_mm,
            "geometry": info["geometry"],
            "density_g_cm3": cls_stats["mean_density_g_cm3"],
            "kcal_per_100g": info["kcal_per_100g"],
            "volume_cm3": volume,
            "mass_g": mass_g,
            "calorie_kcal": kcal,
            "volume_source": "per_class",
            "kcal_source": "food_info",
        }

    # Final fallback: pure geometric model.
    volume, mass_g = estimate_food_geometry(food_detection, mm_per_pixel, info)
    kcal = mass_g * info["kcal_per_100g"] / 100.0
    return {
        **food_detection,
        "class": class_name,
        "width_mm": food_detection["width_px"] * mm_per_pixel,
        "height_mm": food_detection["height_px"] * mm_per_pixel,
        "depth_mm": min(
            food_detection["width_px"] * mm_per_pixel,
            food_detection["height_px"] * mm_per_pixel,
        ) * info["depth_ratio"],
        "geometry": info["geometry"],
        "density_g_cm3": info["density"],
        "kcal_per_100g": info["kcal_per_100g"],
        "volume_cm3": volume,
        "mass_g": mass_g,
        "calorie_kcal": kcal,
        "volume_source": "geometry_fallback",
        "kcal_source": "food_info",
    }


def annotate_summary(image, status: str, total_kcal: float | None) -> None:
    import cv2

    if total_kcal is None:
        text = f"Calorie prototype: {status}"
    else:
        text = f"Calorie prototype: {total_kcal:.1f} kcal"
    cv2.rectangle(image, (8, 8), (620, 42), (0, 0, 0), -1)
    cv2.putText(image, text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)


def write_food_csv(rows: list[dict], output_path: Path) -> None:
    fieldnames = [
        "image",
        "status",
        "class_name",
        "confidence",
        "width_px",
        "height_px",
        "width_mm",
        "height_mm",
        "depth_mm",
        "geometry",
        "density_g_cm3",
        "kcal_per_100g",
        "volume_cm3",
        "mass_g",
        "calorie_kcal",
        "volume_source",
        "kcal_source",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def main() -> None:
    args = parse_args()
    ensure_yolov13_import(args.repo)
    if not args.weights.exists():
        raise FileNotFoundError(f"Missing weights: {args.weights}")
    if not args.source.exists():
        raise FileNotFoundError(f"Missing source: {args.source}")

    density_data = load_density_data(args.density_json)
    print(
        f"[calorie_estimator] Loaded density data: "
        f"{len(density_data.get('per_image', {}))} per-image, "
        f"{len(density_data.get('per_class', {}))} per-class."
    )

    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(".yolo-config").resolve()))
    os.environ.setdefault("YOLO_OFFLINE", "true")

    import cv2
    from ultralytics import YOLO

    args.output.mkdir(parents=True, exist_ok=True)
    images = collect_images(args.source, args.max_images)
    model = YOLO(str(args.weights))

    summaries = []
    food_rows = []
    for image_path in images:
        if image_path.stem in INVALID_CALIBRATION_STEMS:
            summaries.append({"image": str(image_path), "status": "skipped_invalid_calibration"})
            continue

        result = model.predict(source=str(image_path), conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        detections = parse_detections(result, image_path)
        coins = [det for det in detections if det["class_name"] == COIN_CLASS_NAME]
        foods = [det for det in detections if det["class_name"] in FOOD_INFO]

        annotated = result.plot()
        total_kcal = None
        status = "ok"
        estimated_foods = []
        coin = None
        mm_per_pixel = None

        if not coins:
            status = "missing_coin"
        elif not foods:
            status = "no_food"
            coin = max(coins, key=lambda item: item["confidence"])
            coin_px = (coin["width_px"] + coin["height_px"]) / 2.0
            mm_per_pixel = args.coin_diameter_mm / coin_px if coin_px > 0 else None
        else:
            coin = max(coins, key=lambda item: item["confidence"])
            coin_px = (coin["width_px"] + coin["height_px"]) / 2.0
            mm_per_pixel = args.coin_diameter_mm / coin_px if coin_px > 0 else None
            if mm_per_pixel is None:
                status = "invalid_coin_bbox"
            else:
                estimated_foods = [
                    resolve_food_estimate(food["class_name"], food, mm_per_pixel, density_data)
                    for food in foods
                ]
                total_kcal = sum(food["calorie_kcal"] for food in estimated_foods)

        for food in estimated_foods:
            food_rows.append({"status": status, **food})

        annotate_summary(annotated, status, total_kcal)
        output_image = args.output / f"{image_path.stem}_calorie.jpg"
        cv2.imwrite(str(output_image), annotated)

        summaries.append(
            {
                "image": str(image_path),
                "image_id": image_id_from_stem(image_path.stem),
                "output_image": str(output_image),
                "status": status,
                "coin": coin,
                "mm_per_pixel": mm_per_pixel,
                "foods": estimated_foods,
                "total_calorie_kcal": total_kcal,
                "note": "v1: density from data/density_processed.json; nutrition (kcal/100g) from FOOD_INFO fallback.",
            }
        )

    write_food_csv(food_rows, args.output / "food_estimates.csv")
    (args.output / "calorie_estimates.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    (args.output / "fallback_food_info.json").write_text(json.dumps(FOOD_INFO, indent=2), encoding="utf-8")

    # Distribution of volume_source so we can see how often the per-image
    # path kicks in for a smoke test.
    from collections import Counter
    src_counter = Counter(row.get("volume_source") for row in food_rows)
    print(f"Images processed: {len(images)}")
    print(f"Images with calorie estimate: {sum(1 for item in summaries if item.get('status') == 'ok')}")
    print(f"Volume source distribution: {dict(src_counter)}")
    print(f"Output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
