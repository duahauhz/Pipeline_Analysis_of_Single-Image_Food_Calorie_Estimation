import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
INVALID_CALIBRATION_STEMS = {"mix002T(2)", "mix005S(4)"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sample YOLOv13 inference on ECUSTFD test images.")
    parser.add_argument("--source", type=Path, default=Path("datasets/ECUSTFD/images/test"))
    parser.add_argument("--weights", type=Path, default=Path("weights/yolov13n_ecustfd_best.pt"))
    parser.add_argument("--repo", type=Path, default=Path("yolov13"))
    parser.add_argument("--output", type=Path, default=Path("runs/local_food_detect/sample_inference"))
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def class_key(path: Path) -> str:
    match = re.match(r"^[A-Za-z_]+", path.stem)
    return match.group(0) if match else path.stem


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


def detection_rows(result, image_path: Path) -> list[dict]:
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
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width_px": x2 - x1,
                "height_px": y2 - y1,
            }
        )
    return rows


def write_csv(rows: list[dict], output_path: Path) -> None:
    fieldnames = [
        "image",
        "detection_id",
        "class_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "width_px",
        "height_px",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    ensure_yolov13_import(args.repo)
    if not args.weights.exists():
        raise FileNotFoundError(f"Missing weights: {args.weights}")
    if not args.source.exists():
        raise FileNotFoundError(f"Missing source: {args.source}")

    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(".yolo-config").resolve()))
    os.environ.setdefault("YOLO_OFFLINE", "true")

    import cv2
    from ultralytics import YOLO

    args.output.mkdir(parents=True, exist_ok=True)
    images = collect_images(args.source, args.max_images)
    model = YOLO(str(args.weights))

    all_rows = []
    summary = []
    for image_path in images:
        result = model.predict(source=str(image_path), conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        rows = detection_rows(result, image_path)
        all_rows.extend(rows)

        annotated = result.plot()
        output_image = args.output / f"{image_path.stem}_pred.jpg"
        cv2.imwrite(str(output_image), annotated)

        class_counts = {}
        for row in rows:
            class_counts[row["class_name"]] = class_counts.get(row["class_name"], 0) + 1
        summary.append(
            {
                "image": str(image_path),
                "output_image": str(output_image),
                "detections": len(rows),
                "class_counts": class_counts,
                "has_coin": class_counts.get("coin", 0) > 0,
            }
        )

    write_csv(all_rows, args.output / "detections.csv")
    (args.output / "detections.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Images processed: {len(images)}")
    print(f"Detections: {len(all_rows)}")
    print(f"Output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
