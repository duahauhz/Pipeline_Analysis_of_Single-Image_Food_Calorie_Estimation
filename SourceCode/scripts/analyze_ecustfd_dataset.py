import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CLASSES = [
    "apple",
    "banana",
    "bread",
    "bun",
    "doughnut",
    "egg",
    "fired_dough_twist",
    "grape",
    "lemon",
    "litchi",
    "mango",
    "mooncake",
    "orange",
    "peach",
    "pear",
    "plum",
    "qiwi",
    "sachima",
    "tomato",
    "coin",
]

SPLITS = ("train", "val", "test")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
COLORS = [
    (230, 57, 70),
    (29, 53, 87),
    (42, 157, 143),
    (244, 162, 97),
    (131, 56, 236),
    (255, 183, 3),
    (33, 158, 188),
    (106, 153, 78),
]


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def read_yolo_label(label_path: Path) -> tuple[list[dict], list[str]]:
    objects = []
    errors = []

    for line_no, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{label_path}:{line_no}: expected 5 fields, got {len(parts)}")
            continue

        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = (float(value) for value in parts[1:])
        except ValueError:
            errors.append(f"{label_path}:{line_no}: non-numeric YOLO value")
            continue

        if class_id < 0 or class_id >= len(CLASSES):
            errors.append(f"{label_path}:{line_no}: class_id {class_id} out of range")

        coords = (x_center, y_center, width, height)
        if any(value < 0.0 or value > 1.0 for value in coords):
            errors.append(f"{label_path}:{line_no}: normalized coordinate out of [0, 1]")

        if width <= 0.0 or height <= 0.0:
            errors.append(f"{label_path}:{line_no}: width/height must be positive")

        objects.append(
            {
                "class_id": class_id,
                "class_name": CLASSES[class_id] if 0 <= class_id < len(CLASSES) else "unknown",
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height,
            }
        )

    return objects, errors


def yolo_to_xyxy(obj: dict, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    box_w = obj["width"] * img_w
    box_h = obj["height"] * img_h
    x_center = obj["x_center"] * img_w
    y_center = obj["y_center"] * img_h
    x1 = max(0, int(round(x_center - box_w / 2)))
    y1 = max(0, int(round(y_center - box_h / 2)))
    x2 = min(img_w - 1, int(round(x_center + box_w / 2)))
    y2 = min(img_h - 1, int(round(y_center + box_h / 2)))
    return x1, y1, x2, y2


def draw_sample(image_path: Path, objects: list[dict], output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    img_w, img_h = image.size

    for obj in objects:
        color = COLORS[obj["class_id"] % len(COLORS)]
        x1, y1, x2, y2 = yolo_to_xyxy(obj, img_w, img_h)
        label = obj["class_name"]

        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        text_box = draw.textbbox((x1, y1), label, font=font)
        text_w = text_box[2] - text_box[0]
        text_h = text_box[3] - text_box[1]
        label_y1 = max(0, y1 - text_h - 4)
        draw.rectangle((x1, label_y1, x1 + text_w + 6, label_y1 + text_h + 4), fill=color)
        draw.text((x1 + 3, label_y1 + 2), label, fill=(255, 255, 255), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)


def analyze_dataset(dataset_dir: Path, output_dir: Path, sample_count: int) -> dict:
    class_counts_by_split = {split: Counter() for split in SPLITS}
    split_rows = []
    validation_errors = []
    visualized = []

    for split in SPLITS:
        images_dir = dataset_dir / "images" / split
        labels_dir = dataset_dir / "labels" / split

        image_files = sorted(
            path for path in images_dir.iterdir() if path.is_file() and path.suffix in IMAGE_EXTS
        ) if images_dir.exists() else []
        label_files = sorted(labels_dir.glob("*.txt")) if labels_dir.exists() else []

        image_stems = {path.stem for path in image_files}
        label_stems = {path.stem for path in label_files}
        missing_labels = sorted(image_stems - label_stems)
        missing_images = sorted(label_stems - image_stems)

        for stem in missing_labels:
            validation_errors.append(f"{split}: missing label for image {stem}")
        for stem in missing_images:
            validation_errors.append(f"{split}: missing image for label {stem}")

        object_count = 0
        sample_candidates = []
        for label_path in label_files:
            objects, errors = read_yolo_label(label_path)
            validation_errors.extend(errors)
            object_count += len(objects)
            for obj in objects:
                class_counts_by_split[split][obj["class_name"]] += 1

            image_path = find_image(images_dir, label_path.stem)
            if image_path and objects:
                sample_candidates.append((image_path, objects))

        for image_path, objects in sample_candidates[:sample_count]:
            output_path = output_dir / "visualizations" / f"{split}_{image_path.stem}.jpg"
            draw_sample(image_path, objects, output_path)
            visualized.append(str(output_path))

        split_rows.append(
            {
                "split": split,
                "images": len(image_files),
                "labels": len(label_files),
                "objects": object_count,
                "missing_labels": len(missing_labels),
                "missing_images": len(missing_images),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_class_distribution(output_dir / "class_distribution.csv", class_counts_by_split)
    write_split_summary(output_dir / "split_summary.csv", split_rows)

    report = {
        "dataset_dir": str(dataset_dir),
        "class_count": len(CLASSES),
        "classes": CLASSES,
        "splits": split_rows,
        "total_objects_by_class": total_objects_by_class(class_counts_by_split),
        "validation_error_count": len(validation_errors),
        "validation_errors": validation_errors,
        "visualizations": visualized,
    }

    (output_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return report


def total_objects_by_class(class_counts_by_split: dict[str, Counter]) -> dict[str, int]:
    totals = defaultdict(int)
    for counts in class_counts_by_split.values():
        for class_name, count in counts.items():
            totals[class_name] += count
    return {class_name: totals[class_name] for class_name in CLASSES}


def write_class_distribution(path: Path, class_counts_by_split: dict[str, Counter]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["class_id", "class_name", *SPLITS, "total"])
        for class_id, class_name in enumerate(CLASSES):
            row = [class_id, class_name]
            total = 0
            for split in SPLITS:
                count = class_counts_by_split[split][class_name]
                row.append(count)
                total += count
            row.append(total)
            writer.writerow(row)


def write_split_summary(path: Path, split_rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        fieldnames = ["split", "images", "labels", "objects", "missing_labels", "missing_images"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(split_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze ECUSTFD YOLO dataset and draw bbox samples.")
    parser.add_argument("--dataset", type=Path, default=Path("datasets/ECUSTFD"))
    parser.add_argument("--output", type=Path, default=Path("datasets/ECUSTFD/reports"))
    parser.add_argument("--sample-count", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze_dataset(args.dataset, args.output, args.sample_count)

    print(f"Dataset: {report['dataset_dir']}")
    for split in report["splits"]:
        print(
            f"{split['split']}: {split['images']} images, {split['labels']} labels, "
            f"{split['objects']} objects, {split['missing_labels']} missing labels, "
            f"{split['missing_images']} missing images"
        )
    print(f"Validation errors: {report['validation_error_count']}")
    print(f"Reports written to: {args.output}")


if __name__ == "__main__":
    main()
