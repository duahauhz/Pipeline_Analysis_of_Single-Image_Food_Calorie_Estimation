"""
Generate pseudo-labels from YOLOv13n teacher on training set.
Saves YOLO-format labels with teacher's confidence scores.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "yolov13"))

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).parent.parent
TEACHER_WEIGHTS = PROJECT_ROOT / "runs" / "local_food_detect" / "output" / "weights" / "best.pt"
TRAIN_IMG_DIR = PROJECT_ROOT / "datasets" / "ECUSTFD" / "images" / "train"
PSEUDO_LABELS_DIR = PROJECT_ROOT / "datasets" / "ECUSTFD" / "labels" / "train_pseudo_v13"
PSEUDO_LABELS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("Pseudo-Label Generation: YOLOv13n (teacher) -> YOLOv8n (student)")
    print("=" * 70)

    print(f"\n[Setup] Loading teacher {TEACHER_WEIGHTS}...")
    teacher = YOLO(str(TEACHER_WEIGHTS))

    print(f"[Setup] Predicting on training images in {TRAIN_IMG_DIR}...")
    print(f"[Setup] Saving pseudo-labels to {PSEUDO_LABELS_DIR}...")

    # Run prediction, save labels in YOLO format with confidence
    results = teacher.predict(
        source=str(TRAIN_IMG_DIR),
        save=False,
        save_txt=False,  # We'll save custom
        save_conf=False,
        conf=0.25,       # Confidence threshold
        iou=0.45,
        imgsz=640,
        device="cuda:0",
        verbose=True,
        stream=True,     # memory efficient
    )

    # Process results and save pseudo-labels
    n_saved = 0
    n_total = 0
    n_with_detections = 0

    for r in results:
        n_total += 1
        img_path = Path(r.path)
        label_path = PSEUDO_LABELS_DIR / (img_path.stem + ".txt")

        if r.boxes is not None and len(r.boxes) > 0:
            n_with_detections += 1
            with open(label_path, "w") as f:
                for box in r.boxes:
                    cls = int(box.cls.item())
                    conf = float(box.conf.item())
                    # YOLO format: cls cx cy w h (normalized)
                    # Plus confidence as 5th value (non-standard but used for distillation)
                    xywhn = box.xywhn[0].cpu().numpy()
                    cx, cy, w, h = xywhn
                    f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {conf:.6f}\n")
            n_saved += 1
        # If no detections, skip (don't save empty file)

    print(f"\n[Result] Processed {n_total} images")
    print(f"[Result] Images with detections: {n_with_detections} ({n_with_detections/n_total*100:.1f}%)")
    print(f"[Result] Pseudo-labels saved: {n_saved}")
    print(f"[Result] Location: {PSEUDO_LABELS_DIR}")


if __name__ == "__main__":
    main()
