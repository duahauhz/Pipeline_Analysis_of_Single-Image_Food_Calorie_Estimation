"""
Train YOLOv8n as student with distilled labels (GT + YOLOv13n pseudo).
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "yolov13"))

from ultralytics import YOLO

# Paths
YOLOV8N_PRETRAINED = PROJECT_ROOT / "weights" / "yolov8n_pretrained.pt"
DATA_YAML_TRAIN = PROJECT_ROOT / "datasets" / "ECUSTFD" / "ecustfd_distill.yaml"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "distill_v13_to_v8" / "student_train"


def main():
    print("=" * 70)
    print("Training YOLOv8n (student) with distilled labels from YOLOv13n")
    print("=" * 70)
    print(f"Pretrained: {YOLOV8N_PRETRAINED}")
    print(f"Data YAML: {DATA_YAML_TRAIN}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    # Load student
    student = YOLO(str(YOLOV8N_PRETRAINED))

    # Train
    results = student.train(
        data=str(DATA_YAML_TRAIN),
        epochs=50,
        imgsz=640,
        batch=16,
        lr0=1e-3,
        lrf=1e-2,
        optimizer="AdamW",
        device="cuda:0",
        project=str(OUTPUT_DIR.parent),
        name=OUTPUT_DIR.name,
        exist_ok=True,
        verbose=True,
        patience=15,  # early stopping
        save_period=10,
        seed=42,
        workers=0,  # Avoid multiprocessing DLL issues on Windows
        cache=False,
    )

    print(f"\n[Done] Training complete")
    print(f"[Done] Best weights: {OUTPUT_DIR}/weights/best.pt")


if __name__ == "__main__":
    main()
