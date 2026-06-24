"""
Train baseline YOLOv8n với GT labels (không KD) — để so sánh công bằng với student.
"""
import sys
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "yolov13"))

from ultralytics import YOLO

YOLOV8N_PRETRAINED = PROJECT_ROOT / "weights" / "yolov8n_pretrained.pt"
DATA_YAML = PROJECT_ROOT / "datasets" / "ECUSTFD" / "ecustfd.yaml"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "distill_v13_to_v8" / "baseline_v8n_train"


def main():
    print("=" * 70)
    print("Training baseline YOLOv8n (with GT labels only) for fair comparison")
    print("=" * 70)

    # Restore GT labels first
    gt_path = PROJECT_ROOT / "datasets" / "ECUSTFD" / "labels" / "train_gt_backup"
    if gt_path.exists():
        # Remove current (distilled) train labels
        train_path = PROJECT_ROOT / "datasets" / "ECUSTFD" / "labels" / "train"
        if train_path.exists():
            shutil.rmtree(train_path)
        # Copy GT back
        shutil.copytree(gt_path, train_path)
        print("[Restore] GT labels restored")
    else:
        print("[Warning] No GT backup found, training on current labels")

    # Verify
    n = len(list((PROJECT_ROOT / "datasets" / "ECUSTFD" / "labels" / "train").glob("*.txt")))
    print(f"[Verify] Train labels count: {n}")

    # Train
    student = YOLO(str(YOLOV8N_PRETRAINED))
    results = student.train(
        data=str(DATA_YAML),
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
        verbose=False,
        patience=15,
        save_period=10,
        seed=42,
        workers=0,
        cache=False,
    )

    print(f"\n[Done] Baseline training complete")
    print(f"[Done] Best weights: {OUTPUT_DIR}/weights/best.pt")


if __name__ == "__main__":
    main()
