"""
Evaluate YOLOv8n student (after KD) on test set.
Compare with baseline YOLOv8n và YOLOv13n.
"""
import sys
import time
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "yolov13"))

from ultralytics import YOLO

# Paths
STUDENT_WEIGHTS = PROJECT_ROOT / "runs" / "distill_v13_to_v8" / "student_train" / "weights" / "best.pt"
BASELINE_V8N = PROJECT_ROOT / "yolov8n_baseline.pt"  # We'll generate if not exists
TEACHER_V13N = PROJECT_ROOT / "runs" / "local_food_detect" / "output" / "weights" / "best.pt"
TEST_IMG_DIR = PROJECT_ROOT / "datasets" / "ECUSTFD" / "images" / "test"
TEST_LABEL_DIR = PROJECT_ROOT / "datasets" / "ECUSTFD" / "labels" / "test"
DATA_YAML = PROJECT_ROOT / "datasets" / "ECUSTFD" / "ecustfd.yaml"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "distill_v13_to_v8" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model(model_path, name, imgsz=640):
    """Evaluate a model on test set."""
    print(f"\n[Eval] {name}")
    print(f"  Weights: {model_path}")
    if not model_path.exists():
        print(f"  SKIP: file not found")
        return None

    model = YOLO(str(model_path))

    # Run validation
    val_results = model.val(
        data=str(DATA_YAML),
        imgsz=imgsz,
        batch=16,
        device="cuda:0",
        verbose=False,
        plots=False,
    )

    # Run inference for latency
    test_imgs = sorted(TEST_IMG_DIR.glob("*.JPG"))
    if not test_imgs:
        test_imgs = sorted(TEST_IMG_DIR.glob("*.jpg"))
    test_imgs = test_imgs[:50]  # Subset for latency

    # Warmup
    model.predict(str(test_imgs[0]), verbose=False)
    # Measure
    start = time.perf_counter()
    for img in test_imgs:
        model.predict(str(img), verbose=False)
    elapsed = time.perf_counter() - start
    latency_ms = (elapsed / len(test_imgs)) * 1000

    metrics = {
        "model": name,
        "weights": str(model_path),
        "mAP50": float(val_results.box.map50),
        "mAP50-95": float(val_results.box.map),
        "precision": float(val_results.box.mp),
        "recall": float(val_results.box.mr),
        "latency_ms_per_image": latency_ms,
        "fps": 1000 / latency_ms,
        "n_test_images": len(test_imgs),
    }

    print(f"  mAP@0.5:    {metrics['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95: {metrics['mAP50-95']:.4f}")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  Latency:    {latency_ms:.2f} ms")
    print(f"  FPS:        {metrics['fps']:.1f}")

    return metrics


def main():
    print("=" * 70)
    print("Knowledge Distillation — Final Evaluation")
    print("=" * 70)

    results = {}

    # Evaluate teacher (YOLOv13n)
    results["teacher_yolov13n"] = evaluate_model(TEACHER_V13N, "YOLOv13n (teacher)")

    # Evaluate student (YOLOv8n + KD)
    results["student_yolov8n_kd"] = evaluate_model(STUDENT_WEIGHTS, "YOLOv8n (student + KD)")

    # Compare with baseline YOLOv8n (if exists)
    # We'll train baseline in a separate step
    # For now, just save the comparison

    # Save results
    output_file = OUTPUT_DIR / "kd_evaluation.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Result] Saved to: {output_file}")

    # Print comparison table
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"{'Model':<30} {'mAP@0.5':<12} {'Latency (ms)':<15} {'FPS':<10}")
    print("-" * 70)
    for name, m in results.items():
        if m is None:
            continue
        print(f"{m['model']:<30} {m['mAP50']:<12.4f} {m['latency_ms_per_image']:<15.2f} {m['fps']:<10.1f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
