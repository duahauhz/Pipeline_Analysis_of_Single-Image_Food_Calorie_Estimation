"""
Comprehensive evaluation: so sánh 3 models trên val set + latency trên test set.
Đã sửa: chạy từng model trong subprocess riêng để tránh torch CUDA context issues.
"""
import sys
import time
import json
import warnings
import subprocess
warnings.filterwarnings("ignore")

from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent

TEACHER = PROJECT_ROOT / "runs" / "local_food_detect" / "output" / "weights" / "best.pt"
STUDENT = PROJECT_ROOT / "runs" / "distill_v13_to_v8" / "student_train" / "weights" / "best.pt"
BASELINE = PROJECT_ROOT / "runs" / "distill_v13_to_v8" / "baseline_v8n_train" / "weights" / "best.pt"
TEST_DIR = PROJECT_ROOT / "datasets" / "ECUSTFD" / "images" / "test"
DATA_YAML = PROJECT_ROOT / "datasets" / "ECUSTFD" / "ecustfd.yaml"
OUTPUT = PROJECT_ROOT / "runs" / "distill_v13_to_v8" / "evaluation"
OUTPUT.mkdir(parents=True, exist_ok=True)


EVAL_SCRIPT = PROJECT_ROOT / "scripts" / "_eval_one.py"


def evaluate_subprocess(model_path, name):
    """Run evaluation in subprocess to avoid CUDA context issues."""
    if not model_path.exists():
        return None
    print(f"\n[Eval] {name} (subprocess)...")

    result = subprocess.run(
        [sys.executable, str(EVAL_SCRIPT), str(model_path), name, str(DATA_YAML), str(TEST_DIR), str(OUTPUT)],
        capture_output=True, text=True, timeout=600,
    )

    out_file = OUTPUT / f"eval_{name.replace(' ', '_').replace('(', '').replace(')', '')}.json"
    if out_file.exists():
        with open(out_file) as f:
            metrics = json.load(f)
        print(f"  mAP@0.5:    {metrics['mAP50']:.4f}")
        print(f"  mAP@0.5:0.95: {metrics['mAP50_95']:.4f}")
        print(f"  Latency:    {metrics['latency_ms']:.2f} ms")
        print(f"  FPS:        {metrics['fps']:.1f}")
        return metrics
    else:
        print(f"  [ERR] Subprocess failed:")
        print(result.stderr[-1500:] if result.stderr else "(no stderr)")
        return None


def main():
    print("=" * 60)
    print("Comprehensive Evaluation: KD Results")
    print("=" * 60)

    results = {}
    results["teacher_yolov13n"] = evaluate_subprocess(TEACHER, "yolov13n_teacher")
    results["baseline_yolov8n"] = evaluate_subprocess(BASELINE, "yolov8n_baseline")
    results["student_yolov8n_kd"] = evaluate_subprocess(STUDENT, "yolov8n_student_kd")

    out_file = OUTPUT / "kd_full_comparison.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] {out_file}")

    print("\n" + "=" * 100)
    print(f"{'Model':<30} {'mAP@.5':<10} {'mAP@.5:.95':<12} {'P':<8} {'R':<8} {'Latency':<10} {'FPS':<8} {'Params':<10}")
    print("-" * 100)
    for name, m in results.items():
        if m is None:
            print(f"{name:<30} (not found)")
            continue
        print(f"{name:<30} {m['mAP50']:<10.4f} {m['mAP50_95']:<12.4f} {m['precision']:<8.4f} {m['recall']:<8.4f} {m['latency_ms']:<10.2f} {m['fps']:<8.1f} {m['params_M']:<10.2f}")
    print("=" * 100)


if __name__ == "__main__":
    main()
