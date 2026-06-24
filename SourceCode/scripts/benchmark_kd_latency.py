"""
Benchmark latency for KD models: teacher (YOLOv13n), student (YOLOv8n+KD), baseline (YOLOv8n).
Runs in subprocess to avoid CUDA context issues.
"""
import sys
import time
import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

TEACHER = PROJECT_ROOT / "runs" / "local_food_detect" / "output" / "weights" / "best.pt"
STUDENT = PROJECT_ROOT / "runs" / "distill_v13_to_v8" / "student_train" / "weights" / "best.pt"
BASELINE = PROJECT_ROOT / "runs" / "distill_v13_to_v8" / "baseline_v8n_train" / "weights" / "best.pt"
TEST_DIR = PROJECT_ROOT / "datasets" / "ECUSTFD" / "images" / "test"

SCRIPT_CONTENT = r'''
import sys
from pathlib import Path

# Add yolov13 to path for ultralytics
yolov13_root = Path(__file__).parent.parent
sys.path.insert(0, str(yolov13_root / "yolov13"))

import time
import json
from ultralytics import YOLO

model_path = Path(sys.argv[1])
model_name = sys.argv[2]
output_path = Path(sys.argv[3])

print(f"Loading {model_name} from {model_path}...")
model = YOLO(str(model_path))

# Warmup
print("Warming up...")
test_dir = Path(r"{PROJECT_ROOT}") / "datasets" / "ECUSTFD" / "images" / "test"
for _ in range(3):
    sample = list(test_dir.glob("*.jpg"))[0]
    _ = model(str(sample), imgsz=640, verbose=False)

# Benchmark: batch=1
print("Benchmarking batch=1...")
times_batch1 = []
test_images = list(test_dir.glob("*.jpg"))[:50]
for img in test_images:
    start = time.time()
    _ = model(str(img), imgsz=640, verbose=False)
    times_batch1.append((time.time() - start) * 1000)

# Benchmark: batch=8
print("Benchmarking batch=8...")
batch_paths = list(test_dir.glob("*.jpg"))[:16]
times_batch8 = []
for _ in range(3):
    start = time.time()
    _ = model(batch_paths, imgsz=640, verbose=False)
    times_batch8.append((time.time() - start) * 1000 / len(batch_paths))

result = {
    "model": model_name,
    "weights": str(model_path),
    "batch_1": {
        "mean_ms": float(sum(times_batch1) / len(times_batch1)),
        "min_ms": float(min(times_batch1)),
        "max_ms": float(max(times_batch1)),
        "fps": float(1000 / (sum(times_batch1) / len(times_batch1))),
        "n_images": len(times_batch1)
    },
    "batch_8": {
        "mean_ms": float(sum(times_batch8) / len(times_batch8)),
        "fps": float(1000 / (sum(times_batch8) / len(times_batch8)))
    }
}

with open(output_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved to {output_path}")
'''

def run_benchmark(model_path, name, output_path, script_dir):
    """Run benchmark in subprocess."""
    if not model_path.exists():
        print(f"[SKIP] {name}: {model_path} not found")
        return None

    print(f"\n[Benchmark] {name}...")
    script_path = script_dir / "benchmark_temp.py"
    # Replace the Path(__file__) reference with actual PROJECT_ROOT
    script_content = SCRIPT_CONTENT.replace(
        'Path(__file__).parent.parent',
        f'Path(r"{PROJECT_ROOT}")'
    ).replace(
        '"datasets/ECUSTFD/images/test"',
        f'Path(r"{PROJECT_ROOT}") / "datasets" / "ECUSTFD" / "images" / "test"'
    )
    with open(script_path, "w") as f:
        f.write(script_content)

    try:
        result = subprocess.run(
            [sys.executable, str(script_path), str(model_path), name, str(output_path)],
            capture_output=True, text=True, timeout=300,
            cwd=str(PROJECT_ROOT)
        )
        if result.returncode == 0:
            print(f"  [OK] {result.stdout[-500:]}")
            with open(output_path) as f:
                data = json.load(f)
            return data
        else:
            print(f"  [ERR] {result.stderr[-1000:]}")
            return None
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {name}")
        return None
    finally:
        if script_path.exists():
            script_path.unlink()


def main():
    OUTPUT_DIR = PROJECT_ROOT / "runs" / "kd_benchmark"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("KD Models Latency Benchmark")
    print("=" * 60)

    results = {}
    results["teacher_yolov13n"] = run_benchmark(
        TEACHER, "YOLOv13n-teacher",
        OUTPUT_DIR / "yolov13n.json", OUTPUT_DIR
    )
    results["baseline_yolov8n"] = run_benchmark(
        BASELINE, "YOLOv8n-baseline",
        OUTPUT_DIR / "yolov8n_baseline.json", OUTPUT_DIR
    )
    results["student_yolov8n_kd"] = run_benchmark(
        STUDENT, "YOLOv8n+KD-student",
        OUTPUT_DIR / "yolov8n_kd.json", OUTPUT_DIR
    )

    # Summary
    print("\n" + "=" * 80)
    print(f"{'Model':<25} {'Batch=1 (ms)':<15} {'FPS@1':<10} {'Batch=8 (ms)':<15} {'FPS@8':<10}")
    print("-" * 80)
    for name, r in results.items():
        if r is None:
            print(f"{name:<25} (failed)")
            continue
        b1 = r["batch_1"]
        b8 = r["batch_8"]
        print(f"{name:<25} {b1['mean_ms']:<15.2f} {b1['fps']:<10.1f} {b8['mean_ms']:<15.2f} {b8['fps']:<10.1f}")
    print("=" * 80)

    # Save comparison
    out_file = OUTPUT_DIR / "latency_comparison.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] {out_file}")


if __name__ == "__main__":
    main()
