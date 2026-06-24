"""
Test batch processing YOLOv13n - amortize per-image overhead.
"""
import sys
import time
from pathlib import Path
import statistics

sys.path.insert(0, str(Path(__file__).parent.parent / "yolov13"))
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).parent.parent
V13_WEIGHTS = PROJECT_ROOT / "runs" / "local_food_detect" / "output" / "weights" / "best.pt"
V8_WEIGHTS = PROJECT_ROOT / "runs" / "local_food_detect" / "yolov8n_ecustfd_local" / "weights" / "best.pt"
VAL_IMG_DIR = PROJECT_ROOT / "datasets" / "ECUSTFD" / "images" / "val"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "v13_speedup"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def time_batch(model, img_paths, batch_size, name, n_runs=3):
    print(f"  Warming up {name}...")
    for _ in range(2):
        model([str(p) for p in img_paths[:batch_size]], verbose=False, device="cuda:0", imgsz=640)
    torch.cuda.synchronize()

    print(f"  Measuring {name} (batch={batch_size})...")
    latencies = []
    for run in range(n_runs):
        # Process all in batches
        run_times = []
        for i in range(0, len(img_paths), batch_size):
            batch = img_paths[i:i+batch_size]
            t0 = time.perf_counter()
            model([str(p) for p in batch], verbose=False, device="cuda:0", imgsz=640)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            run_times.append((t1 - t0) * 1000 / len(batch))  # per-image latency
        latencies.append(run_times)

    all_latencies = [t for r in latencies for t in r]
    return {
        "name": name,
        "batch_size": batch_size,
        "mean_ms": statistics.mean(all_latencies),
        "p95_ms": sorted(all_latencies)[int(0.95 * len(all_latencies))],
        "img_per_s": 1000.0 / statistics.mean(all_latencies),
        "n_samples": len(all_latencies),
    }


def main():
    print("=" * 70)
    print("YOLOv13n Batch Processing Test")
    print("=" * 70)

    img_paths = sorted(VAL_IMG_DIR.glob("*.JPG"))[:40]

    print("\n[Setup] Loading YOLOv13n...")
    v13 = YOLO(str(V13_WEIGHTS))

    results = {}
    for bs in [1, 2, 4, 8, 16]:
        r = time_batch(v13, img_paths, bs, f"YOLOv13n batch={bs}", n_runs=3)
        results[f"YOLOv13n batch={bs}"] = r
        print(f"  → {r['mean_ms']:.2f}ms/img ({r['img_per_s']:.1f} img/s)")

    # YOLOv8n reference batch=8
    print("\n[Setup] Loading YOLOv8n...")
    v8 = YOLO(str(V8_WEIGHTS))
    r8 = time_batch(v8, img_paths, 8, "YOLOv8n batch=8", n_runs=3)
    results["YOLOv8n batch=8"] = r8
    print(f"  → {r8['mean_ms']:.2f}ms/img ({r8['img_per_s']:.1f} img/s)")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Method':<25} {'Mean (ms)':<12} {'img/s':<10} {'vs v8 batch=8':<15}")
    for name, r in results.items():
        vs_v8 = r8['mean_ms'] / r['mean_ms']
        status = "FASTER" if r['mean_ms'] < r8['mean_ms'] else "slower"
        print(f"{name:<25} {r['mean_ms']:<12.2f} {r['img_per_s']:<10.1f} {vs_v8:.2f}x {status}")

    import json
    with open(OUTPUT_DIR / "batch_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {OUTPUT_DIR / 'batch_results.json'}")


if __name__ == "__main__":
    main()
