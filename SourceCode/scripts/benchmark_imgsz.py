"""
Benchmark YOLOv13n at multiple image sizes to find best speed/accuracy tradeoff.

Trade-off:
- imgsz=640: full accuracy (mAP50-95 ≈ 0.917), 29.1 FPS
- imgsz=416: -1 to -2% mAP, 1.5-2x FPS
- imgsz=320: -3 to -5% mAP, 2-3x FPS

For each imgsz, runs val() to get mAP, then benchmarks latency on test images.
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "yolov13"))

import warnings
warnings.filterwarnings("ignore")

from ultralytics import YOLO

WEIGHTS = PROJECT_ROOT / "runs" / "local_food_detect" / "yolov13n_ecustfd_local" / "weights" / "best.pt"
DATA_YAML = PROJECT_ROOT / "datasets" / "ECUSTFD" / "ecustfd.yaml"
TEST_DIR = PROJECT_ROOT / "datasets" / "ECUSTFD" / "images" / "test"

IMGSZ_LIST = [320, 416, 512, 640]

print(f"Loading model: {WEIGHTS.name}")
model = YOLO(str(WEIGHTS))

import torch
device = "cuda:0" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
print(f"Device: {device}\n")

results = []
for imgsz in IMGSZ_LIST:
    print(f"=== imgsz={imgsz} ===")

    # 1. Quick mAP on val split (this is the expensive step)
    print(f"  Running val() to get mAP...", flush=True)
    val = model.val(
        data=str(DATA_YAML),
        imgsz=imgsz,
        batch=16,
        device=device,
        workers=0,
        verbose=False,
        plots=False,
    )
    map50 = float(val.box.map50)
    map50_95 = float(val.box.map)
    print(f"  mAP50={map50:.4f}, mAP50-95={map50_95:.4f}")

    # 2. Latency benchmark on test images
    test_imgs = sorted(TEST_DIR.glob("*.JPG"))[:50]
    # Warmup
    model.predict(str(test_imgs[0]), imgsz=imgsz, device=device, verbose=False)
    # Benchmark
    t0 = time.perf_counter()
    for img in test_imgs:
        model.predict(str(img), imgsz=imgsz, device=device, verbose=False)
    elapsed = time.perf_counter() - t0
    latency_ms = (elapsed / len(test_imgs)) * 1000
    fps = 1000 / latency_ms
    print(f"  Latency: {latency_ms:.2f} ms, FPS: {fps:.1f}\n")

    results.append({
        "imgsz": imgsz,
        "mAP50": map50,
        "mAP50_95": map50_95,
        "latency_ms": latency_ms,
        "fps": fps,
    })

# Summary
print("=" * 70)
print(f"{'imgsz':>8} | {'mAP50':>8} | {'mAP50-95':>9} | {'latency_ms':>11} | {'FPS':>6}")
print("-" * 70)
for r in results:
    print(f"{r['imgsz']:>8} | {r['mAP50']:>8.4f} | {r['mAP50_95']:>9.4f} | {r['latency_ms']:>11.2f} | {r['fps']:>6.1f}")
print("=" * 70)

# Compute tradeoffs
base = results[-1]  # imgsz=640
print(f"\nSpeedup vs imgsz=640:")
for r in results:
    speedup = r['fps'] / base['fps']
    map_drop = (base['mAP50_95'] - r['mAP50_95']) * 100
    print(f"  imgsz={r['imgsz']}: {speedup:.2f}x faster ({r['fps']:.1f} vs {base['fps']:.1f} FPS), "
          f"mAP50-95 drop {map_drop:+.2f} pp")

# Save markdown report
out = PROJECT_ROOT / "runs" / "distill_v13_to_v8" / "evaluation" / "imgsz_benchmark.md"
lines = [
    "# YOLOv13n_local — imgsz tradeoff benchmark",
    "",
    "- Model: `yolov13n_ecustfd_local/best.pt`",
    "- Device: NVIDIA GeForce RTX 4050 Laptop GPU (6 GB)",
    f"- Test images: 50 từ `datasets/ECUSTFD/images/test`",
    "",
    "| imgsz | mAP50 | mAP50-95 | Latency (ms) | FPS | Δ mAP50-95 vs 640 | Δ FPS vs 640 |",
    "|---:|---:|---:|---:|---:|---:|---:|",
]
for r in results:
    speedup = r['fps'] / base['fps']
    map_drop = (base['mAP50_95'] - r['mAP50_95']) * 100
    lines.append(
        f"| {r['imgsz']} | {r['mAP50']:.4f} | {r['mAP50_95']:.4f} "
        f"| {r['latency_ms']:.2f} | {r['fps']:.1f} | {map_drop:+.2f} pp | {speedup:.2f}x |"
    )
lines.append("")
lines.append("## Kết luận")
# Find sweet spot: max FPS while keeping mAP drop < 1%
sweet = None
for r in results:
    if base['mAP50_95'] - r['mAP50_95'] <= 0.01:  # 1pp drop acceptable
        sweet = r
sweet_str = f"imgsz={sweet['imgsz']} ({sweet['fps']:.1f} FPS, mAP drop {(base['mAP50_95']-sweet['mAP50_95'])*100:.2f} pp)" if sweet else "không có (giữ nguyên 640)"
lines.append(f"- **Sweet spot** (mAP drop ≤ 1 pp): {sweet_str}")
lines.append("- Nếu paper cần so sánh với YOLOv8n (51-55 FPS): chọn imgsz=320 hoặc 416")
lines.append("- Nếu paper nhấn mạnh accuracy: giữ imgsz=640")

out.write_text("\n".join(lines), encoding="utf-8")
print(f"\nSaved report: {out}")
