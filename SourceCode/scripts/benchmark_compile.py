"""
Benchmark YOLOv13n with torch.compile to measure speedup.

torch.compile fuses kernels and removes Python overhead. On RTX 4050
+ PyTorch 2.11 expect 10-30% speedup for inference.

Usage: python benchmark_compile.py
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "yolov13"))

import torch
import warnings
warnings.filterwarnings("ignore")

from ultralytics import YOLO

WEIGHTS = PROJECT_ROOT / "runs" / "local_food_detect" / "yolov13n_ecustfd_local" / "weights" / "best.pt"
TEST_DIR = PROJECT_ROOT / "datasets" / "ECUSTFD" / "images" / "test"

print(f"Loading model: {WEIGHTS.name}")
model = YOLO(str(WEIGHTS))

# Pick test images
test_imgs = sorted(TEST_DIR.glob("*.JPG"))[:50]
print(f"Test images: {len(test_imgs)}")

# Warmup
model.predict(str(test_imgs[0]), device="cuda:0", verbose=False, imgsz=640)
torch.cuda.synchronize()

# === Eager mode (no compile) ===
print("\n=== Eager mode ===")
t0 = time.perf_counter()
for img in test_imgs:
    model.predict(str(img), device="cuda:0", verbose=False, imgsz=640)
torch.cuda.synchronize()
elapsed_eager = time.perf_counter() - t0
fps_eager = len(test_imgs) / elapsed_eager
print(f"Eager: {elapsed_eager:.2f}s, {fps_eager:.1f} FPS, {1000/fps_eager:.2f} ms/img")

# === torch.compile ===
print("\n=== torch.compile (mode=reduce-overhead) ===")
# Ultralytics wraps model in a Predictor; compile inner model
try:
    compiled_model = torch.compile(model.model, mode="reduce-overhead", backend="inductor")
    model.model = compiled_model
    print("Compiled successfully")
except Exception as e:
    print(f"Compile failed: {e}")
    sys.exit(1)

# Warmup after compile (first 2-3 iters are slow due to compilation)
print("Warming up compiled model (3 iters)...")
for img in test_imgs[:3]:
    model.predict(str(img), device="cuda:0", verbose=False, imgsz=640)
torch.cuda.synchronize()

# Benchmark
t0 = time.perf_counter()
for img in test_imgs:
    model.predict(str(img), device="cuda:0", verbose=False, imgsz=640)
torch.cuda.synchronize()
elapsed_compile = time.perf_counter() - t0
fps_compile = len(test_imgs) / elapsed_compile
print(f"Compiled: {elapsed_compile:.2f}s, {fps_compile:.1f} FPS, {1000/fps_compile:.2f} ms/img")

speedup = fps_compile / fps_eager
print(f"\nSpeedup: {speedup:.2f}x ({fps_compile - fps_eager:+.1f} FPS)")
