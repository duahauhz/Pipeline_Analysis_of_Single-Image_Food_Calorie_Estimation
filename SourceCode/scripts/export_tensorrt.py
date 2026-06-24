"""
Export YOLOv13n to ONNX FP16 + benchmark with ONNX Runtime (TensorRT EP).

Why: PyTorch .pt has high Python overhead. ONNX + TensorRT EP gives
2-4x speedup on RTX 4050 by using FP16 + kernel autotuning.

Usage: python export_tensorrt.py [model_path] [output_dir]
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "yolov13"))

import numpy as np
import warnings
warnings.filterwarnings("ignore")

from ultralytics import YOLO
import onnxruntime as ort

model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    PROJECT_ROOT / "runs" / "local_food_detect" / "yolov13n_ecustfd_local" / "weights" / "best.pt"
)
out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else model_path.parent
out_dir.mkdir(parents=True, exist_ok=True)

onnx_path = out_dir / f"{model_path.stem}_fp16.onnx"

print(f"Loading model: {model_path}")
model = YOLO(str(model_path))

# Export to ONNX FP16
print(f"\nExporting ONNX FP16 → {onnx_path}")
model.export(
    format="onnx",
    half=True,        # FP16
    imgsz=640,
    simplify=True,    # ONNX simplification
    opset=13,
    device=0,
    project=str(out_dir),
    name="onnx_export",
    exist_ok=True,
)
# Find the actual exported file (ultralytics may append suffix)
candidates = list(out_dir.glob("**/*.onnx"))
if candidates:
    real_onnx = candidates[0]
    if real_onnx != onnx_path:
        real_onnx.rename(onnx_path)
print(f"ONNX file: {onnx_path} ({onnx_path.stat().st_size/1e6:.1f} MB)")

# Verify ONNX loads with ORT
print("\n=== Verifying ONNX with ONNX Runtime ===")
available_providers = ort.get_available_providers()
print(f"Available providers: {available_providers}")

# Prefer TensorRT > CUDA > CPU
if "TensorrtExecutionProvider" in available_providers and "CUDAExecutionProvider" in available_providers:
    providers = [("TensorrtExecutionProvider", {
        "trt_fp16_enable": True,
        "trt_max_workspace_size": str(2 * 1024 * 1024 * 1024),  # 2GB
    }), "CUDAExecutionProvider", "CPUExecutionProvider"]
elif "CUDAExecutionProvider" in available_providers:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
else:
    providers = ["CPUExecutionProvider"]
print(f"Using providers: {providers}")

session = ort.InferenceSession(str(onnx_path), providers=providers)
print(f"Active provider: {session.get_providers()}")
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
output_names = [o.name for o in session.get_outputs()]
print(f"Input: {input_name} {input_shape}")
print(f"Outputs: {len(output_names)} tensors")

# === Benchmark ===
TEST_DIR = PROJECT_ROOT / "datasets" / "ECUSTFD" / "images" / "test"
test_imgs = sorted(TEST_DIR.glob("*.JPG"))[:50]
print(f"\nBenchmarking on {len(test_imgs)} images...")

# Preprocess: load image, resize to 640x640, normalize to FP16 NCHW
import cv2

def preprocess(img_path):
    img = cv2.imread(str(img_path))
    img = cv2.resize(img, (640, 640))
    img = img[:, :, ::-1].astype(np.float16) / 255.0  # BGR→RGB, normalize
    img = np.transpose(img, (2, 0, 1))  # HWC→CHW
    img = np.expand_dims(img, 0)        # add batch
    return img

# Warmup
for img_path in test_imgs[:5]:
    x = preprocess(img_path)
    session.run(output_names, {input_name: x})

# Benchmark
times = []
for img_path in test_imgs:
    x = preprocess(img_path)
    t0 = time.perf_counter()
    session.run(output_names, {input_name: x})
    times.append(time.perf_counter() - t0)

times = np.array(times)
fps = 1.0 / times.mean()
latency_ms = times.mean() * 1000
p50 = np.percentile(times, 50) * 1000
p95 = np.percentile(times, 95) * 1000
p99 = np.percentile(times, 99) * 1000

print(f"\n=== ONNX Runtime Results (providers={session.get_providers()[0]}) ===")
print(f"Mean latency: {latency_ms:.2f} ms/img")
print(f"Mean FPS:     {fps:.1f}")
print(f"P50 latency:  {p50:.2f} ms")
print(f"P95 latency:  {p95:.2f} ms")
print(f"P99 latency:  {p99:.2f} ms")

# Compare to PyTorch baseline (from earlier eval)
pytorch_fps = 29.1
speedup = fps / pytorch_fps
print(f"\nSpeedup vs PyTorch: {speedup:.2f}x ({fps - pytorch_fps:+.1f} FPS)")

# Save report
report = out_dir / "tensorrt_benchmark.md"
report.write_text(
    f"""# ONNX Runtime Benchmark — YOLOv13n_local

- Model: `{model_path.name}`
- Engine: `{onnx_path.name}` ({onnx_path.stat().st_size/1e6:.1f} MB)
- Providers: `{session.get_providers()[0]}` (fallback chain: {providers})
- Input: 640×640 FP16, batch=1
- Test: {len(test_imgs)} images from ECUSTFD test split

## Results

| Metric | Value |
|---|---:|
| Mean latency | {latency_ms:.2f} ms |
| Mean FPS | {fps:.1f} |
| P50 latency | {p50:.2f} ms |
| P95 latency | {p95:.2f} ms |
| P99 latency | {p99:.2f} ms |
| Speedup vs PyTorch (.pt) | {speedup:.2f}x |

## Comparison

| Engine | FPS | Latency (ms) |
|---|---:|---:|
| PyTorch (.pt) | {pytorch_fps:.1f} | {1000/pytorch_fps:.2f} |
| ONNX FP16 ({session.get_providers()[0]}) | {fps:.1f} | {latency_ms:.2f} |
""",
    encoding="utf-8",
)
print(f"\nReport: {report}")

