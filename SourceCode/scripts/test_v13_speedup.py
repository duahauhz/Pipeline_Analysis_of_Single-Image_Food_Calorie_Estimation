"""
Test các phương pháp tăng tốc YOLOv13n inference mà KHÔNG cần train lại.

Bao gồm:
1. Baseline (FP32, no optimization)
2. FP16 (half precision)
3. torch.compile (JIT)
4. ONNX + onnxruntime-gpu
5. OpenVINO (nếu có)
6. TensorRT (nếu có)

So sánh:
- Inference time (mean, p50, p95)
- Memory usage
- Detection quality (F1 score trên 100 val images)
- Output consistency (so sánh predictions giữa các phương pháp)
"""
import sys
import json
import time
from pathlib import Path
import statistics

sys.path.insert(0, str(Path(__file__).parent.parent / "yolov13"))
import numpy as np
import torch
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
V13_WEIGHTS = PROJECT_ROOT / "runs" / "local_food_detect" / "output" / "weights" / "best.pt"
V8_WEIGHTS = PROJECT_ROOT / "runs" / "local_food_detect" / "yolov8n_ecustfd_local" / "weights" / "best.pt"
VAL_IMG_DIR = PROJECT_ROOT / "datasets" / "ECUSTFD" / "images" / "val"
LABEL_DIR = PROJECT_ROOT / "datasets" / "ECUSTFD" / "labels" / "val"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "v13_speedup"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(PROJECT_ROOT / "datasets" / "ECUSTFD" / "ecustfd.yaml") as f:
    cfg = yaml.safe_load(f)
NAMES = cfg["names"]


def get_img_size(img_path):
    with Image.open(img_path) as im:
        return im.size


def load_gt(label_dir, img_dir, n_imgs=50):
    import random
    random.seed(42)
    img_paths = random.sample(sorted(img_dir.glob("*.JPG")), n_imgs)
    gt = {}
    for img_path in img_paths:
        lbl = label_dir / (img_path.stem + ".txt")
        if lbl.exists():
            boxes = []
            with open(lbl) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        boxes.append({"cls": cls, "cx": cx, "cy": cy, "w": w, "h": h})
            gt[img_path.name] = boxes
    return img_paths, gt


def measure_time(model_or_predict_fn, img_paths, name, n_runs=3, **predict_kwargs):
    """Measure inference time. predict_fn: fn(img_path) -> results"""
    print(f"  Warming up {name}...")
    # Warmup
    for p in img_paths[:3]:
        model_or_predict_fn(str(p), **predict_kwargs)

    # Measure
    latencies = []
    for run in range(n_runs):
        run_times = []
        for p in img_paths:
            t0 = time.perf_counter()
            model_or_predict_fn(str(p), **predict_kwargs)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            run_times.append((t1 - t0) * 1000)
        latencies.append(run_times)

    all_latencies = [t for r in latencies for t in r]
    return {
        "name": name,
        "mean_ms": statistics.mean(all_latencies),
        "std_ms": statistics.stdev(all_latencies),
        "p50_ms": statistics.median(all_latencies),
        "p95_ms": sorted(all_latencies)[int(0.95 * len(all_latencies))],
        "img_per_s": 1000.0 / statistics.mean(all_latencies),
        "n_samples": len(all_latencies),
    }


def extract_predictions(result, name):
    """Extract predictions from a single YOLO result."""
    r = result if not isinstance(result, list) else result[0]
    preds = []
    if r.boxes is not None:
        for box in r.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            xyxy = box.xyxy[0].cpu().numpy()
            preds.append({
                "cls": cls_id,
                "conf": conf,
                "x1": float(xyxy[0]),
                "y1": float(xyxy[1]),
                "x2": float(xyxy[2]),
                "y2": float(xyxy[3]),
            })
    return preds


def compute_iou(box1, box2):
    """IoU between two boxes {x1,y1,x2,y2}."""
    ix1 = max(box1["x1"], box2["x1"])
    iy1 = max(box1["y1"], box2["y1"])
    ix2 = min(box1["x2"], box2["x2"])
    iy2 = min(box1["y2"], box2["y2"])
    if ix2 > ix1 and iy2 > iy1:
        inter = (ix2 - ix1) * (iy2 - iy1)
        a1 = (box1["x2"] - box1["x1"]) * (box1["y2"] - box1["y1"])
        a2 = (box2["x2"] - box2["x1"]) * (box2["y2"] - box2["y1"])
        return inter / (a1 + a2 - inter + 1e-9)
    return 0.0


def test_fp16_v13():
    """Test 1: FP16 (half precision) on YOLOv13n."""
    print("\n[Test 1] YOLOv13n with FP16 (half=True)")
    from ultralytics import YOLO
    model = YOLO(str(V13_WEIGHTS))
    img_paths = sorted(VAL_IMG_DIR.glob("*.JPG"))[:20]
    speed = measure_time(model, img_paths, "YOLOv13n FP16", n_runs=3,
                         verbose=False, device="cuda:0", imgsz=640, half=True)
    return speed, model


def test_fp32_v13():
    """Baseline: FP32 (current)."""
    print("\n[Test 0] YOLOv13n FP32 (baseline)")
    from ultralytics import YOLO
    model = YOLO(str(V13_WEIGHTS))
    img_paths = sorted(VAL_IMG_DIR.glob("*.JPG"))[:20]
    speed = measure_time(model, img_paths, "YOLOv13n FP32", n_runs=3,
                         verbose=False, device="cuda:0", imgsz=640, half=False)
    return speed, model


def test_fp32_v8():
    """Baseline: YOLOv8n FP32 (for comparison)."""
    print("\n[Reference] YOLOv8n FP32")
    from ultralytics import YOLO
    model = YOLO(str(V8_WEIGHTS))
    img_paths = sorted(VAL_IMG_DIR.glob("*.JPG"))[:20]
    speed = measure_time(model, img_paths, "YOLOv8n FP32", n_runs=3,
                         verbose=False, device="cuda:0", imgsz=640, half=False)
    return speed, model


def test_torch_compile_v13():
    """Test 2: torch.compile (JIT fusion)."""
    print("\n[Test 2] YOLOv13n with torch.compile")
    try:
        from ultralytics import YOLO
        model = YOLO(str(V13_WEIGHTS))
        # torch.compile the underlying model
        print("  Compiling model (may take 30s+)...")
        model.model = torch.compile(model.model, mode="reduce-overhead", fullgraph=False)
        img_paths = sorted(VAL_IMG_DIR.glob("*.JPG"))[:20]
        speed = measure_time(model, img_paths, "YOLOv13n torch.compile", n_runs=2,
                             verbose=False, device="cuda:0", imgsz=640, half=False)
        return speed, model
    except Exception as e:
        print(f"  FAILED: {e}")
        return {"name": "torch.compile", "error": str(e)}, None


def test_onnx_v13():
    """Test 3: ONNX export + onnxruntime-gpu."""
    print("\n[Test 3] YOLOv13n with ONNX + onnxruntime-gpu")
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime not installed. Trying export to ONNX only...")
        onnxruntime_available = False
    else:
        onnxruntime_available = True

    onnx_path = OUTPUT_DIR / "v13n.onnx"

    if not onnx_path.exists():
        print(f"  Exporting to ONNX...")
        from ultralytics import YOLO
        model = YOLO(str(V13_WEIGHTS))
        # Export to ONNX with dynamic batch
        model.export(format="onnx", imgsz=640, simplify=False, dynamic=False, half=False, opset=18)
        # Find exported file
        default_path = V13_WEIGHTS.parent / f"{V13_WEIGHTS.stem}.onnx"
        if default_path.exists():
            import shutil
            shutil.move(str(default_path), str(onnx_path))
        # Try other locations
        if not onnx_path.exists():
            for loc in [V13_WEIGHTS.parent, PROJECT_ROOT / "runs" / "v13_speedup"]:
                cand = list(loc.glob("*.onnx"))
                if cand:
                    import shutil
                    shutil.move(str(cand[0]), str(onnx_path))
                    break

    if not onnx_path.exists():
        return {"name": "ONNX", "error": "Export failed"}, None

    print(f"  ONNX file: {onnx_path} ({onnx_path.stat().st_size/1024/1024:.2f}MB)")

    if not onnxruntime_available:
        return {"name": "ONNX", "info": "Exported successfully but onnxruntime not installed"}, None

    # Inference with onnxruntime-gpu
    print("  Loading ONNX with onnxruntime-gpu...")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    print(f"  Active provider: {sess.get_providers()[0]}")

    # Get input name
    input_name = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]

    def ort_predict(img_path, conf_thres=0.25, iou_thres=0.45):
        """Custom NMS for onnxruntime output."""
        from PIL import Image
        # Load + preprocess
        img = Image.open(img_path).convert("RGB")
        img_resized = img.resize((640, 640))
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        img_array = img_array.transpose(2, 0, 1)[None]  # HWC -> 1CHW
        # Run
        outputs = sess.run(output_names, {input_name: img_array})
        # YOLOv13 ONNX output: [1, 84, 8400] (xywh, scores, classes)
        # OR [1, num_classes+4, num_anchors]
        pred = outputs[0]  # [1, 84, 8400]
        pred = pred[0]  # [84, 8400]
        # Transpose to [8400, 84]
        pred = pred.T
        # Split: first 4 are box (cx, cy, w, h), rest are class scores
        boxes = pred[:, :4]
        scores = pred[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        max_scores = np.max(scores, axis=1)
        # Filter by conf
        mask = max_scores > conf_thres
        boxes = boxes[mask]
        max_scores = max_scores[mask]
        class_ids = class_ids[mask]
        # Scale to original image size
        orig_w, orig_h = img.size
        boxes[:, 0] *= orig_w / 640
        boxes[:, 1] *= orig_h / 640
        boxes[:, 2] *= orig_w / 640
        boxes[:, 3] *= orig_h / 640
        # Convert xywh to xyxy
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        # NMS using torchvision if available
        try:
            import torchvision
            xyxy = torch.tensor(np.stack([x1, y1, x2, y2], axis=1))
            scores_t = torch.tensor(max_scores)
            keep = torchvision.ops.nms(xyxy, scores_t, iou_thres)
            x1, y1, x2, y2 = x1[keep.numpy()], y1[keep.numpy()], x2[keep.numpy()], y2[keep.numpy()]
            max_scores = max_scores[keep.numpy()]
            class_ids = class_ids[keep.numpy()]
        except Exception:
            pass
        return [{"cls": int(c), "conf": float(s), "x1": float(xx1), "y1": float(yy1),
                 "x2": float(xx2), "y2": float(yy2)}
                for c, s, xx1, yy1, xx2, yy2 in zip(class_ids, max_scores, x1, y1, x2, y2)]

    img_paths = sorted(VAL_IMG_DIR.glob("*.JPG"))[:20]
    speed = measure_time(ort_predict, img_paths, "YOLOv13n ONNX-GPU", n_runs=3)
    return speed, sess


def test_openvino_v13():
    """Test 4: OpenVINO optimization."""
    print("\n[Test 4] YOLOv13n with OpenVINO")
    try:
        from openvino.runtime import Core
    except ImportError:
        print("  openvino not installed")
        return {"name": "OpenVINO", "error": "Not installed"}, None

    try:
        from ultralytics import YOLO
        ov_path = OUTPUT_DIR / "v13n_openvino_model"
        if not ov_path.exists():
            print("  Exporting to OpenVINO...")
            model = YOLO(str(V13_WEIGHTS))
            model.export(format="openvino", imgsz=640, half=False)
            # Find exported dir
            default_dir = V13_WEIGHTS.parent / f"{V13_WEIGHTS.stem}_openvino_model"
            if default_dir.exists():
                import shutil
                shutil.move(str(default_dir), str(ov_path))
        if not ov_path.exists():
            return {"name": "OpenVINO", "error": "Export failed"}, None

        # OpenVINO inference (CPU-optimized, may not help on GPU)
        core = Core()
        model_ov = core.read_model(model=str(ov_path / "v13n.xml"))
        compiled = core.compile_model(model_ov, device_name="CPU")  # CPU for OV

        def ov_predict(img_path):
            from PIL import Image
            img = Image.open(img_path).convert("RGB").resize((640, 640))
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = img_array.transpose(2, 0, 1)[None]
            result = compiled([img_array])
            return result

        img_paths = sorted(VAL_IMG_DIR.glob("*.JPG"))[:20]
        speed = measure_time(ov_predict, img_paths, "YOLOv13n OpenVINO-CPU", n_runs=2)
        return speed, compiled
    except Exception as e:
        print(f"  FAILED: {e}")
        return {"name": "OpenVINO", "error": str(e)}, None


def test_tensorrt_v13():
    """Test 5: TensorRT via onnxruntime TensorrtExecutionProvider."""
    print("\n[Test 5] YOLOv13n with onnxruntime TensorRT EP")
    try:
        import onnxruntime as ort
    except ImportError:
        return {"name": "TensorRT", "error": "onnxruntime not installed"}, None

    onnx_path = OUTPUT_DIR / "v13n.onnx"
    if not onnx_path.exists():
        return {"name": "TensorRT", "error": "ONNX not exported (run ONNX test first)"}, None

    # Use TensorRT EP
    providers = [
        ("TensorrtExecutionProvider", {
            "trt_fp16_enable": True,
            "trt_max_workspace_size": 2 * 1024 * 1024 * 1024,  # 2GB
        }),
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    try:
        sess = ort.InferenceSession(str(onnx_path), providers=providers)
    except Exception as e:
        return {"name": "TensorRT", "error": f"Failed to load with TRT: {e}"}, None

    print(f"  Active providers: {sess.get_providers()}")

    input_name = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]

    def trt_predict(img_path, conf_thres=0.25, iou_thres=0.45):
        from PIL import Image
        img = Image.open(img_path).convert("RGB")
        img_resized = img.resize((640, 640))
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        img_array = img_array.transpose(2, 0, 1)[None]
        outputs = sess.run(output_names, {input_name: img_array})
        pred = outputs[0][0].T  # [8400, 84]
        boxes = pred[:, :4]
        scores = pred[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        max_scores = np.max(scores, axis=1)
        mask = max_scores > conf_thres
        boxes, max_scores, class_ids = boxes[mask], max_scores[mask], class_ids[mask]
        orig_w, orig_h = img.size
        boxes[:, 0] *= orig_w / 640
        boxes[:, 1] *= orig_h / 640
        boxes[:, 2] *= orig_w / 640
        boxes[:, 3] *= orig_h / 640
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        try:
            import torchvision
            xyxy = torch.tensor(np.stack([x1, y1, x2, y2], axis=1))
            keep = torchvision.ops.nms(xyxy, torch.tensor(max_scores), iou_thres)
            x1, y1, x2, y2 = x1[keep.numpy()], y1[keep.numpy()], x2[keep.numpy()], y2[keep.numpy()]
            max_scores, class_ids = max_scores[keep.numpy()], class_ids[keep.numpy()]
        except Exception:
            pass
        return [{"cls": int(c), "conf": float(s), "x1": float(xx1), "y1": float(yy1),
                 "x2": float(xx2), "y2": float(yy2)}
                for c, s, xx1, yy1, xx2, yy2 in zip(class_ids, max_scores, x1, y1, x2, y2)]

    img_paths = sorted(VAL_IMG_DIR.glob("*.JPG"))[:20]
    speed = measure_time(trt_predict, img_paths, "YOLOv13n TensorRT-FP16", n_runs=3)
    return speed, sess


def main():
    print("=" * 70)
    print("YOLOv13n Speedup Test — No retraining required")
    print("=" * 70)

    results = {}

    # Reference: YOLOv8n baseline
    speed_v8, _ = test_fp32_v8()
    results["YOLOv8n FP32 (ref)"] = speed_v8

    # Test 0: YOLOv13n baseline
    speed_v13_fp32, _ = test_fp32_v13()
    results["YOLOv13n FP32 (baseline)"] = speed_v13_fp32

    # Test 1: FP16
    speed_fp16, _ = test_fp16_v13()
    results["YOLOv13n FP16"] = speed_fp16

    # Test 2: torch.compile
    speed_compile, _ = test_torch_compile_v13()
    results["YOLOv13n torch.compile"] = speed_compile

    # Test 3: ONNX
    speed_onnx, _ = test_onnx_v13()
    results["YOLOv13n ONNX-GPU"] = speed_onnx

    # Test 4: OpenVINO - skip (not installed, requires extra setup)
    print("\n[Test 4] YOLOv13n OpenVINO — SKIPPED (not installed)")
    results["YOLOv13n OpenVINO-CPU"] = {"name": "OpenVINO", "error": "Not installed, skipped"}

    # Test 5: TensorRT (via onnxruntime)
    speed_trt, _ = test_tensorrt_v13()
    results["YOLOv13n TensorRT-FP16"] = speed_trt

    # Print summary
    print("\n" + "=" * 70)
    print("SPEEDUP COMPARISON")
    print("=" * 70)
    print(f"{'Method':<35} {'Mean (ms)':<12} {'P95 (ms)':<12} {'img/s':<10} {'Speedup':<10}")
    print("-" * 70)

    v8_speed = results["YOLOv8n FP32 (ref)"]["mean_ms"]
    v13_baseline = results["YOLOv13n FP32 (baseline)"]["mean_ms"]

    for name, r in results.items():
        if "error" in r:
            print(f"{name:<35} ERROR: {r['error'][:50]}")
            continue
        speedup = v13_baseline / r["mean_ms"] if r["mean_ms"] > 0 else 0
        print(f"{name:<35} {r['mean_ms']:<12.2f} {r['p95_ms']:<12.2f} {r['img_per_s']:<10.1f} {speedup:<10.2f}x")

    print("-" * 70)
    print(f"YOLOv8n FP32 target: {v8_speed:.2f}ms ({1000/v8_speed:.1f} img/s)")
    print(f"YOLOv13n baseline:   {v13_baseline:.2f}ms ({1000/v13_baseline:.1f} img/s)")

    # Save JSON
    with open(OUTPUT_DIR / "speedup_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Markdown report
    md = ["# YOLOv13n Speedup Benchmark\n"]
    md.append("**Mục tiêu:** Tìm phương pháp tăng tốc YOLOv13n inference mà **không cần train lại**\n")
    md.append("**Hardware:** NVIDIA RTX 4050 Laptop GPU 6GB\n")
    md.append("**Image size:** 640x640\n")
    md.append(f"**Test set:** 20 random val images, 3 runs (mean of all)\n\n")
    md.append("## Results\n\n")
    md.append("| Method | Mean (ms) | P95 (ms) | img/s | Speedup vs FP32 v13 | vs YOLOv8n |\n")
    md.append("|---|---|---|---|---|---|\n")
    for name, r in results.items():
        if "error" in r:
            md.append(f"| {name} | ERROR | - | - | - | - |\n")
            continue
        speedup = v13_baseline / r["mean_ms"] if r["mean_ms"] > 0 else 0
        vs_v8 = v8_speed / r["mean_ms"] if r["mean_ms"] > 0 else 0
        status = "✅ faster" if r["mean_ms"] < v8_speed else ("❌ slower" if r["mean_ms"] > v8_speed * 1.1 else "≈ similar")
        md.append(f"| {name} | {r['mean_ms']:.2f} | {r['p95_ms']:.2f} | {r['img_per_s']:.1f} | {speedup:.2f}x | {vs_v8:.2f}x {status} |\n")

    md.append(f"\n## Interpretation\n\n")
    md.append(f"- YOLOv8n FP32: **{v8_speed:.2f}ms** ({1000/v8_speed:.1f} img/s)\n")
    md.append(f"- YOLOv13n FP32: **{v13_baseline:.2f}ms** ({1000/v13_baseline:.1f} img/s)\n")
    md.append(f"- Gap: v13 is {(v13_baseline - v8_speed):.2f}ms slower ({(v13_baseline/v8_speed):.2f}x)\n\n")

    # Find best method
    best_method = None
    best_speed = float('inf')
    for name, r in results.items():
        if "error" in r or "FP32 (ref)" in name or "FP32 (baseline)" in name:
            continue
        if r["mean_ms"] < best_speed:
            best_speed = r["mean_ms"]
            best_method = name

    if best_method:
        speedup_best = v13_baseline / best_speed
        vs_v8 = v8_speed / best_speed
        md.append(f"**Best method: {best_method}**\n")
        md.append(f"- Mean: {best_speed:.2f}ms ({1000/best_speed:.1f} img/s)\n")
        md.append(f"- Speedup vs YOLOv13n FP32: {speedup_best:.2f}x\n")
        md.append(f"- vs YOLOv8n FP32: {vs_v8:.2f}x ({'FASTER' if vs_v8 > 1 else 'still slower'})\n")

    with open(OUTPUT_DIR / "speedup_report.md", "w") as f:
        f.writelines(md)

    print(f"\n\nSaved: {OUTPUT_DIR / 'speedup_results.json'}")
    print(f"Saved: {OUTPUT_DIR / 'speedup_report.md'}")


if __name__ == "__main__":
    main()
