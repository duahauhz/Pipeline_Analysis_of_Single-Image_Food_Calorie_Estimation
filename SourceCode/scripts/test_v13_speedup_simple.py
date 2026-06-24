"""
Test tối ưu inference YOLOv13n - đơn giản hóa, chạy tuần tự.
Focus: FP16, ONNX export, onnxruntime-gpu, onnxruntime TensorRT.
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
OUTPUT_DIR = PROJECT_ROOT / "runs" / "v13_speedup"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(PROJECT_ROOT / "datasets" / "ECUSTFD" / "ecustfd.yaml") as f:
    cfg = yaml.safe_load(f)
NAMES = cfg["names"]


def time_predict(fn, img_paths, name, n_runs=3, **kwargs):
    """Time a predict function."""
    print(f"  Warming up {name}...")
    for p in img_paths[:3]:
        fn(str(p), **kwargs)
    torch.cuda.synchronize()

    print(f"  Measuring {name}...")
    latencies = []
    for run in range(n_runs):
        run_times = []
        for p in img_paths:
            t0 = time.perf_counter()
            fn(str(p), **kwargs)
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


def main():
    print("=" * 70)
    print("YOLOv13n Speedup — Simple Test")
    print("=" * 70)

    img_paths = sorted(VAL_IMG_DIR.glob("*.JPG"))[:20]

    results = {}

    # ===== Test 0: YOLOv8n reference =====
    print("\n[Reference] YOLOv8n FP32")
    from ultralytics import YOLO
    v8 = YOLO(str(V8_WEIGHTS))
    results["YOLOv8n FP32"] = time_predict(v8, img_paths, "YOLOv8n FP32", n_runs=3,
                                            verbose=False, device="cuda:0", imgsz=640, half=False)
    print(f"  → {results['YOLOv8n FP32']['mean_ms']:.2f}ms ({results['YOLOv8n FP32']['img_per_s']:.1f} img/s)")
    del v8
    torch.cuda.empty_cache()

    # ===== Test 1: YOLOv13n FP32 =====
    print("\n[Test 1] YOLOv13n FP32 (baseline)")
    v13 = YOLO(str(V13_WEIGHTS))
    results["YOLOv13n FP32"] = time_predict(v13, img_paths, "YOLOv13n FP32", n_runs=3,
                                             verbose=False, device="cuda:0", imgsz=640, half=False)
    print(f"  → {results['YOLOv13n FP32']['mean_ms']:.2f}ms ({results['YOLOv13n FP32']['img_per_s']:.1f} img/s)")
    del v13
    torch.cuda.empty_cache()

    # ===== Test 2: YOLOv13n FP16 =====
    print("\n[Test 2] YOLOv13n FP16 (half=True)")
    v13_fp16 = YOLO(str(V13_WEIGHTS))
    try:
        results["YOLOv13n FP16"] = time_predict(v13_fp16, img_paths, "YOLOv13n FP16", n_runs=3,
                                                 verbose=False, device="cuda:0", imgsz=640, half=True)
        print(f"  → {results['YOLOv13n FP16']['mean_ms']:.2f}ms ({results['YOLOv13n FP16']['img_per_s']:.1f} img/s)")
    except Exception as e:
        print(f"  FAILED: {e}")
        results["YOLOv13n FP16"] = {"name": "YOLOv13n FP16", "error": str(e)}
    del v13_fp16
    torch.cuda.empty_cache()

    # ===== Test 3: YOLOv13n with imgsz=512 (smaller) =====
    print("\n[Test 3] YOLOv13n FP32 imgsz=512 (smaller input)")
    v13_512 = YOLO(str(V13_WEIGHTS))
    try:
        results["YOLOv13n FP32 imgsz=512"] = time_predict(v13_512, img_paths, "YOLOv13n imgsz=512", n_runs=3,
                                                            verbose=False, device="cuda:0", imgsz=512, half=False)
        print(f"  → {results['YOLOv13n FP32 imgsz=512']['mean_ms']:.2f}ms")
    except Exception as e:
        print(f"  FAILED: {e}")
        results["YOLOv13n FP32 imgsz=512"] = {"name": "YOLOv13n imgsz=512", "error": str(e)}
    del v13_512
    torch.cuda.empty_cache()

    # ===== Test 4: YOLOv13n FP16 + imgsz=512 =====
    print("\n[Test 4] YOLOv13n FP16 imgsz=512 (combined)")
    v13_512_fp16 = YOLO(str(V13_WEIGHTS))
    try:
        results["YOLOv13n FP16 imgsz=512"] = time_predict(v13_512_fp16, img_paths, "YOLOv13n FP16+512", n_runs=3,
                                                            verbose=False, device="cuda:0", imgsz=512, half=True)
        print(f"  → {results['YOLOv13n FP16 imgsz=512']['mean_ms']:.2f}ms")
    except Exception as e:
        print(f"  FAILED: {e}")
        results["YOLOv13n FP16 imgsz=512"] = {"name": "YOLOv13n FP16+512", "error": str(e)}
    del v13_512_fp16
    torch.cuda.empty_cache()

    # ===== Test 5: ONNX export =====
    print("\n[Test 5] YOLOv13n ONNX export")
    onnx_path = OUTPUT_DIR / "v13n.onnx"
    if not onnx_path.exists():
        print("  Exporting...")
        v13_exp = YOLO(str(V13_WEIGHTS))
        try:
            v13_exp.export(format="onnx", imgsz=640, simplify=False, dynamic=True, half=False, opset=18)
            # Find exported file
            for cand in [V13_WEIGHTS.parent, PROJECT_ROOT]:
                for f in cand.glob("*.onnx"):
                    if f.stat().st_mtime > time.time() - 60:
                        import shutil
                        shutil.move(str(f), str(onnx_path))
                        break
                if onnx_path.exists():
                    break
        except Exception as e:
            print(f"  Export FAILED: {e}")
        del v13_exp
        torch.cuda.empty_cache()

    if onnx_path.exists():
        print(f"  ONNX file: {onnx_path.stat().st_size/1024/1024:.2f}MB")

        # ONNX + onnxruntime-gpu (CUDA EP)
        print("\n[Test 5a] ONNX + onnxruntime-gpu (CUDA EP)")
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(str(onnx_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            print(f"  Active providers: {sess.get_providers()}")
            input_name = sess.get_inputs()[0].name
            output_names = [o.name for o in sess.get_outputs()]
            print(f"  Input: {sess.get_inputs()[0]}")
            print(f"  Outputs: {[o.name for o in sess.get_outputs()]}")

            def onnx_predict(img_path, conf_thres=0.25, iou_thres=0.45):
                img = Image.open(img_path).convert("RGB")
                img_resized = img.resize((640, 640))
                img_array = np.array(img_resized, dtype=np.float32) / 255.0
                img_array = img_array.transpose(2, 0, 1)[None]
                outputs = sess.run(output_names, {input_name: img_array})
                return outputs[0]

            # Just measure pre+inference (not NMS)
            results["YOLOv13n ONNX-GPU"] = time_predict(onnx_predict, img_paths, "ONNX-GPU", n_runs=3)
            print(f"  → {results['YOLOv13n ONNX-GPU']['mean_ms']:.2f}ms")
            del sess
        except Exception as e:
            print(f"  FAILED: {e}")
            results["YOLOv13n ONNX-GPU"] = {"name": "ONNX-GPU", "error": str(e)}
        torch.cuda.empty_cache()

        # ONNX + onnxruntime TensorRT EP
        print("\n[Test 5b] ONNX + onnxruntime TensorRT EP")
        try:
            import onnxruntime as ort
            providers = [
                ("TensorrtExecutionProvider", {
                    "trt_fp16_enable": True,
                    "trt_max_workspace_size": 2 * 1024 * 1024 * 1024,
                }),
                "CUDAExecutionProvider",
            ]
            sess_trt = ort.InferenceSession(str(onnx_path), providers=providers)
            print(f"  Active providers: {sess_trt.get_providers()}")
            input_name = sess_trt.get_inputs()[0].name
            output_names = [o.name for o in sess_trt.get_outputs()]

            def trt_predict(img_path):
                img = Image.open(img_path).convert("RGB").resize((640, 640))
                img_array = np.array(img, dtype=np.float32) / 255.0
                img_array = img_array.transpose(2, 0, 1)[None]
                outputs = sess_trt.run(output_names, {input_name: img_array})
                return outputs[0]

            results["YOLOv13n TensorRT-EP"] = time_predict(trt_predict, img_paths, "TensorRT-EP", n_runs=3)
            print(f"  → {results['YOLOv13n TensorRT-EP']['mean_ms']:.2f}ms")
            del sess_trt
        except Exception as e:
            print(f"  FAILED: {e}")
            results["YOLOv13n TensorRT-EP"] = {"name": "TensorRT-EP", "error": str(e)}
        torch.cuda.empty_cache()

    # ===== Summary =====
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    v8_speed = results["YOLOv8n FP32"]["mean_ms"]
    v13_fp32 = results["YOLOv13n FP32"]["mean_ms"]
    print(f"{'Method':<35} {'Mean (ms)':<12} {'img/s':<10} {'vs v13 FP32':<12} {'vs v8':<10}")
    print("-" * 80)
    for name, r in results.items():
        if "error" in r:
            print(f"{name:<35} ERROR: {r['error'][:50]}")
            continue
        speedup_v13 = v13_fp32 / r["mean_ms"] if r["mean_ms"] > 0 else 0
        vs_v8 = v8_speed / r["mean_ms"] if r["mean_ms"] > 0 else 0
        status = "FASTER" if r["mean_ms"] < v8_speed else ("similar" if r["mean_ms"] < v8_speed * 1.1 else "slower")
        print(f"{name:<35} {r['mean_ms']:<12.2f} {r['img_per_s']:<10.1f} {speedup_v13:<12.2f}x {vs_v8:.2f}x {status}")

    print(f"\nYOLOv8n FP32 target: {v8_speed:.2f}ms ({1000/v8_speed:.1f} img/s)")
    print(f"YOLOv13n FP32:      {v13_fp32:.2f}ms ({1000/v13_fp32:.1f} img/s)")
    print(f"Gap: v13 is {(v13_fp32 - v8_speed):.2f}ms slower ({(v13_fp32/v8_speed):.2f}x)")

    with open(OUTPUT_DIR / "speedup_simple.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nSaved: {OUTPUT_DIR / 'speedup_simple.json'}")


if __name__ == "__main__":
    main()
