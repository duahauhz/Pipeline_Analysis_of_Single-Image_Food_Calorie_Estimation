import importlib
import json
import os
import sys
from pathlib import Path


REQUIRED_MODULES = {
    "torch": "torch",
    "ultralytics": "ultralytics",
    "cv2": "opencv-python",
    "PIL": "pillow",
    "thop": "ultralytics-thop",
    "timm": "timm",
    "safetensors": "safetensors",
    "huggingface_hub": "huggingface_hub",
}


def check_module(import_name: str) -> dict:
    try:
        module = importlib.import_module(import_name)
        return {
            "ok": True,
            "version": getattr(module, "__version__", "unknown"),
            "file": getattr(module, "__file__", "built-in"),
        }
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


def file_ok(path: Path) -> dict:
    return {"ok": path.exists(), "path": str(path), "size": path.stat().st_size if path.exists() else 0}


def main() -> None:
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(".yolo-config").resolve()))
    repo_path = Path("yolov13").resolve()
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    report = {
        "python": {
            "version": sys.version,
            "version_info": list(sys.version_info[:3]),
            "supported_for_torch": (3, 10) <= sys.version_info[:2] <= (3, 12),
        },
        "files": {
            "dataset_yaml": file_ok(Path("datasets/ECUSTFD/ecustfd.yaml")),
            "weights": file_ok(Path("weights/yolov13n_pretrained.pt")),
            "yolov13_repo": file_ok(Path("yolov13/ultralytics/__init__.py")),
        },
        "modules": {},
        "cuda": {"ok": False, "details": "torch is not importable"},
        "status": "UNKNOWN",
        "blocking": [],
    }

    for import_name in REQUIRED_MODULES:
        report["modules"][import_name] = check_module(import_name)

    if report["modules"]["torch"].get("ok"):
        import torch

        cuda_ok = torch.cuda.is_available()
        report["cuda"] = {
            "ok": cuda_ok,
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0) if cuda_ok else None,
            "torch_cuda": getattr(torch.version, "cuda", None),
        }

    if not report["python"]["supported_for_torch"]:
        report["blocking"].append("Use Python 3.10-3.12 for PyTorch.")
    for name, result in report["files"].items():
        if not result["ok"]:
            report["blocking"].append(f"Missing file: {name}")
    for import_name, result in report["modules"].items():
        if not result["ok"]:
            report["blocking"].append(
                f"Missing module {import_name} (install package {REQUIRED_MODULES[import_name]})"
            )
    if not report["cuda"]["ok"]:
        report["blocking"].append("CUDA is not available from torch.")

    report["status"] = "LOCAL_TRAIN_READY" if not report["blocking"] else "LOCAL_TRAIN_NOT_READY"

    output_dir = Path("datasets/ECUSTFD/reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "local_train_readiness_report.json"
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Status: {report['status']}")
    for item in report["blocking"]:
        print(f"- {item}")
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
