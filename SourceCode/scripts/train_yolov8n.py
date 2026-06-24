"""Train YOLOv8n on ECUSTFD with the same hyperparameters as YOLOv13n.

This is the apples-to-apples baseline for the Phase C ablation. The two
runs must share data, epochs, batch size, image size, seed, and
augmentation so that any difference in downstream calorie metrics can be
attributed to the architecture.

Output
------
runs/local_food_detect/yolov8n_ecustfd_local/
  weights/{best.pt, last.pt}
  results.csv
  args.yaml
  ...other ultralytics artifacts
"""

import argparse
import os
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("datasets/ECUSTFD/ecustfd.yaml"))
    parser.add_argument("--weights", type=Path, default=Path("weights/yolov8n_pretrained.pt"))
    parser.add_argument("--repo", type=Path, default=Path("yolov13"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--name", default="yolov8n_ecustfd_local")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path.resolve()


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def write_local_yaml(source_yaml: Path, output_dir: Path) -> Path:
    import yaml

    data = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    dataset_root = source_yaml.parent.resolve()
    data["path"] = dataset_root.as_posix()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_yaml = output_dir / "ecustfd_local.yaml"
    output_yaml.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return output_yaml


def main() -> None:
    args = parse_args()

    repo_dir = require_file(args.repo / "ultralytics" / "__init__.py", "YOLOv13 repo").parents[1]
    data_yaml = require_file(args.data, "dataset YAML")
    weights = require_file(args.weights, "pretrained weights")

    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(".yolo-config").resolve()))
    os.environ.setdefault("YOLO_OFFLINE", "true")
    data_yaml = write_local_yaml(data_yaml, Path("runs/local_food_detect"))

    import torch
    from ultralytics import YOLO

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Python: {sys.version}")
    print(f"Torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Device: {device}")
    print(f"Data: {data_yaml}")
    print(f"Weights: {weights}")
    print(f"Seed: {args.seed}")

    model = YOLO(str(weights))
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        device=device,
        project="runs/local_food_detect",
        name=args.name,
        pretrained=True,
        save=True,
        plots=True,
        workers=args.workers,
        amp=False,
        cache=False,
        exist_ok=True,
        verbose=True,
        seed=args.seed,
        deterministic=True,
    )

    run_dir = Path("runs/local_food_detect") / args.name
    best_model_path = run_dir / "weights" / "best.pt"
    if not best_model_path.exists():
        raise FileNotFoundError(f"Training finished but best.pt was not found: {best_model_path}")

    best_model = YOLO(str(best_model_path))
    print("\n=== Validation ===")
    val_metrics = best_model.val(data=str(data_yaml), split="val")
    print(f"mAP50: {val_metrics.box.map50:.4f}")
    print(f"mAP50-95: {val_metrics.box.map:.4f}")
    print(f"Precision: {val_metrics.box.mp:.4f}")
    print(f"Recall: {val_metrics.box.mr:.4f}")

    print("\n=== Test ===")
    test_metrics = best_model.val(data=str(data_yaml), split="test")
    print(f"mAP50: {test_metrics.box.map50:.4f}")
    print(f"mAP50-95: {test_metrics.box.map:.4f}")
    print(f"Precision: {test_metrics.box.mp:.4f}")
    print(f"Recall: {test_metrics.box.mr:.4f}")

    # Mirror the YOLOv13n layout so downstream eval is identical.
    output_dir = Path("runs/local_food_detect/output_yolov8n")
    output_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "weights").exists():
        shutil.copytree(run_dir / "weights", output_dir / "weights", dirs_exist_ok=True)
    for filename in [
        "results.csv",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "labels.jpg",
        "labels_correlogram.jpg",
    ]:
        copy_if_exists(run_dir / filename, output_dir / filename)

    print(f"\n=== Output saved to: {output_dir.resolve()} ===")
    return results


if __name__ == "__main__":
    main()
