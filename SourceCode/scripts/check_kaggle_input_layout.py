import argparse
import json
from pathlib import Path


SPLITS = ("train", "val", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def is_split_dir(root: Path, split: str) -> bool:
    return (root / "images" / split).is_dir() and (root / "labels" / split).is_dir()


def is_yolo_dataset(root: Path) -> bool:
    return (root / "images").is_dir() and (root / "labels").is_dir() and all(
        is_split_dir(root, split) for split in ("train", "val")
    )


def image_count(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for item in path.iterdir() if item.is_file() and item.suffix in IMAGE_EXTS)


def label_count(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for item in path.glob("*.txt"))


def split_summary(dataset_root: Path) -> dict:
    summary = {}
    for split in SPLITS:
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        summary[split] = {
            "exists": image_dir.is_dir() and label_dir.is_dir(),
            "images": image_count(image_dir),
            "labels": label_count(label_dir),
        }
    return summary


def scan_input_root(input_root: Path) -> dict:
    dataset_roots = []
    yolov13_repos = []
    weights = []
    zip_files = []

    candidates = [input_root]
    if input_root.exists():
        candidates.extend(path for path in input_root.rglob("*") if path.is_dir())

    for candidate in candidates:
        if is_yolo_dataset(candidate):
            dataset_roots.append(
                {
                    "path": str(candidate),
                    "splits": split_summary(candidate),
                    "has_yaml": any(candidate.glob("*.yaml")),
                }
            )
        if (candidate / "ultralytics" / "__init__.py").exists():
            yolov13_repos.append(str(candidate))

    if input_root.exists():
        weights = [str(path) for path in sorted(input_root.rglob("yolov13n.pt"))]
        zip_files = [str(path) for path in sorted(input_root.rglob("*.zip"))]

    return {
        "input_root": str(input_root),
        "input_root_exists": input_root.exists(),
        "dataset_roots": dataset_roots,
        "yolov13_repos": yolov13_repos,
        "weights": weights,
        "zip_files": zip_files,
    }


def assess(report: dict) -> dict:
    checks = {
        "has_dataset_root": bool(report["dataset_roots"]),
        "has_yolov13_repo": bool(report["yolov13_repos"]),
        "has_yolov13n_weight": bool(report["weights"]),
    }
    blocking = [name for name, ok in checks.items() if not ok]
    status = "KAGGLE_INPUT_READY" if not blocking else "KAGGLE_INPUT_NOT_READY"
    return {"status": status, "checks": checks, "blocking": blocking}


def write_markdown(report: dict, assessment: dict, output_path: Path) -> None:
    lines = [
        "# Kaggle Input Layout Check",
        "",
        f"- Status: `{assessment['status']}`",
        f"- Input root: `{report['input_root']}`",
        "",
        "## Required Items",
        "",
        f"- YOLO dataset root: `{assessment['checks']['has_dataset_root']}`",
        f"- YOLOv13 repo with `ultralytics/__init__.py`: `{assessment['checks']['has_yolov13_repo']}`",
        f"- `yolov13n.pt`: `{assessment['checks']['has_yolov13n_weight']}`",
        "",
        "## Dataset Roots",
        "",
    ]

    if report["dataset_roots"]:
        for dataset in report["dataset_roots"]:
            lines.append(f"### `{dataset['path']}`")
            lines.append("")
            lines.append("| Split | Exists | Images | Labels |")
            lines.append("|---|---:|---:|---:|")
            for split in SPLITS:
                details = dataset["splits"][split]
                lines.append(
                    f"| {split} | {details['exists']} | {details['images']} | {details['labels']} |"
                )
            lines.append("")
    else:
        lines.append("- Không tìm thấy dataset root có cấu trúc `images/train` và `labels/train`.")
        lines.append("")

    lines.extend(
        [
            "## YOLOv13 Repos",
            "",
            *(f"- `{path}`" for path in report["yolov13_repos"]),
            "",
            "## Weights",
            "",
            *(f"- `{path}`" for path in report["weights"]),
            "",
            "## Zip Files",
            "",
            *(f"- `{path}`" for path in report["zip_files"]),
            "",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    default_root = Path("/kaggle/input") if Path("/kaggle/input").exists() else Path(".")
    parser = argparse.ArgumentParser(description="Check Kaggle input layout before running YOLOv13 training.")
    parser.add_argument("--input-root", type=Path, default=default_root)
    parser.add_argument("--output", type=Path, default=Path("datasets/ECUSTFD/reports"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    report = scan_input_root(args.input_root)
    assessment = assess(report)
    full_report = {**report, "assessment": assessment}

    json_path = args.output / "kaggle_input_layout_report.json"
    md_path = args.output / "kaggle_input_layout_summary.md"
    json_path.write_text(json.dumps(full_report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(report, assessment, md_path)

    print(f"Status: {assessment['status']}")
    if assessment["blocking"]:
        print("Blocking items:")
        for item in assessment["blocking"]:
            print(f"  - {item}")
    print(f"Report: {json_path}")
    print(f"Summary: {md_path}")


if __name__ == "__main__":
    main()
