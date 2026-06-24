import argparse
import json
from pathlib import Path


SPLITS = ("train", "val", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
EXPECTED_CLASSES = [
    "apple",
    "banana",
    "bread",
    "bun",
    "doughnut",
    "egg",
    "fired_dough_twist",
    "grape",
    "lemon",
    "litchi",
    "mango",
    "mooncake",
    "orange",
    "peach",
    "pear",
    "plum",
    "qiwi",
    "sachima",
    "tomato",
    "coin",
]
REQUIRED_NOTEBOOK_TOKENS = {
    "--no-deps": "offline pip install guard",
    "pt_working": "copy weights to Kaggle working directory",
    "model.train(": "training cell",
    "best_model.val": "validation/test evaluation cell",
    "best_model.predict": "prediction cell",
    "results.csv": "training results export",
    "output_dir": "Kaggle output export directory",
}


def status(ok: bool, details: str) -> dict:
    return {"ok": ok, "details": details}


def list_image_stems(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {item.stem for item in path.iterdir() if item.is_file() and item.suffix in IMAGE_EXTS}


def list_label_stems(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {item.stem for item in path.glob("*.txt")}


def check_dataset(dataset_dir: Path) -> dict:
    checks = {}
    checks["dataset_dir"] = status(dataset_dir.exists(), str(dataset_dir))

    split_summaries = {}
    all_splits_ok = True
    for split in SPLITS:
        image_dir = dataset_dir / "images" / split
        label_dir = dataset_dir / "labels" / split
        image_stems = list_image_stems(image_dir)
        label_stems = list_label_stems(label_dir)
        missing_labels = sorted(image_stems - label_stems)
        missing_images = sorted(label_stems - image_stems)
        split_ok = image_dir.exists() and label_dir.exists() and not missing_labels and not missing_images
        all_splits_ok = all_splits_ok and split_ok
        split_summaries[split] = {
            "ok": split_ok,
            "images": len(image_stems),
            "labels": len(label_stems),
            "missing_labels": missing_labels[:20],
            "missing_images": missing_images[:20],
        }

    checks["splits"] = {"ok": all_splits_ok, "details": split_summaries}
    return checks


def check_yaml(yaml_path: Path) -> dict:
    if not yaml_path.exists():
        return {"yaml_file": status(False, f"Missing {yaml_path}")}

    text = yaml_path.read_text(encoding="utf-8")
    required_tokens = ["train: images/train", "val: images/val", "test: images/test", "nc: 20"]
    missing = [token for token in required_tokens if token not in text]
    missing_classes = [name for name in EXPECTED_CLASSES if name not in text]
    ok = not missing and not missing_classes

    return {
        "yaml_file": status(ok, str(yaml_path)),
        "missing_tokens": missing,
        "missing_classes": missing_classes,
    }


def check_weights(paths: list[Path]) -> dict:
    items = {str(path): path.exists() and path.stat().st_size > 0 for path in paths}
    return {
        "weights": {
            "ok": any(items.values()),
            "details": items,
        }
    }


def check_notebook(notebook_path: Path) -> dict:
    if not notebook_path.exists():
        return {"notebook": status(False, f"Missing {notebook_path}")}

    text = notebook_path.read_text(encoding="utf-8")
    missing = {
        token: description
        for token, description in REQUIRED_NOTEBOOK_TOKENS.items()
        if token not in text
    }

    return {
        "notebook": status(not missing, str(notebook_path)),
        "missing_tokens": missing,
    }


def check_dataset_validation_report(report_path: Path) -> dict:
    if not report_path.exists():
        return {"dataset_validation_report": status(False, f"Missing {report_path}")}

    report = json.loads(report_path.read_text(encoding="utf-8"))
    error_count = report.get("validation_error_count")
    return {
        "dataset_validation_report": status(error_count == 0, f"{error_count} validation errors"),
    }


def check_existing_training_artifacts(runs_dir: Path) -> dict:
    best_weights = sorted(runs_dir.rglob("best.pt")) if runs_dir.exists() else []
    results_csv = sorted(runs_dir.rglob("results.csv")) if runs_dir.exists() else []
    return {
        "existing_training_artifacts": {
            "ok": True,
            "details": {
                "best_pt": [str(path) for path in best_weights],
                "results_csv": [str(path) for path in results_csv],
                "note": "Empty lists are expected before training.",
            },
        }
    }


def flatten_blocking_checks(report: dict) -> list[str]:
    failures = []
    for section, values in report["checks"].items():
        if isinstance(values, dict) and "ok" in values and not values["ok"]:
            failures.append(section)
        elif isinstance(values, dict):
            for key, value in values.items():
                if isinstance(value, dict) and "ok" in value and not value["ok"]:
                    failures.append(f"{section}.{key}")
    return failures


def write_summary(report: dict, output_path: Path) -> None:
    lines = [
        "# Train Readiness Summary",
        "",
        f"- Status: `{report['status']}`",
        f"- Blocking failures: {len(report['blocking_failures'])}",
        "",
        "## Dataset Splits",
        "",
        "| Split | Images | Labels | Missing labels | Missing images |",
        "|---|---:|---:|---:|---:|",
    ]

    split_details = report["checks"]["dataset"]["splits"]["details"]
    for split in SPLITS:
        details = split_details[split]
        lines.append(
            f"| {split} | {details['images']} | {details['labels']} | "
            f"{len(details['missing_labels'])} | {len(details['missing_images'])} |"
        )

    lines.extend(
        [
            "",
            "## Checks",
            "",
        ]
    )
    for section, values in report["checks"].items():
        lines.append(f"- {section}: `{values}`")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether this repo is ready to start Kaggle training.")
    parser.add_argument("--dataset", type=Path, default=Path("datasets/ECUSTFD"))
    parser.add_argument("--notebook", type=Path, default=Path("train_kaggle_v2.ipynb"))
    parser.add_argument("--output", type=Path, default=Path("datasets/ECUSTFD/reports"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    report = {
        "status": "UNKNOWN",
        "checks": {},
        "blocking_failures": [],
    }

    report["checks"]["dataset"] = check_dataset(args.dataset)
    report["checks"]["yaml"] = check_yaml(args.dataset / "ecustfd.yaml")
    report["checks"]["weights"] = check_weights([Path("datasets/yolov13n.pt"), Path("yolov13/yolov13n.pt")])
    report["checks"]["notebook"] = check_notebook(args.notebook)
    report["checks"]["dataset_validation"] = check_dataset_validation_report(
        args.dataset / "reports" / "validation_report.json"
    )
    report["checks"].update(check_existing_training_artifacts(Path("yolov13/runs")))

    report["blocking_failures"] = flatten_blocking_checks(report)
    report["status"] = "READY_FOR_TRAIN" if not report["blocking_failures"] else "NOT_READY"

    report_path = args.output / "train_readiness_report.json"
    summary_path = args.output / "train_readiness_summary.md"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary(report, summary_path)

    print(f"Status: {report['status']}")
    if report["blocking_failures"]:
        print("Blocking failures:")
        for failure in report["blocking_failures"]:
            print(f"  - {failure}")
    print(f"Report: {report_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
